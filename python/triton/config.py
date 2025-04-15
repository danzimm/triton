from __future__ import annotations

import importlib
import os
import re
import subprocess
import sysconfig

from dataclasses import dataclass
from typing import cast, Any, Callable, Generic, Optional, Protocol, Self, Type, TypeVar, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime.cache import CacheManager, RemoteCacheBackend
    from .runtime.jit import JitFunctionInfo, KernelParam
    from .compiler.compiler import LazyDict


class Env:
    pass


env = Env()


def getenv(key: str) -> Optional[str]:
    res = os.getenv(key)
    return res.strip() if res is not None else res


# There's an asymmetry here so that e.g. env_nvidia_tool can be specified with a
# a string but return an NvidiaTool.
SetType = TypeVar("SetType")
GetType = TypeVar("GetType")


class env_base(Generic[SetType, GetType]):

    def __init__(self, key: str, default: SetType | Callable[[], SetType]) -> None:
        self.key = key
        self.default: Callable[[], SetType] = default if callable(default) else lambda: default

    def __set_name__(self, objclass: Type[object], name: str) -> None:
        self.name = name

    def __get__(self, obj: Optional[object], objclass: Optional[Type[object]]) -> GetType:
        if obj is None:
            raise AttributeError("Cannot access {type(self)} on non-instance")

        if self.name in obj.__dict__:
            return self.transform(obj.__dict__[self.name])
        else:
            return self.get()

    def get(self) -> GetType:
        env = getenv(self.key)
        return self.transform(self.default() if env is None else self.from_env(env))

    def __set__(self, obj: object, value: SetType | Env) -> None:
        if isinstance(value, Env):
            obj.__dict__.pop(self.name, None)
        else:
            obj.__dict__[self.name] = value
            self.set(value)

    def __delete__(self, obj: object) -> None:
        obj.__dict__.pop(self.name, None)

    def transform(self, val: SetType) -> GetType:
        # See comment about GetType/SetType in their definition above. Only needed
        # if GetType != SetType.
        return cast(GetType, val)

    def set(self, val: SetType) -> None:
        pass

    def from_env(self, val: str) -> SetType:
        raise NotImplementedError()


class env_str(env_base[str, str]):

    def set(self, value: Optional[str]) -> None:
        if value is None:
            os.unsetenv(self.key)
        else:
            os.putenv(self.key, value)

    def from_env(self, val: str) -> str:
        return val


class env_bool(env_base[bool, bool]):

    def __init__(self, key: str, default: bool | Callable[[], bool] = False) -> None:
        super().__init__(key, default)

    def from_env(self, val: str) -> bool:
        return val.lower() in ("1", "true", "yes", "on", "y")


class env_int(env_base[int, int]):

    def __init__(self, key: str, default: int | Callable[[], int] = 0) -> None:
        super().__init__(key, default)

    def from_env(self, val: str) -> int:
        try:
            return int(val)
        except ValueError as exc:
            raise RuntimeError(f"Unable to use {self.key}={val}: expected int") from exc


class env_opt_base(Generic[GetType, SetType], env_base[Optional[GetType], Optional[SetType]]):

    def __init__(self, key: str) -> None:
        super().__init__(key, None)


ClassType = TypeVar("ClassType")


class env_class(Generic[ClassType], env_opt_base[Type[ClassType], Type[ClassType]]):

    def __init__(self, key: str, type: str) -> None:
        super().__init__(key)
        # We can't pass the type directly to avoid import cycles
        self.type = type

    def from_env(self, val: str) -> Type[ClassType]:
        comps = val.split(":", 1)
        if len(comps) != 2:
            raise RuntimeError(f"Unable to read {self.key}: '{val}' isn't of the form MODULE:CLASS")
        cls = getattr(importlib.import_module(comps[0]), comps[1])

        if not any((c.__name__ == self.type for c in cls.mro())):
            raise RuntimeError(f"Unable to use '{val}' from {self.key}: not of type '{self.type}'")

        return cast(Type[ClassType], cls)


@dataclass
class NvidiaTool:
    path: str
    version: str

    @staticmethod
    def from_path(path: str) -> NvidiaTool | None:
        try:
            result = subprocess.check_output([path, "--version"], stderr=subprocess.STDOUT)
            if result is None:
                return None
            version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
            if version is None:
                return None
            return NvidiaTool(path, version.group(1))
        except subprocess.CalledProcessError:
            return None


class env_nvidia_tool(env_base[str, NvidiaTool]):

    def __init__(self, binary: str) -> None:
        binary += sysconfig.get_config_var("EXE")
        self.binary = binary
        super().__init__(f"TRITON_{binary.upper()}_PATH", lambda: os.path.join(
            os.path.dirname(__file__),
            "backends",
            "nvidia",
            "bin",
            self.binary,
        ))

    def transform(self, path: str) -> NvidiaTool:
        paths = [
            path,
            # We still add default as fallback in case the pointed binary isn't
            # accessible.
            self.default(),
        ]
        for path in paths:
            if not path or not os.access(path, os.X_OK):
                continue
            if tool := NvidiaTool.from_path(path):
                return tool

        raise RuntimeError(f"Cannot find {self.binary}")

    def from_env(self, val: str) -> str:
        return val


# Separate classes so that types are correct
class env_opt_str(env_opt_base[str, str], env_str):
    pass


class env_opt_bool(env_opt_base[bool, bool], env_bool):
    pass


def get_triton_dir(dirname: str) -> str:
    return os.path.join(
        getenv("TRITON_HOME") or os.path.expanduser("~/"),
        ".triton",
        dirname,
    )


class base_config:

    @property
    def knobs(self) -> dict[str, Any]:
        return {
            k: getattr(self, k)
            # data descriptors live on the class object
            for k, v in type(self).__dict__.items()
            if isinstance(v, env_base)
        }

    def copy(self) -> Self:
        res = type(self)()
        for k, v in self.__dict__.items():
            res.__dict__[k] = v
        return res

    def reset(self) -> Self:
        for knob in self.knobs.keys():
            delattr(self, knob)
        return self


class build_config(base_config):
    """Configuration controlling how the native compiler is invoked"""
    cc: env_opt_str = env_opt_str("CC")

    cudacrt_path: env_opt_str = env_opt_str("TRITON_CUDACRT_PATH")
    cudart_path: env_opt_str = env_opt_str("TRITON_CUDART_PATH")

    @property
    def backend_dirs(self) -> set[str]:
        return {path for path in (self.cudacrt_path, self.cudart_path) if path is not None}


class redis_config(base_config):
    key_format: env_str = env_str("TRITON_REDIS_KEY_FORMAT", "triton:{key}:{filename}")
    host: env_str = env_str("TRITON_REDIS_HOST", "localhost")
    port: env_int = env_int("TRITON_REDIS_PORT", 6379)


class cache_config(base_config):
    dump_dir: env_str = env_str("TRITON_DUMP_DIR", lambda: get_triton_dir("dump"))
    override_dir: env_str = env_str("TRITON_OVERRIDE_DIR", lambda: get_triton_dir("override"))
    dir: env_str = env_str("TRITON_CACHE_DIR", lambda: get_triton_dir("cache"))

    manager_class: env_class[CacheManager] = env_class("TRITON_CACHE_MANAGER", "CacheManager")
    remote_manager_class: env_class[RemoteCacheBackend] = env_class("TRITON_REMOTE_CACHE_BACKEND", "RemoteCacheBackend")


class compilation_config(base_config):
    override: env_bool = env_bool("TRITON_KERNEL_OVERRIDE")
    dump_ir: env_bool = env_bool("TRITON_KERNEL_DUMP")
    store_binary_only: env_bool = env_bool("TRITON_STORE_BINARY_ONLY")
    always_compile: env_bool = env_bool("TRITON_ALWAYS_COMPILE")
    # TODO: Use enum to constrain / 'typecheck' the values
    use_ir_loc: env_opt_str = env_opt_str("USE_IR_LOC")
    enable_asan: env_bool = env_bool("TRITON_ENABLE_ASAN")
    disable_line_info: env_bool = env_bool("TRITON_DISABLE_LINE_INFO")
    front_end_debugging: env_bool = env_bool("TRITON_FRONT_END_DEBUGGING")
    allow_non_constexpr_globals: env_bool = env_bool("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS")


class autotuning_config(base_config):
    cache: env_bool = env_bool("TRITON_CACHE_AUTOTUNING")
    print: env_bool = env_bool("TRITON_PRINT_AUTOTUNING")


class LaunchHook(Protocol):

    def __call__(self, metadata: LazyDict) -> None:
        ...


# This is of the form [attr_name, attr_val]
# TODO: Use tuple instead of list for better typing.
KernelAttr = list[str | int]


class JITHookCompileInfo(TypedDict):
    key: str
    signature: dict[KernelParam, str]
    device: int
    constants: None
    num_warps: int
    num_ctas: int
    num_stages: int
    enable_fp_fusion: bool
    launch_cooperative_grid: bool
    extern_libs: tuple[tuple[str, str], ...]
    configs: list[dict[tuple[int, ...], list[KernelAttr]]]
    specialization_data: str
    is_warmup: bool


class JITHook(Protocol):

    def __call__(self, *, key: str, repr: str, fn: JitFunctionInfo, compile: JITHookCompileInfo, is_manual_warmup: bool,
                 already_compiled: bool) -> Optional[bool]:
        ...


class runtime_config(base_config):
    interpret: env_bool = env_bool("TRITON_INTERPRET")
    debug: env_bool = env_bool("TRITON_DEBUG")
    override_arch: env_opt_str = env_opt_str("TRITON_OVERRIDE_ARCH")

    launch_enter_hook: Optional[LaunchHook] = None
    launch_exit_hook: Optional[LaunchHook] = None

    # Hook for inspecting compiled functions and modules
    jit_cache_hook: Optional[JITHook] = None
    # Hook to signal that a kernel is done compiling and inspect compiled function.
    # jit_cache_hook will always be called before compilation and jit_post_compile_hook after.
    jit_post_compile_hook: Optional[JITHook] = None


class language_config(base_config):
    fp32_default: env_opt_str = env_opt_str("TRITON_F32_DEFAULT")
    default_fp_fusion: env_bool = env_bool("TRITON_DEFAULT_FP_FUSION", True)


class nvidia_config(base_config):
    cuobjdump: env_nvidia_tool = env_nvidia_tool("cuobjdump")
    nvdisasm: env_nvidia_tool = env_nvidia_tool("nvdisasm")
    ptxas: env_nvidia_tool = env_nvidia_tool("ptxas")

    dump_nvptx: env_bool = env_bool("NVPTX_ENABLE_DUMP")
    disable_ptxas_opt: env_bool = env_bool("DISABLE_PTXAS_OPT")
    mock_ptx_version: env_opt_str = env_opt_str("TRITON_MOCK_PTX_VERSION")

    libdevice_path: env_opt_str = env_opt_str("TRITON_LIBDEVICE_PATH")
    libcuda_path: env_opt_str = env_opt_str("TRITON_LIBCUDA_PATH")


class amd_config(base_config):
    use_buffer_ops: env_bool = env_bool("AMDGCN_USE_BUFFER_OPS", True)
    dump_amdgcn: env_bool = env_bool("AMDGCN_ENABLE_DUMP")
    libhip_path: env_opt_str = env_opt_str("TRITON_LIBHIP_PATH")
    lld_path: env_opt_str = env_opt_str("TRITON_HIP_LLD_PATH")

    # We use strs so that we can have a default value based on other runtime info
    use_block_pingpong: env_opt_bool = env_opt_bool("TRITON_HIP_USE_BLOCK_PINGPONG")
    use_in_thread_transpose: env_opt_bool = env_opt_bool("TRITON_HIP_USE_IN_THREAD_TRANSPOSE")

    global_prefetch: env_int = env_int("TRITON_HIP_GLOBAL_PREFETCH")
    local_prefetch: env_int = env_int("TRITON_HIP_GLOBAL_PREFETCH")
    use_async_copy: env_bool = env_bool("TRITON_HIP_GLOBAL_PREFETCH")


class proton_config(base_config):
    cupti_path: env_opt_str = env_opt_str("TRITON_CUPTI_LIB_PATH")


build = build_config()
redis = redis_config()
cache = cache_config()
compilation = compilation_config()
autotuning = autotuning_config()
runtime = runtime_config()
language = language_config()
nvidia = nvidia_config()
amd = amd_config()
proton = proton_config()
