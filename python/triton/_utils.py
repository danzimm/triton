from functools import reduce
from typing import TypeVar, Iterable


# Poor man's PyTree

T = TypeVar("T")


def list_list_flatten(x: list[list[T]]) -> tuple[list[int], list[T]]:
    spec = []
    flat = []
    for l in x:
        spec.append(len(l))
        flat.extend(l)
    return spec, flat


def list_list_unflatten(spec: list[int], flat: list[T]) -> list[list[T]]:
    ret = []
    idx = 0
    for size in spec:
        ret.append(flat[idx : idx + size])
        idx += size
    assert idx == len(flat)
    return ret


def get_iterable_path(iterable: Iterable[int], path):
    return reduce(lambda a, idx: a[idx], path, iterable)


def set_iterable_path(iterable, path, val):
    prev = iterable if len(path) == 1 else get_iterable_path(iterable, path[:-1])
    prev[path[-1]] = val


def find_paths_if(iterable, pred):
    from .language import core

    is_iterable = lambda x: isinstance(x, (list, tuple, core.tuple, core.tuple_type))
    ret = dict()

    def _impl(current, path):
        path = (path[0],) if len(path) == 1 else tuple(path)
        if is_iterable(current):
            for idx, item in enumerate(current):
                _impl(item, path + (idx,))
        elif pred(path, current):
            if len(path) == 1:
                ret[(path[0],)] = None
            else:
                ret[tuple(path)] = None

    if is_iterable(iterable):
        _impl(iterable, [])
    elif pred(list(), iterable):
        ret = {tuple(): None}
    else:
        ret = dict()
    return list(ret.keys())


def parse_list_string(s: str) -> list[str]:
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    result = []
    current = ""
    depth = 0
    for c in s:
        if c == "[":
            depth += 1
            current += c
        elif c == "]":
            depth -= 1
            current += c
        elif c == "," and depth == 0:
            result.append(current.strip())
            current = ""
        else:
            current += c
    if current.strip():
        result.append(current.strip())
    return result
