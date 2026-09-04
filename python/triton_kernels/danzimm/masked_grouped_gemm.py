"""Masked+grouped GEMM; repro for conservative s_waitcnt vmcnt(0) in main loop.

Reproduce with:
    python masked_grouped_gemm.py --num-groups 8 --M 40000 --N 4096 --K 4096 \
        --block-n 256

Include new waitcnt optimization by also passing --enable-waitcnt-padding
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import torch
import triton
import triton.language as tl


_BUFFER_LOAD_OPS = ("amdg.buffer_load", "amdg.buffer_load_to_local")


def matmul_tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    return 2.0 * m * n * k / (latency_ms * 1e9)


def _uses_buffer_ops(ttgir: str, pointer_name: str) -> bool:
    return any(f"{op} %{pointer_name}[" in ttgir for op in _BUFFER_LOAD_OPS)


def format_buffer_ops_usage(ttgir: str | None) -> str:
    if ttgir is None:
        return (
            "buffer ops: a=unknown b=unknown "
            "(final TTGIR unavailable; ensure TRITON_STORE_BINARY_ONLY is unset)"
        )

    a_status = "used" if _uses_buffer_ops(ttgir, "a_ptr") else "not-used"
    b_status = "used" if _uses_buffer_ops(ttgir, "b_ptr") else "not-used"
    return f"buffer ops: a={a_status} b={b_status} (source=final TTGIR)"


def assert_grouped_result_close(
    a: torch.Tensor,
    b: torch.Tensor,
    actual: torch.Tensor,
    split_sizes: Sequence[int],
    block_k: int,
    mask_other: float,
) -> None:
    k_padding = (-a.shape[1]) % block_k
    reference_b = torch.nn.functional.pad(b, (0, k_padding), value=mask_other)

    row_start = 0
    for group_index, m_size in enumerate(split_sizes):
        row_end = row_start + m_size
        reference_a = torch.nn.functional.pad(
            a[row_start:row_end], (0, k_padding), value=mask_other
        )
        expected = torch.matmul(reference_a, reference_b.T)
        try:
            torch.testing.assert_close(
                actual[row_start:row_end],
                expected,
                rtol=1e-2,
                atol=1e-2,
            )
        except AssertionError as error:
            raise AssertionError(
                f"grouped GEMM mismatch in group {group_index} "
                f"(rows [{row_start}, {row_end})):\n{error}"
            ) from error
        row_start = row_end


def _print_performance_report(
    compiled_kernel: Any,
    latency_p20_ms: float,
    latency_p50_ms: float,
    latency_p80_ms: float,
    m: int,
    n: int,
    k: int,
) -> None:
    print(
        f"performance: latency_ms_p20={latency_p20_ms:.3f} "
        f"latency_ms_p50={latency_p50_ms:.3f} "
        f"latency_ms_p80={latency_p80_ms:.3f} "
        f"tflops_p50={matmul_tflops(m, n, k, latency_p50_ms):.2f}"
    )
    ttgir = compiled_kernel.asm.get("ttgir")
    print(format_buffer_ops_usage(ttgir if isinstance(ttgir, str) else None))


@triton.jit
def grouped_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    split_sizes_ptr,
    G: tl.constexpr,
    N,
    K,
    NUM_SMS,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MASK_OTHER: tl.constexpr,
):
    # Mirrors the real grouped-GEMM fprop control flow: outer group loop over G
    # experts with a data-dependent m_size, a participation guard, a persistent
    # inner tile loop, and loop-carried jagged offsets — wrapping the async K-loop.
    tile_idx = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    start_am = 0
    tile_start = 0
    for g in range(G):
        m_size = tl.load(split_sizes_ptr + g).to(tl.int32)
        num_m = tl.cdiv(m_size, BLOCK_M)
        num_tiles = num_m * num_n
        tile_end = tile_start + num_tiles
        if (tile_idx >= tile_start) and (tile_idx < tile_end):
            num_iter = tl.cdiv(tile_end - tile_idx, NUM_SMS)
            for _ in range(num_iter):
                cur = tile_idx - tile_start
                tm = cur // num_n
                tn = cur % num_n
                offs_m = tm * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = tn * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)
                # B is transposed [N, K] and dotted as b.T (matches real grouped-GEMM);
                # loops use tl.range (scheduling hints) like the real kernel.
                a_ptrs = (
                    a_ptr
                    + (start_am + offs_m[:, None]) * stride_am
                    + offs_k[None, :] * stride_ak
                )
                b_ptrs = (
                    b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
                )
                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    kmask = offs_k[None, :] < K - k * BLOCK_K
                    a = tl.load(
                        a_ptrs,
                        mask=(offs_m[:, None] < m_size) & kmask,
                        other=MASK_OTHER,
                    )
                    b = tl.load(
                        b_ptrs, mask=(offs_n[:, None] < N) & kmask, other=MASK_OTHER
                    )
                    acc += tl.dot(a, b.T, allow_tf32=False, out_dtype=tl.float32)
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk
                c = acc.to(tl.bfloat16)
                c_ptrs = (
                    c_ptr
                    + (start_am + offs_m[:, None]) * stride_cm
                    + offs_n[None, :] * stride_cn
                )
                tl.store(
                    c_ptrs, c, mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < N)
                )
                tile_idx += NUM_SMS
        start_am += m_size
        tile_start += num_tiles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--block-m", type=int, default=128)
    ap.add_argument("--block-n", type=int, default=256)
    ap.add_argument("--block-k", type=int, default=64)
    ap.add_argument("--num-stages", type=int, default=3)
    ap.add_argument("--num-warps", type=int, default=8)
    ap.add_argument("--num-sms", type=int, default=4)
    ap.add_argument("--num-groups", type=int, default=4)
    ap.add_argument(
        "--mask-other",
        type=float,
        default=0.0,
        help="masked-load fill value",
    )
    ap.add_argument(
        "--enable-waitcnt-padding",
        action="store_true",
        help="enable the amdgpu-waitcnt-branch-padding LLVM function attribute",
    )
    args = ap.parse_args()

    dev = "cuda"
    G = args.num_groups
    m_per = args.M
    total_m = m_per * G
    split_sizes = torch.full((G,), m_per, device=dev, dtype=torch.int32)
    a = torch.randn((total_m, args.K), device=dev, dtype=torch.bfloat16)
    b = torch.randn(
        (args.N, args.K), device=dev, dtype=torch.bfloat16
    )  # transposed [N, K]
    c = torch.empty((total_m, args.N), device=dev, dtype=torch.bfloat16)
    grid = (args.num_sms,)
    grouped_launch: Any = grouped_kernel[grid]

    def launch_grouped() -> Any:
        return grouped_launch(
            a,
            b,
            c,
            split_sizes,
            G,
            args.N,
            args.K,
            args.num_sms,
            a.stride(0),
            a.stride(1),
            b.stride(1),
            b.stride(0),
            c.stride(0),
            c.stride(1),
            BLOCK_M=args.block_m,
            BLOCK_N=args.block_n,
            BLOCK_K=args.block_k,
            MASK_OTHER=args.mask_other,
            num_stages=args.num_stages,
            num_warps=args.num_warps,
            llvm_fn_attrs=(
                "amdgpu-waitcnt-branch-padding=true"
                if args.enable_waitcnt_padding
                else ""
            ),
        )

    compiled_kernel = launch_grouped()
    assert_grouped_result_close(
        a,
        b,
        c,
        split_sizes.tolist(),
        args.block_k,
        args.mask_other,
    )
    print(
        f"compiled MASKED GROUPED: G={G} m_per={m_per} "
        f"block=({args.block_m},{args.block_n},{args.block_k}) "
        f"stages={args.num_stages} warps={args.num_warps} num_sms={args.num_sms} "
        f"mask_other={args.mask_other} "
        f"waitcnt_padding={args.enable_waitcnt_padding}"
    )
    print(f"correctness: passed {G} groups against PyTorch")
    latency_p20_ms, latency_p50_ms, latency_p80_ms = triton.testing.do_bench(
        launch_grouped,
        warmup=100,
        rep=500,
        quantiles=[0.2, 0.5, 0.8],
    )
    _print_performance_report(
        compiled_kernel,
        float(latency_p20_ms),
        float(latency_p50_ms),
        float(latency_p80_ms),
        total_m,
        args.N,
        args.K,
    )


if __name__ == "__main__":
    main()
