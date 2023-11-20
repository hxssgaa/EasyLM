import time
from typing import Callable

import jax
import jax.numpy as jnp

from EasyLM.flash_attn_utils import flash_attention_implementation, mha_reference, mha_reference2

_BENCHMARK_CONFIGS = {
    "1.2b": dict(
        num_heads=32,
        per_head_dim=64,
    ),
    # "12.6b": dict(
    #     num_heads=40,
    #     per_head_dim=128,
    # ),
    # "29.6b": dict(
    #     num_heads=56,
    #     per_head_dim=128,
    # ),
    # "65.2b": dict(
    #     num_heads=72,
    #     per_head_dim=128,
    # ),
    # "134b": dict(
    #     num_heads=88,
    #     per_head_dim=128,
    # ),
    # "261.7b": dict(
    #     num_heads=110,
    #     per_head_dim=128,
    # ),
   # "539.5b": dict(
   #     num_heads=140,
   #     per_head_dim=128,
   # ),
}


def _time_call(fn: Callable, description, *, num_iters: int = 1) -> float:
    """Times average execution time for fn call after warmup over num_iters."""
    fn().block_until_ready()
    tic = time.perf_counter()
    for _ in range(num_iters):
        jax.debug.print("debug (%s): {x}" % description, x=fn()[1][1][1])
    toc = time.perf_counter()
    return (toc - tic) / num_iters


def _benchmark(
    *,
    batch_size: int,
    seq_len: int,
    block_size: int,
    num_heads: int,
    per_head_dim: int,
    causal: bool = True,
):
    """Benchmarks TPU FlashAttention vs reference impl."""
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16)
    bias = None#jax.random.normal(k4, (batch_size, num_heads, seq_len, seq_len), dtype=jnp.bfloat16)

    softmax_scale = per_head_dim**-0.5
    ref_fwd_time = _time_call(
        lambda: mha_reference(q, k, v, bias, causal=causal, softmax_scale=softmax_scale), 'ref_fwd'
    )

    grad_fn = jax.jit(
        jax.grad(
            lambda q, k, v, b: mha_reference(
                q, k, v, b, causal=causal, softmax_scale=softmax_scale
            ).mean(),
            argnums=(0, 1, 2),
        )
    )
    ref_bwd_time = _time_call(lambda: grad_fn(q, k, v, bias)[0], 'ref_bwd')

    # ref_fwd_time2 = _time_call(
    #     lambda: mha_reference2(q, k, v, bias, causal=causal, softmax_scale=softmax_scale), 'ref_fwd2'
    # )

    # grad_fn2 = jax.jit(
    #     jax.grad(
    #         lambda q, k, v, b: mha_reference2(
    #             q, k, v, b, causal=causal, softmax_scale=softmax_scale
    #         ).mean(),
    #         argnums=(0, 1, 2),
    #     )
    # )
    # ref_bwd_time2 = _time_call(lambda: grad_fn2(q, k, v, bias)[0], 'ref_bwd2')

    # Get fwd & bwd timing information when softmax scaling applied before calling the kernel.
    mha_impl = flash_attention_implementation(
        "tpu", causal=causal, softmax_scale=softmax_scale, block_size=block_size
    )

    flash_fwd_time = _time_call(lambda: mha_impl(q, k, v, bias), 'flash_fwd')

    flash_grad_fn = jax.jit(
        jax.grad(lambda q, k, v, b: mha_impl(q, k, v, b).mean(), argnums=(0, 1, 2))
    )
    flash_bwd_time = _time_call(lambda: flash_grad_fn(q, k, v, bias)[0], 'flash_bwd')

    print(f"ref_fwd:{ref_fwd_time:.4f}s, flash_fwd:{flash_fwd_time:.4f}s") #ref_fwd2:{ref_fwd_time2:.4f}s, 
    print(f"ref_bwd:{ref_bwd_time:.4f}s, flash_bwd:{flash_bwd_time:.4f}s\n") #ref_bwd2:{ref_bwd_time2:.4f}s, 


if __name__ == "__main__":
    assert jax.default_backend() == "tpu", "Benchmarking requires a TPU backend."
    device_kind = jax.devices()[0].device_kind
    for name, cfg in _BENCHMARK_CONFIGS.items():
        print(f"Benchmarking attention representative of {name} model layer on {device_kind}.")
        _benchmark(
            batch_size=2,
            seq_len=4096,
            block_size=4 * 128,
            **cfg,
        )
