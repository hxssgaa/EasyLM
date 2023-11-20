from typing import Callable, Literal, Optional

import functools
import jax
import jax.numpy as jnp
from absl import logging
from jax import lax
from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes
from EasyLM.tpu_attention import tpu_flash_attention
from flax.linen.attention import dot_product_attention_weights
from flax.linen import combine_masks, make_causal_mask

NEG_INF=-1e15

@functools.partial(jax.jit, static_argnames=["causal", "softmax_scale"])
@jax.default_matmul_precision("bfloat16")
def mha_reference(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias: Optional[jnp.ndarray] = None,
    *,
    causal: bool = False,
    softmax_scale: float = 1.0,
) -> jnp.ndarray:
    """Reference multi-headed attention implementation."""
    # We apply the scale factor before the attention biases.
    q *= softmax_scale
    logits = jnp.einsum("btnh,bsnh->bnts", q, k)

    if bias is not None:
        logits += bias.astype(logits.dtype)

    if causal:
        mask_shape = (q.shape[1], k.shape[1])
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = (row_ids < col_ids)[None, None, :, :]
        logits = jnp.where(causal_mask, NEG_INF, logits)

    logits_dtype = logits.dtype
    logits = logits.astype(jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1).astype(logits_dtype)
    result = jnp.einsum("bnts,bsnh->btnh", probs, v)
    return result


@functools.partial(jax.jit, static_argnames=["causal", "softmax_scale"])
@jax.default_matmul_precision("bfloat16")
def mha_reference2(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias: Optional[jnp.ndarray] = None,
    *,
    causal: bool = False,
    softmax_scale: float = 1.0,
) -> jnp.ndarray:
    """Reference multi-headed attention implementation."""
    # We apply the scale factor before the attention biases.
    causal_mask = make_causal_mask(jnp.ones((1, q.shape[2]), dtype="bool"), dtype="bool")
    batch_size = q.shape[0]
    query_length, key_length = q.shape[1], k.shape[1]
    causal_mask = causal_mask[:, :, :query_length, :key_length]
    import pdb; pdb.set_trace()
    causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
    attention_bias = lax.select(
        causal_mask > 0,
        jnp.full(causal_mask.shape, 0.0).astype(q.dtype),
        jnp.full(causal_mask.shape, jnp.finfo(q.dtype).min).astype(q.dtype),
    )
    if bias is not None:
        attention_bias += bias
    attn_weights = dot_product_attention_weights(
        q,
        k,
        bias=attention_bias,
        dropout_rng=None,
        dropout_rate=0.0,
        deterministic=True,
        dtype=jnp.promote_types(q.dtype, jnp.bfloat16),
        precision="bfloat16",
    )
    attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, v, precision="bfloat16")
    return jnp.swapaxes(attn_output, 1, 2)

# Accepts [query, key, value, attention_bias] tensors and returns the context Tensor.
MultiHeadAttentionImpl = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]


def flash_attention_implementation(
    backend: Literal["cpu", "tpu", "gpu"],
    *,
    causal: bool,
    softmax_scale: float,
    block_size: int = 128,
) -> MultiHeadAttentionImpl:
    """Returns a jitted "flash" multihead-attention implementation for the given backend.

    Args:
        backend: A valid XLA backend name. 'cpu' intended for testing only.
        causal: Whether the attention is causal (allows for additional efficiency optimizations).
        softmax_scale: A scalar value applied to the logits before softmax.
        block_size: The size of the computation-block unit, only applies to the 'tpu' backend.
            A multiple of 128, and should be less than the target sequence length.
            Smaller values are more memory efficient but less compute efficient.

    Returns:
        A jitted function implementing multi-head attention for the given backend.

    Raises:
        NotImplementedError: If implementation for the backend is not available.
    """
    if backend == "tpu":
        block_sizes = BlockSizes(
            block_q=block_size,
            block_k_major=block_size,
            block_k=block_size,
            block_b=1,
            block_q_major_dkv=block_size,
            block_k_major_dkv=block_size,
            block_k_dkv=block_size,
            block_q_dkv=block_size,
            block_k_major_dq=block_size,
            block_k_dq=block_size,
            block_q_dq=block_size,
        )

        # shard_map-decorated function needs to be jitted.
        @jax.jit
        def jit_attn(query, key, value, bias):
            query = jnp.reshape(query, (query.shape[0], query.shape[2], query.shape[1], query.shape[3]))
            key = jnp.reshape(key, query.shape)
            value = jnp.reshape(value, value.shape)
            context = tpu_flash_attention(
                query,
                key,
                value,
                ab=bias,
                causal=causal,
                sm_scale=softmax_scale,
                block_sizes=block_sizes,
            )
            context = jnp.reshape(context, (context.shape[0], context.shape[2], context.shape[1], context.shape[3]))
            return context

        return jit_attn

    elif backend == "cpu":
        logging.warning("Flash attention CPU backend is for testing only.")

        # shard_map-decorated function needs to be jitted.
        @jax.jit
        def jit_attn(query, key, value, bias):
            return mha_reference(
                query, key, value, bias=bias, causal=causal, softmax_scale=softmax_scale
            )

        return jit_attn

    else:
        raise NotImplementedError(f"Backend ({backend}) does not have an implementation.")