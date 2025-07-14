import triton
import triton.language as tl
import torch


# (batch, head, seq, head_dim)
@triton.jit
def _window_fwd_kernel(
    Q,
    K,
    V,
    bias,
    O,
    scale_qk: tl.constexpr,
    batch: tl.constexpr,
    head: tl.constexpr,
    head_dim: tl.constexpr,
    head_chunk: tl.constexpr,
    chunk_dim: tl.constexpr,
    seq: tl.constexpr,
    seq_pad: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    stride_head = seq * head_dim
    stride_batch = stride_head * head
    offset = batch_id * stride_batch + head_id * stride_head

    # load bias
    if bias is not None:
        # (head, seq, seq)
        Bias_ptr = tl.make_block_ptr(
            base=bias + head_id * seq * seq,
            shape=(seq, seq),
            strides=(seq, 1),
            offsets=(0, 0),
            block_shape=(seq_pad, seq_pad),
            order=(1, 0),
        )
        bias_data = tl.load(Bias_ptr, boundary_check=(0, 1), padding_option="zero")

    mask = tl.arange(0, seq_pad) < seq
    # compute attn matrix
    attn = tl.zeros((seq_pad, seq_pad), dtype=tl.float32)
    Q_ptr = tl.make_block_ptr(
        base=Q + offset,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    K_ptr = tl.make_block_ptr(
        base=K + offset,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    for _ in range(head_chunk):
        # load data
        q_data = tl.load(Q_ptr, boundary_check=(0, 1), padding_option="zero")
        k_data = tl.load(K_ptr, boundary_check=(0, 1), padding_option="zero")
        # dot of bf16 -> fp32
        attn = tl.dot(q_data, k_data.trans(1, 0), attn)
        Q_ptr = tl.advance(Q_ptr, (0, chunk_dim))
        K_ptr = tl.advance(K_ptr, (0, chunk_dim))

    attn *= scale_qk
    if bias is not None:
        attn += bias_data

    attn += tl.where(mask[None, :], 0, -float("inf"))

    # softmax
    attn -= tl.max(attn, axis=1, keep_dims=True)
    attn = tl.math.exp(attn)
    p_sum = tl.sum(attn, axis=1, keep_dims=True)
    attn /= p_sum
    attn = attn.cast(Q.dtype.element_ty)

    # save output
    V_ptr = tl.make_block_ptr(
        base=V + offset,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    index = offset + tl.arange(0, seq_pad)[:, None] * head_dim + tl.arange(0, chunk_dim)[None, :]
    O_ptr = O + index
    for _ in range(head_chunk):
        v_data = tl.load(V_ptr, boundary_check=(0, 1), padding_option="zero")
        o_data = tl.dot(attn, v_data)
        o_data = o_data.cast(Q.dtype.element_ty)
        tl.store(O_ptr, o_data, mask=mask[:, None])
        V_ptr = tl.advance(V_ptr, (0, chunk_dim))
        O_ptr += chunk_dim
