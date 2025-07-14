import triton
import triton.language as tl
import torch


# (batch, head, seq, head_dim)
@triton.jit
def _window_bwd_kernel(
    Q,
    K,
    V,
    bias,
    d_O,
    d_Q,
    d_K,
    d_V,
    d_bias,
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

    # compute d_V, d_attn
    d_attn = tl.zeros((seq_pad, seq_pad), dtype=tl.float32)
    d_O_ptr = tl.make_block_ptr(
        base=d_O + offset,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    V_ptr = tl.make_block_ptr(
        base=V + offset,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )

    index = offset + tl.arange(0, seq_pad)[:, None] * head_dim + tl.arange(0, chunk_dim)[None, :]
    d_V_ptr = d_V + index

    for _ in range(head_chunk):
        # load data
        d_o_data = tl.load(d_O_ptr, boundary_check=(0, 1), padding_option="zero")
        v_data = tl.load(V_ptr, boundary_check=(0, 1), padding_option="zero")

        # accumulate d_attn
        d_attn = tl.dot(d_o_data, v_data.trans(1, 0), d_attn)
        d_v_data = tl.dot(attn.trans(1, 0), d_o_data).cast(Q.dtype.element_ty)
        tl.store(d_V_ptr, d_v_data, mask=mask[:, None])

        d_O_ptr = tl.advance(d_O_ptr, (0, chunk_dim))
        V_ptr = tl.advance(V_ptr, (0, chunk_dim))
        d_V_ptr += chunk_dim

    d_attn = d_attn.cast(Q.dtype.element_ty)
    attn_sum = tl.sum(attn * d_attn, axis=1, keep_dims=True)
    attn_sum = attn_sum.cast(Q.dtype.element_ty)
    d_attn = attn * (d_attn - attn_sum)

    # compute d_bias
    if bias is not None:
        # (head, seq, seq)
        index_bias = head_id * seq * seq + tl.arange(0, seq_pad)[:, None] * seq + tl.arange(0, seq_pad)[None, :]
        d_Bias_ptr = d_bias + index_bias
        tl.atomic_add(d_Bias_ptr, d_attn, mask=(mask[:, None] & mask[None, :]))

    # compute d_Q, d_K
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
    d_Q_ptr = d_Q + index
    d_K_ptr = d_K + index

    for _ in range(head_chunk):
        # load data
        q_data = tl.load(Q_ptr, boundary_check=(0, 1), padding_option="zero")
        k_data = tl.load(K_ptr, boundary_check=(0, 1), padding_option="zero")

        d_q_data = tl.dot(d_attn, k_data).cast(q_data.dtype)
        tl.store(d_Q_ptr, d_q_data, mask=mask[:, None])

        d_k_data = tl.dot(d_attn.trans(1, 0), q_data).cast(k_data.dtype)
        tl.store(d_K_ptr, d_k_data, mask=mask[:, None])

        Q_ptr = tl.advance(Q_ptr, (0, chunk_dim))
        K_ptr = tl.advance(K_ptr, (0, chunk_dim))
        d_Q_ptr += chunk_dim
        d_K_ptr += chunk_dim
