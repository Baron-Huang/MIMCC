from .kernels import _window_fwd_kernel, _window_bwd_kernel
import torch
import triton


HEAD_CHUNK_DIM = 16


def ceil_pow2(x):
    n = x.bit_length() - 1
    if x == (1 << n):
        return x
    else:
        return 1 << (n + 1)


def flash_swin_attn_fwd_func(Q, K, V, bias, scale_qk):
    batch, head, seq, head_dim = Q.size()
    seq_pad = ceil_pow2(seq)
    head_chunk = head_dim // HEAD_CHUNK_DIM
    O = torch.empty_like(Q)

    grid = (batch, head, 1)
    _window_fwd_kernel[grid](
        Q,
        K,
        V,
        bias,
        O,
        scale_qk,
        batch,
        head,
        head_dim,
        head_chunk,
        HEAD_CHUNK_DIM,
        seq,
        seq_pad,
    )

    return O


def flash_swin_attn_bwd_func(Q, K, V, bias, d_O, scale_qk):
    batch, head, seq, head_dim = Q.size()
    seq_pad = ceil_pow2(seq)
    head_chunk = head_dim // HEAD_CHUNK_DIM

    d_Q = torch.empty_like(Q)
    d_K = torch.empty_like(K)
    d_V = torch.empty_like(V)
    d_bias = torch.zeros_like(bias) if bias is not None else None

    grid = (batch, head, 1)
    _window_bwd_kernel[grid](
        Q,
        K,
        V,
        bias,
        d_O,
        d_Q,
        d_K,
        d_V,
        d_bias,
        scale_qk,
        batch,
        head,
        head_dim,
        head_chunk,
        HEAD_CHUNK_DIM,
        seq,
        seq_pad,
    )

    return d_Q, d_K, d_V, d_bias


class FlashSwinFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, scale_qk):
        assert q.size(-1) & 15 == 0, "head_dim must be divisible by 16"
        o = flash_swin_attn_fwd_func(q, k, v, bias, scale_qk)
        ctx.save_for_backward(q, k, v, bias)
        ctx.scale_qk = scale_qk
        return o

    @staticmethod
    def backward(ctx, d_o):
        q, k, v, bias = ctx.saved_tensors
        d_q, d_k, d_v, d_bias = flash_swin_attn_bwd_func(q, k, v, bias, d_o, ctx.scale_qk)

        return d_q, d_k, d_v, d_bias, None


flash_swin_attn_func = FlashSwinFunc.apply
