import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    windows = einops.rearrange(x, 'b (h p1) (w p2) c -> (b h w) p1 p2 c', p1=window_size, p2=window_size)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    x = einops.rearrange(windows, '(b h w) p1 p2 c -> b (h p1) (w p2) c', h=H // window_size, w=W // window_size)

    return x


def mha_core(q, k, v, bias, mask, scale_qk):
    # (B, heads, N, C)
    B, heads, N, head_dim = q.size()
    q = q * scale_qk
    attn = q @ k.transpose(-2, -1)
    if bias is not None:
        attn = attn + bias.unsqueeze(0)
    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B // nW, nW, heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, heads, N, N)
        attn = F.softmax(attn, -1)
    else:
        attn = F.softmax(attn, -1)
    return attn @ v
