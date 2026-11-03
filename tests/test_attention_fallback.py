"""Checks the PyTorch SDPA fallback of utils.attention_layers.flash_attention (used when flash-attn is not
installed) against plain per-sequence softmax attention on padded, variable-length batches. Needs a GPU.

Run from the repo root: python tests/test_attention_fallback.py   (pytest also works)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.attention_layers as attn


def test_sdpa_fallback():
    torch.manual_seed(0)
    flags = attn.FLASH_ATTN_2_AVAILABLE, attn.FLASH_ATTN_3_AVAILABLE
    attn.FLASH_ATTN_2_AVAILABLE = attn.FLASH_ATTN_3_AVAILABLE = False
    try:
        b, lq, lk, h, d = 3, 7, 5, 4, 32
        q, k, v = (torch.randn(b, n, h, d, device='cuda', dtype=torch.bfloat16) for n in (lq, lk, lk))
        q_lens, k_lens = torch.tensor([7, 2, 4], device='cuda'), torch.tensor([5, 5, 3], device='cuda')
        scale = 0.3
        out = attn.flash_attention(q, k, v, q_lens=q_lens, k_lens=k_lens, softmax_scale=scale, unflatten=False)
        ref = torch.cat([torch.softmax(scale * q[i, :q_lens[i]].float().transpose(0, 1) @ k[i, :k_lens[i]].float().permute(1, 2, 0), -1)
                         @ v[i, :k_lens[i]].float().transpose(0, 1) for i in range(b)], dim=1).transpose(0, 1)
        assert out.shape == (int(q_lens.sum()), h, d)
        torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
        full = attn.flash_attention(q, k, v)  # no lengths: [b, lq, h, d]
        assert full.shape == (b, lq, h, d)
    finally:
        attn.FLASH_ATTN_2_AVAILABLE, attn.FLASH_ATTN_3_AVAILABLE = flags


if __name__ == '__main__':
    test_sdpa_fallback()
    print('ok')
