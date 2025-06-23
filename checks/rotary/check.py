import torch
import jax
import jax.numpy as jnp
import numpy as np
from jax_llama.model import precompute_freqs_cis, apply_rotary_emb as apply_rotary_emb_jax
from typing import Tuple

def precompute_freqs_cis_hf_git(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb_hf(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def compare_freqs_and_rotary(dim: int = 128, seq_len: int = 5, n_heads: int = 4):
    batch = 2
    head_dim = dim // n_heads
    dtype = jnp.float32
    key = jax.random.PRNGKey(0)

    # Random input
    xq = jax.random.normal(key, (batch, seq_len, n_heads, head_dim), dtype)
    xk = jax.random.normal(key + 1, (batch, seq_len, n_heads, head_dim), dtype)

    # === FREQS_CIS ===
    flax_freqs = precompute_freqs_cis(dim=head_dim, end=seq_len, dtype=dtype)
    hf_freqs = precompute_freqs_cis_hf_git(dim=head_dim, end=seq_len)

    flax_real, flax_imag = np.array(jnp.real(flax_freqs)), np.array(jnp.imag(flax_freqs))
    hf_real, hf_imag = hf_freqs.real.numpy(), hf_freqs.imag.numpy()
    print(f"✅ freqs_cis real diff: {np.max(np.abs(flax_real - hf_real)):.6e}")
    print(f"✅ freqs_cis imag diff: {np.max(np.abs(flax_imag - hf_imag)):.6e}")

    # === ROTARY ===
    xq_out_jax, xk_out_jax = apply_rotary_emb_jax(xq, xk, flax_freqs[None], dtype=dtype)

    xq_torch = torch.tensor(np.array(xq))
    xk_torch = torch.tensor(np.array(xk))
    freqs_cis_torch = hf_freqs.to(torch.complex64)

    xq_out_hf, xk_out_hf = apply_rotary_emb_hf(xq_torch, xk_torch, freqs_cis_torch)

    xq_diff = np.max(np.abs(np.array(xq_out_jax) - xq_out_hf.numpy()))
    xk_diff = np.max(np.abs(np.array(xk_out_jax) - xk_out_hf.numpy()))
    print(f"✅ xq rotary diff: {xq_diff:.6e}")
    print(f"✅ xk rotary diff: {xk_diff:.6e}")

    print("\nℹ️ Sample xq (JAX vs HF):")
    print("JAX:", np.array(xq_out_jax)[0, 0, 0, :5])
    print("HF :", xq_out_hf.numpy()[0, 0, 0, :5])

if __name__ == "__main__":
    compare_freqs_and_rotary()
