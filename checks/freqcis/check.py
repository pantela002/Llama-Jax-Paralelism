import torch
import numpy as np
import torch
import jax.numpy as jnp

from jax_llama.model import precompute_freqs_cis  # ← your JAX RoPE freq function


# official HF implementation
def precompute_freqs_cis_hf_git(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def compare_freqs_cis(dim: int = 128, seq_len: int = 5, theta: float = 10000.0):
    flax_freqs = precompute_freqs_cis(dim=dim, end=seq_len, theta=theta)
    
    # HF_git
    hf_freq_fromgitcode = precompute_freqs_cis_hf_git(dim=dim, end=seq_len, theta=theta)
    
    # Convert to numpy
    flax_real = np.asarray(jnp.real(flax_freqs))
    flax_imag = np.asarray(jnp.imag(flax_freqs))

    hf_real_fromgit = hf_freq_fromgitcode.real.numpy()
    hf_imag_fromgit = hf_freq_fromgitcode.imag.numpy()
    
    # Diff from HF git code
    real_diff_git = np.max(np.abs(flax_real - hf_real_fromgit))
    imag_diff_git = np.max(np.abs(flax_imag - hf_imag_fromgit))

    print("Flax real[:2, :4]:\n", flax_real[:2, :4])
    print("Flax imag[:2, :4]:\n", flax_imag[:2, :4])
    print("HF git real[:2, :4]:\n", hf_real_fromgit[:2, :4])
    print("HF git imag[:2, :4]:\n", hf_imag_fromgit[:2, :4])


    print(f"✅ Max real diff (HF git code): {real_diff_git:.6e}")
    print(f"✅ Max imag diff (HF git code): {imag_diff_git:.6e}")


if __name__ == "__main__":
    compare_freqs_cis()