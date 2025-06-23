import numpy as np
from scipy.stats import pearsonr

# Load complex freqs from files
jax_freqs = np.loadtxt("freqs_cis.txt", dtype=np.complex64)
hf_freqs = np.loadtxt("freqs_cis_hf.txt", dtype=np.complex64)

# Ensure shapes match
assert jax_freqs.shape == hf_freqs.shape, "Shape mismatch between JAX and HF freqs"

# Compute correlation for real and imaginary parts separately
real_corr, _ = pearsonr(jax_freqs.real.flatten(), hf_freqs.real.flatten())
imag_corr, _ = pearsonr(jax_freqs.imag.flatten(), hf_freqs.imag.flatten())

print(f"Pearson correlation (real parts): {real_corr:.6f}")
print(f"Pearson correlation (imag parts): {imag_corr:.6f}")
