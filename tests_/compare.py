import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# Load logits from .npy files
logits1 = np.load("/root/tt/output_hf_unsharded.npy")
logits2 = np.load("/root/tt/output_jax_unsharded.npy")

# Flatten both arrays
logits1 = logits1.flatten()
logits2 = logits2.flatten()

# Ensure shapes match
if logits1.shape != logits2.shape:
    raise ValueError(f"Shape mismatch: {logits1.shape} vs {logits2.shape}")

# Compute Pearson correlation
corr, _ = pearsonr(logits1, logits2)

# Compute Mean Squared Error
mse = mean_squared_error(logits1, logits2)

print(f"✅ Pearson correlation: {corr:.6f}")
print(f"✅ Mean Squared Error: {mse:.6f}")
