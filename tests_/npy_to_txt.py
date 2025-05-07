import numpy as np
import os

# === Path to your .npy file ===
npy_path = "/root/tt/output_hf_unsharded.npy"
txt_path = "/root/tt/output_hf_unsharded.txt"

# === Load the .npy array ===
array = np.load(npy_path)

# === Optional formatting ===
np.set_printoptions(threshold=10000, linewidth=np.inf, precision=6)

# === Save to .txt ===
with open(txt_path, "w") as f:
    f.write(f"shape={array.shape}, dtype={array.dtype}\n")
    f.write(np.array2string(array, separator=', ', max_line_width=np.inf))
    f.write("\n")

print(f"✅ Saved {txt_path}")
