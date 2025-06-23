import numpy as np
import torch
import jax.numpy as jnp
from pathlib import Path
from transformers import AutoModelForCausalLM
from jax_llama.model import RMSNorm

def load_rmsnorm_weight(ckpt_dir: str, layer_idx: int = 0) -> np.ndarray:
    ckpt_paths = sorted(Path(ckpt_dir).glob("*.pth"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

    print(f"Loading: {ckpt_paths[0]}")
    ckpt = torch.load(ckpt_paths[0], map_location="cpu")
    key = f"layers.{layer_idx}.attention_norm.weight"
    if key not in ckpt:
        raise KeyError(f"Key not found: {key}")
    return ckpt[key].to(torch.float32).numpy()


def compare_rmsnorm_jax_vs_hf(weight: np.ndarray):
    dim = weight.shape[0]
    batch_size = 2
    eps = 1e-5

    # Random input
    x_np = np.random.randn(batch_size, dim).astype(np.float32)

    # JAX RMSNorm
    flax_rms = RMSNorm(dim=dim, eps=eps)
    flax_out = flax_rms.apply({'params': {'kernel': jnp.array(weight)}}, jnp.array(x_np))
    flax_out = np.asarray(flax_out)

    # HF model RMSNorm
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_rms = hf_model.model.layers[0].input_layernorm  # attention_norm of layer 0
    #hf_rms.weight.data = torch.tensor(weight)
    torch_out = hf_rms(torch.tensor(x_np)).detach().numpy()

    # Compare
    print("Flax RMSNorm (first 5 dims):", flax_out[0, :5])
    print("HF RMSNorm   (first 5 dims):", torch_out[0, :5])
    diff = np.max(np.abs(flax_out - torch_out))
    print(f"\n✅ Max RMSNorm difference: {diff:.6f}")


if __name__ == "__main__":
    model_id = "meta-llama/Meta-Llama-3.1-8B"
    ckpt_path = "/root/tt/3_1_8b/llama3.1-8B/8B"
    weight = load_rmsnorm_weight(ckpt_path, layer_idx=0)
    print("Loaded RMSNorm weight shape:", weight.shape)
    compare_rmsnorm_jax_vs_hf(weight)
