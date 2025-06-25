import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import torch # type: ignore
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import numpy as np # type: ignore
from flax.core import freeze # type: ignore
from scipy.stats import pearsonr # type: ignore
from pathlib import Path
from typing import Optional
from flax.core import unfreeze, freeze # type: ignore
import torch.distributed as dist # type: ignore
import fairscale.nn.model_parallel.initialize as fs_init # type: ignore

from jax_llama import config
from jax_llama.model import FlaxLLaMAMLP
from jax_llama import hf_model
from jax.sharding import Mesh, PartitionSpec # type: ignore
from jax.experimental.shard_map import shard_map # type: ignore

# Setup mesh (manual tensor parallelism over 'mp' axis)
devices = jax.devices()
mesh = Mesh(devices, axis_names=('mp',))

# === Distributed init ===
if not dist.is_initialized():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    
    fs_init.initialize_model_parallel(1)


# === Model config ===
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = 8
    vocab_size: int = 128256
    multiple_of: int = 1024
    ffn_dim_multiplier: Optional[float] = 1.3
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_batch_size: int = 1
    max_seq_len: int = 2048
    hidden_dim: Optional[int] = 14336
    
    def transformers_config(self) -> config.LLaMAConfig:

        return config.LLaMAConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.dim,
            intermediate_size=self.hidden_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            num_key_value_heads=self.n_kv_heads,
            max_position_embeddings=self.max_seq_len,
            rms_norm_eps=self.norm_eps,
            rope_theta=self.rope_theta,
        )


# === Load MLP weights from Meta checkpoint ===
def load_mlp_weights(ckpt_path: str, layer_idx: int = 0):
    ckpt = torch.load(sorted(Path(ckpt_path).glob("*.pth"))[0], map_location="cpu")
    return {
        "w1": ckpt[f"layers.{layer_idx}.feed_forward.w1.weight"].to(torch.float32).numpy(),
        "w2": ckpt[f"layers.{layer_idx}.feed_forward.w2.weight"].to(torch.float32).numpy(),
        "w3": ckpt[f"layers.{layer_idx}.feed_forward.w3.weight"].to(torch.float32).numpy(),
    }


# === Run HF (PyTorch) MLP ===
def test_hf_mlp(args: ModelArgs, ckpt_dir: str, x: np.ndarray, layer_idx: int = 0):
    weights = load_mlp_weights(ckpt_dir, layer_idx)
    torch_mlp = hf_model.FeedForward(
        dim=args.dim,
        hidden_dim=4*args.dim,
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
    )

    torch_mlp.load_state_dict({
        "w1.weight": torch.tensor(weights["w1"]),
        "w2.weight": torch.tensor(weights["w2"]),
        "w3.weight": torch.tensor(weights["w3"]),
    })
    with torch.no_grad():
        torch_out = torch_mlp(torch.tensor(x)).numpy()
    print("HF MLP output shape:", torch_out.shape)
    return torch_out


# === Run Flax (JAX) MLP ===
def test_jax_mlp(args: ModelArgs, ckpt_dir: str, x: np.ndarray, layer_idx: int = 0):
    weights = load_mlp_weights(ckpt_dir, layer_idx)

    jax_mlp = FlaxLLaMAMLP(args.transformers_config(), precision='highest')
    dummy_params = jax_mlp.init(jax.random.PRNGKey(0), jnp.asarray(x))
    params_unfrozen = unfreeze(dummy_params)
    params_unfrozen["params"]["w1"]["kernel"] = jnp.asarray(weights["w1"].T)
    params_unfrozen["params"]["w2"]["kernel"] = jnp.asarray(weights["w2"].T)
    params_unfrozen["params"]["w3"]["kernel"] = jnp.asarray(weights["w3"].T)
    jax_params = freeze(params_unfrozen)
    print("JAX MLP params:", jax.tree.map(lambda x: x.shape, jax_params))
    with mesh:
        jax_out = jax_mlp.apply(jax_params, jnp.asarray(x), deterministic=True)  
    print("Sharding spec:", jax.tree.map(lambda x: getattr(x, 'sharding', 'none'), jax_out))
    print("JAX MLP output shape:", jax_out.shape)
    return np.asarray(jax_out)


# === Run test ===
if __name__ == "__main__":
    args = ModelArgs()
    x = np.random.randn(args.max_batch_size, args.max_seq_len, args.dim).astype(np.float32)

    torch_out = test_hf_mlp(args, "/root/tt/3_1_8b/Llama-Jax-Paralelism/llama3.1-8B/8B", x, layer_idx=0)
    jax_out = test_jax_mlp(args, "/root/tt/3_1_8b/Llama-Jax-Paralelism/llama3.1-8B/8B", x, layer_idx=0)

    corr, _ = pearsonr(jax_out.flatten(), torch_out.flatten())
    np.savetxt("jax_out.txt", jax_out.flatten())
    np.savetxt("torch_out.txt", torch_out.flatten())
    print(f"Pearson correlation: {corr:.6f}")
