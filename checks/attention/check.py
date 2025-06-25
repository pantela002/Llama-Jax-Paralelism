import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import torch # type: ignore
import jax  # type: ignore
import jax.numpy as jnp # type: ignore
import numpy as np # type: ignore
from flax.core import freeze, unfreeze # type: ignore
from scipy.stats import pearsonr # type: ignore

from jax.sharding import Mesh, PartitionSpec # type: ignore

from flax.core import freeze, unfreeze # type: ignore
from transformers.models.llama.modeling_llama import LlamaAttention  # type: ignore
from pathlib import Path
from flax.linen import make_causal_mask # type: ignore
from jax_llama import hf_model
from jax_llama.model import FlaxLLaMAAttention  # adjust if your path differs
from typing import Optional, Tuple
import os
import torch.distributed as dist # type: ignore
import fairscale.nn.model_parallel.initialize as fs_init # type: ignore
from jax_llama import config
# Setup mesh (manual tensor parallelism over 'mp' axis)
devices = jax.devices()
mesh = Mesh(devices, axis_names=('mp',))

# Only initialize once
if not dist.is_initialized():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    fs_init.initialize_model_parallel(1)

class ModelArgs:
    dim: int = 4096 #4096
    n_layers: int = 32 # 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = 8
    vocab_size: int = 128256 #128256
    multiple_of: int = 1024
    ffn_dim_multiplier: Optional[float] = 1.3
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_batch_size: int = 1
    max_seq_len: int = 2048

    def transformers_config(self) -> config.LLaMAConfig:
        hidden_dim = int(2 * (4 * self.dim) / 3)
        hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)

        return config.LLaMAConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.dim,
            intermediate_size=hidden_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            num_key_value_heads=self.n_kv_heads,
            max_position_embeddings=self.max_seq_len,
            rms_norm_eps=self.norm_eps,
            rope_theta=self.rope_theta,
        )


# Load weights from Meta checkpoint (Llama 3.1 8B)
def load_attention_weights(ckpt_path: str, layer_idx: int = 0):
    ckpt = torch.load(sorted(Path(ckpt_path).glob("*.pth"))[0], map_location="cpu")
    print(ckpt[f"layers.{layer_idx}.attention.wq.weight"].shape)
    print(ckpt[f"layers.{layer_idx}.attention.wk.weight"].shape)
    print(ckpt[f"layers.{layer_idx}.attention.wv.weight"].shape)
    print(ckpt[f"layers.{layer_idx}.attention.wo.weight"].shape)
    return {
        "wq": ckpt[f"layers.{layer_idx}.attention.wq.weight"].to(torch.float32).numpy(),
        "wk": ckpt[f"layers.{layer_idx}.attention.wk.weight"].to(torch.float32).numpy(),
        "wv": ckpt[f"layers.{layer_idx}.attention.wv.weight"].to(torch.float32).numpy(),
        "wo": ckpt[f"layers.{layer_idx}.attention.wo.weight"].to(torch.float32).numpy(),
    }

def test_hf_attention(args, ckpt_dir: str,x, layer_idx: int = 0, atol: float = 1e-4):
    kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        
    weights = load_attention_weights(ckpt_dir, layer_idx)        
    torch_attn = hf_model.Attention(
        hf_model.ModelArgs(
            n_heads=args.n_heads, 
            n_kv_heads=kv_heads,
            dim=args.dim, 
            max_batch_size=args.max_batch_size, 
            max_seq_len=args.max_seq_len,
            rope_theta=args.rope_theta,
        ), 
    )
    
    torch_attn.load_state_dict({
        "wq.weight": torch.tensor(weights["wq"]),
        "wk.weight": torch.tensor(weights["wk"]),
        "wv.weight": torch.tensor(weights["wv"]),
        "wo.weight": torch.tensor(weights["wo"]),
    })
    freqs_cis = hf_model.precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
    #np.savetxt("freqs_cis_hf.txt", freqs_cis.reshape(-1, args.dim // args.n_heads), fmt='%.6f')

    torch_output = torch_attn(
        torch.tensor(x), 
        0, 
        freqs_cis, 
        torch.where(torch.tensor(np.asarray(make_causal_mask(jnp.ones((1, args.max_seq_len), dtype="bool"), dtype="bool"))) == False, float(jnp.finfo(jnp.float32).min), 0.0), 
    )
    torch_output = torch_output.detach().numpy()

    
    print("HF output shape:", torch_output.shape)

    return torch_output
        
    
    
def test_jax_attention(args, ckpt_dir: str,x, layer_idx: int = 0, atol: float = 1e-4):
    # Generate dummy input
    # Load Meta weights
    weights = load_attention_weights(ckpt_dir, layer_idx)
    flax_attn = FlaxLLaMAAttention(args.transformers_config(), precision='highest')
    dummy_vars = flax_attn.init(
        jax.random.PRNGKey(0),
        jnp.asarray(x),
        attention_mask=jnp.ones((args.max_batch_size, args.max_seq_len), dtype=jnp.int32),
        position_ids=jnp.broadcast_to(
            jnp.arange(args.max_seq_len, dtype=jnp.int32)[None, :],
            (args.max_batch_size, args.max_seq_len)
        )
    )
    params_unfrozen = unfreeze(dummy_vars)
    params_unfrozen["params"]["wq"]["kernel"] = jnp.asarray(weights["wq"].T)
    params_unfrozen["params"]["wk"]["kernel"] = jnp.asarray(weights["wk"].T)
    params_unfrozen["params"]["wv"]["kernel"] = jnp.asarray(weights["wv"].T)
    params_unfrozen["params"]["wo"]["kernel"] = jnp.asarray(weights["wo"].T)
    flax_params = freeze(params_unfrozen)
    attention_mask = jnp.ones((args.max_batch_size, args.max_seq_len), dtype=jnp.int32)
    position_ids = jnp.broadcast_to(
        jnp.arange(args.max_seq_len, dtype=jnp.int32)[None, :],
        (args.max_batch_size, args.max_seq_len)
    )
    
    jax_params = freeze({
        "wq": {"kernel": jnp.asarray(weights["wq"].T)},
        "wk": {"kernel": jnp.asarray(weights["wk"].T)},
        "wv": {"kernel": jnp.asarray(weights["wv"].T)},
        "wo": {"kernel": jnp.asarray(weights["wo"].T)},
    })

    with mesh:
        jax_output = flax_attn.apply(
            flax_params,
            jnp.asarray(x),
            attention_mask,
            position_ids,
            deterministic=True
        )[0]
        
    jax_out = np.asarray(jax_output)
    print("✅ JAX output shape:", jax_out.shape)
    return jax_out

    
    
x = np.random.randn(ModelArgs().max_batch_size, ModelArgs().max_seq_len, ModelArgs().dim).astype(np.float32)

p1 =test_hf_attention(ModelArgs(), "/root/tt/3_1_8b/Llama-Jax-Paralelism/llama3.1-8B/8B",x, layer_idx=0)


p2 =test_jax_attention(ModelArgs(), "/root/tt/3_1_8b/Llama-Jax-Paralelism/llama3.1-8B/8B",x, layer_idx=0)
# correlation
corr, _ = pearsonr(p1.flatten(), p2.flatten())
print(f"Pearson correlation: {corr:.4f}")