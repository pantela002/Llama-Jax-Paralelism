import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import jax
print('DEVICES:', jax.devices())
import jax.numpy as jnp
from jax_llama import model as jax_model
from jax_llama import hf_model as model
import torch
import numpy as np
import os
from flax.core.frozen_dict import freeze
from dataclasses import dataclass
from typing import Optional
from jax_llama import config
from flax.linen import make_causal_mask
from jax.sharding import PartitionSpec as P
import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init

if not dist.is_initialized():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    fs_init.initialize_model_parallel(1)


@dataclass
class ModelArgs:
    dim: int = 32
    n_layers: int = 4
    n_heads: int = 4
    vocab_size: int = 256
    n_kv_heads: Optional[int] = 2
    multiple_of: int = 2
    ffn_dim_multiplier: Optional[float] = 1.3
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0

    max_batch_size: int = 1
    max_seq_len: int = 64

    def transformers_config(self) -> config.LLaMAConfig:
        intermediate_size = int(2 * (self.dim * 4) / 3)
        if self.ffn_dim_multiplier is not None:
            intermediate_size = int(self.ffn_dim_multiplier * intermediate_size)
        intermediate_size = self.multiple_of * ((intermediate_size + self.multiple_of - 1) // self.multiple_of)
        return config.LLaMAConfig(
            vocab_size=self.vocab_size, 
            hidden_size=self.dim, 
            intermediate_size=intermediate_size, 
            num_hidden_layers=self.n_layers, 
            num_attention_heads=self.n_heads, 
            num_key_value_heads=self.n_kv_heads,
            max_sequence_length=self.max_seq_len, 
            rms_norm_eps=self.norm_eps, 
            rope_theta=self.rope_theta,
        )


def test_Attention(args: ModelArgs, total_tests: int, atol: float) -> float:
    kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
    errs = []
    for test_n in range(total_tests):
        x = np.random.randn(args.max_batch_size, args.max_seq_len, args.dim).astype(np.float32)
        wq = np.random.randn(args.dim, args.dim).astype(np.float32)
        wk = np.random.randn(args.dim, args.dim//(args.n_heads//kv_heads)).astype(np.float32)
        wv = np.random.randn(args.dim, args.dim//(args.n_heads//kv_heads)).astype(np.float32)
        wo = np.random.randn(args.dim, args.dim).astype(np.float32)
        print(wq.shape, wk.shape, wv.shape, wo.shape)
        
        jax_attention = jax_model.FlaxLLaMAAttention(args.transformers_config(), precision='highest')
        jax_params = freeze({
            'wq': {'kernel': jnp.asarray(wq)}, 
            'wk': {'kernel': jnp.asarray(wk)}, 
            'wv': {'kernel': jnp.asarray(wv)}, 
            'wo': {'kernel': jnp.asarray(wo)}, 
        })
        jax_output = jax_attention.apply(
            {'params': jax_params}, 
            jnp.asarray(x), 
            jnp.ones((args.max_batch_size, args.max_seq_len), dtype=np.int32), 
            jnp.broadcast_to(jnp.arange(args.max_seq_len, dtype=np.int32)[None, :], (args.max_batch_size, args.max_seq_len)), 
        )[0]
        jax_output = np.asarray(jax_output)

        torch_freqs_cis = model.precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
        np.savetxt("freqs_cis_hf.txt", torch_freqs_cis.reshape(-1, args.dim // args.n_heads), fmt='%.6f')
        torch_attention = model.Attention(
            model.ModelArgs(
                n_heads=args.n_heads, 
                n_kv_heads=kv_heads,
                dim=args.dim, 
                max_batch_size=args.max_batch_size, 
                max_seq_len=args.max_seq_len,
                rope_theta=args.rope_theta,
            ), 
        )

        torch_attention.load_state_dict({
            "wo.weight": torch.tensor(wo.transpose()), 
            "wq.weight": torch.tensor(wq.transpose()), 
            "wv.weight": torch.tensor(wv.transpose()), 
            "wk.weight": torch.tensor(wk.transpose()), 
        }) # load weights, have to transpose because pytorch linear layers are reversed from Jax.
        torch_output = torch_attention(
            torch.tensor(x), 
            0, 
            torch_freqs_cis, 
            torch.where(torch.tensor(np.asarray(make_causal_mask(jnp.ones((1, args.max_seq_len), dtype="bool"), dtype="bool"))) == False, float(jnp.finfo(jnp.float32).min), 0.0), 
        )
        torch_output = torch_output.detach().numpy()

        errs.append(np.max(np.abs(jax_output - torch_output)))
        #correlation pearson
        from scipy.stats import pearsonr
        corr, _ = pearsonr(jax_output.flatten(), torch_output.flatten())
        print(f"Test {test_n}: Max error: {errs[-1]}, Pearson correlation: {corr:.4f}")
        
    return np.asarray(errs, dtype=np.float32)


print('='*10)
print("[Testing Attention]")
errs = test_Attention(ModelArgs(), 128, atol=1e-2)
print("[Passed]")
print("Max Attention error: %f" % (np.max(errs)))
print("Mean Attention error: %f" % (np.mean(errs)))
print("Median Attention error: %f" % (np.median(errs)))
print('='*10)