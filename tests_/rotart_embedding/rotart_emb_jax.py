"""Equinox implementation of the Llama model with LoRA support."""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Any, List
import ml_dtypes
import jax.nn.initializers as init


DTYPE_MAP = {
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
    "int32": jnp.int32,
    "int64": jnp.int64,
}

# Define rematerialization policies
remat_policy = {
    "nothing": jax.checkpoint_policies.nothing_saveable,
    "dots": jax.checkpoint_policies.checkpoint_dots,
    "dots_with_no_batch_dims": jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    "everything": jax.checkpoint_policies.everything_saveable,
}


class LlamaConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 128256)
        self.hidden_size = kwargs.get("hidden_size", 4096)
        self.intermediate_size = kwargs.get("intermediate_size", 14336)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        self.num_attention_heads = kwargs.get("num_attention_heads", 32)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 131072)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-5)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.hidden_dropout = kwargs.get("hidden_dropout", 0.0)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.use_cache = kwargs.get("use_cache", True)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", False)
        self.attention_bias = kwargs.get("attention_bias", False)
        self.pretraining_tp = kwargs.get("pretraining_tp", 1)
        self.torch_dtype = kwargs.get("torch_dtype", "float32")

        self.bos_token_id = kwargs.get("bos_token_id", 128000)
        self.eos_token_id = kwargs.get("eos_token_id", 128001)
        self.pad_token_id = kwargs.get("pad_token_id", None)

        self.rope_theta = kwargs.get("rope_theta", 500000.0)
        self.rope_scaling = kwargs.get("rope_scaling", {
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        })

        # Derived
        self.head_dim = self.hidden_size // self.num_attention_heads

        # Not in HF config, but useful internally
        self.lora_rank = kwargs.get("lora_rank", 0)
        self._param_dtype = kwargs.get("param_dtype", "bfloat16")
        self._compute_dtype = kwargs.get("compute_dtype", "bfloat16")
        self.use_optimized_decoder = kwargs.get("use_optimized_decoder", True)

        # Compatibility / experimental
        self.bias = kwargs.get("bias", False)
        self.rope_type = kwargs.get("rope_type", "llama3")
        self.partial_rotary_factor = kwargs.get("partial_rotary_factor", 1.0)


    @property
    def param_dtype(self):
        """Gets the parameter dtype, converting from string to JAX dtype."""
        return DTYPE_MAP.get(self._param_dtype, jnp.bfloat16)

    @property
    def compute_dtype(self):
        """Gets the compute dtype, converting from string to JAX dtype."""
        return DTYPE_MAP.get(self._compute_dtype, jnp.bfloat16)

    def to_dict(self):
        """Serializes the configuration to a dictionary."""
        return self.__dict__

    def __repr__(self):
        return f"LlamaConfig({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"


class LlamaEmbedding(eqx.Module):
    weight: jnp.ndarray
    param_dtype: Any
    compute_dtype: Any

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        key=None,
    ):
        self.param_dtype = param_dtype
        self.compute_dtype = compute_dtype
        if key is None:
            key = jax.random.PRNGKey(99)
        self.weight = jax.random.normal(
            key, (num_embeddings, embedding_dim), dtype=self.param_dtype
        )

    def __call__(self, x):
        weight = self.weight.astype(self.compute_dtype)
        embeddings = jnp.take(weight, x, axis=0)
        return embeddings.astype(self.compute_dtype)


class LlamaRotaryEmbedding(eqx.Module):
    inv_freq: jnp.ndarray
    max_seq_len_cached: int
    param_dtype: Any
    compute_dtype: Any

    def __init__(
        self, config, param_dtype=jnp.float32, compute_dtype=jnp.float32
    ):
        self.param_dtype = param_dtype
        self.compute_dtype = compute_dtype

        dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len_cached = config.max_position_embeddings
        inv_freq = 1.0 / (
            config.rope_theta
            ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim)
        )
        # TODO(mixed_precision): always using float32 for inv_freq.
        self.inv_freq = inv_freq.astype(jnp.float32)

    def __call__(self, x, position_ids):
        # TODO(mixed_precision): check if x should be retained as float32 for rotary embeddings.
        x = x.astype(jnp.float32)
        seq_len = position_ids.shape[1]
        t = position_ids.astype(jnp.float32)
        inv_freq = self.inv_freq

        # Reshape t to match the expected input shape
        t = t.reshape(-1, seq_len, 1)  # Shape: (batch_size, seq_len, 1)

        # Compute freqs directly without using einsum
        freqs = (
            t * inv_freq[None, None, :]
        )  # Shape: (batch_size, seq_len, dim//2)

        emb = jnp.concatenate(
            (freqs, freqs), axis=-1
        )  # Shape: (batch_size, seq_len, dim)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        return cos.astype(jnp.float32), sin.astype(jnp.float32)



import numpy as np
import jax.numpy as jnp

hf_cos = np.load("hf_cos.npy").astype(np.float32)
hf_sin = np.load("hf_sin.npy").astype(np.float32)

# Wrap with jax
hf_cos = jnp.array(hf_cos)
hf_sin = jnp.array(hf_sin)

# Setup config same as HF
config = LlamaConfig(
    hidden_size=4096,
    num_attention_heads=32,
    max_position_embeddings=131072,
    rope_theta=500000.0,
)

# Create JAX rotary module
rotary = LlamaRotaryEmbedding(config)

# Define dummy input: same shape used in HF export
batch_size = 1
seq_len = 6
head_dim = 128

dummy_input = jnp.zeros((batch_size, seq_len, head_dim))
position_ids = jnp.arange(seq_len)[None, :]  # shape: [1, seq_len]

# Call JAX rotary embedding
cos_jax, sin_jax = rotary(dummy_input, position_ids)

# Compare
cos_diff = jnp.max(jnp.abs(hf_cos - cos_jax))
sin_diff = jnp.max(jnp.abs(hf_sin - sin_jax))

print(f"Cos diff: {cos_diff}")
print(f"Sin diff: {sin_diff}")
np.savetxt("tests_/rotart_embedding/jax_cos.txt", np.array(cos_jax).reshape(seq_len, -1), fmt="%.6f")
np.savetxt("tests_/rotart_embedding/jax_sin.txt", np.array(sin_jax).reshape(seq_len, -1), fmt="%.6f")