from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

from jax_llama.config import LLaMAConfig

import flax.nnx as nnx
from flax.linen import partitioning as nn_partitioning
remat = nn_partitioning.remat


logger = logging.get_logger(__name__)

class RMSNorm(nnx.Module):
    """RMSNorm module."""
    def __init__(self, dim: int, eps: float=1e-6, dtype=jnp.float32, param_dtype=jnp.float32,rngs: nnx.Rngs = nnx.Rngs(0)):
        self.dim = dim
        self.eps = eps
        self.dtype = dtype
        self.param_dtype = param_dtype
        
        # define parameter directly
        self.weight = nnx.Param(
            jnp.ones((dim,), dtype=param_dtype)
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        output = self._norm(x.astype(self.dtype)).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight

def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0, dtype: jnp.dtype=jnp.float32) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = np.arange(end)  # type: ignore
    freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)

def apply_rotary_emb(
    xq: jnp.ndarray, 
    xk: jnp.ndarray, 
    freqs_cis: jnp.ndarray, 
    dtype: jnp.dtype=jnp.float32, 
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
    
    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    # add head dim
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))
    
    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype), xk_out.astype(dtype)

def repeat_kv(
    hidden_states: jnp.ndarray,
    n_rep: int,
) -> jnp.ndarray:
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :]
    hidden_states = jnp.repeat(hidden_states, n_rep, axis=3)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

class FlaxLLaMAAttention(nnx.Module):
    """LLaMA attention module."""
    def __init__(
        self,
        config: LLaMAConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        init_fn = nnx.initializers.normal(config.initializer_range)

        # Define Layers
        self.wq = nnx.Linear(
            self.embed_dim,
            self.num_heads * self.head_dim,
            kernel_init=nnx.with_partitioning(init_fn, (None, 'model')),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            rngs=rngs,
        )
        self.wk = nnx.Linear(
            self.embed_dim,
            self.num_key_value_heads * self.head_dim,
            kernel_init=nnx.with_partitioning(init_fn, (None, 'model')),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            rngs=rngs,
        )
        self.wv = nnx.Linear(
            self.embed_dim,
            self.num_key_value_heads * self.head_dim,
            kernel_init=nnx.with_partitioning(init_fn, (None, 'model')),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            rngs=rngs,
        )
        self.wo = nnx.Linear(
            self.embed_dim,
            self.embed_dim,
            kernel_init=nnx.with_partitioning(init_fn, ('model', None)),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            rngs=rngs,
        )

        # Dropout
        self.resid_dropout = nnx.Dropout(rate=config.resid_pdrop)

        # Precompute masks
        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_sequence_length), dtype="bool"), dtype="bool")
        self.freqs_cis = precompute_freqs_cis(
            self.head_dim,
            config.max_sequence_length * 2,
            theta=config.rope_theta,
            dtype=self.dtype,
        )
    
    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def _concatenate_to_cache(self, key, value, query, attention_mask):
        if not hasattr(self, 'cached_key'):
            self.cached_key = key
            self.cached_value = value
            self.cache_index = jnp.array(0, dtype=jnp.int32)

        batch_dims = key.shape[:-3]
        max_length = self.cached_key.shape[1]
        cur_index = self.cache_index
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)

        key = lax.dynamic_update_slice(self.cached_key, key, indices)
        value = lax.dynamic_update_slice(self.cached_value, value, indices)
        self.cached_key = key
        self.cached_value = value

        num_updated = query.shape[1]
        self.cache_index = self.cache_index + num_updated

        pad_mask = jnp.broadcast_to(
            jnp.arange(max_length) < cur_index + num_updated,
            tuple(batch_dims) + (1, num_updated, max_length),
        )
        attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        xq, xk, xv = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        freqs_cis = jnp.take(self.freqs_cis, position_ids, axis=0)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=self.dtype)

        query_length, key_length = xq.shape[1], xk.shape[1]
        
        if hasattr(self, 'cached_key'):
            mask_shift = self.cache_index
            max_decoder_length = self.cached_key.shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]
        
        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)

        dropout_rng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")
        
        if hasattr(self, 'cached_key') or init_cache:
            xk, xv, attention_mask = self._concatenate_to_cache(xk, xv, xq, attention_mask)
        
        # transform boolean mask into float mask
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0, dtype=self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min, dtype=self.dtype),
        )

        xk = repeat_kv(xk, self.num_key_value_groups)
        xv = repeat_kv(xv, self.num_key_value_groups)

        # usual dot product attention
        attn_weights = dot_product_attention_weights(
            xq, 
            xk, 
            bias=attention_bias, 
            dropout_rng=dropout_rng, 
            dropout_rate=self.config.attn_pdrop, 
            deterministic=deterministic, 
            dtype=self.dtype, 
            precision=self.precision, 
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, xv, precision=self.precision)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs
        
class FlaxLLaMAMLP(nnx.Module):
    def __init__(
        self,
        config: LLaMAConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        init_fn = nnx.initializers.normal(config.initializer_range)

        # Apply partitioning to all dense layers
        self.w1 = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.w2 = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.w3 = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.dropout = nnx.Dropout(rate=config.resid_pdrop)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x, deterministic=deterministic)
        return x
    
class FlaxLLaMABlock(nnx.Module):
    def __init__(
        self,
        config: LLaMAConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.attention = FlaxLLaMAAttention(
            config, rngs=rngs, dtype=dtype, param_dtype=param_dtype, precision=precision
        )
        self.feed_forward = FlaxLLaMAMLP(
            config, rngs=rngs, dtype=dtype, param_dtype=param_dtype, precision=precision
        )
        self.attention_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype
        )
        self.ffn_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        normed_attn_input = self.attention_norm(hidden_states)
        attn_outputs = self.attention(
            normed_attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output

        normed_ffn_input = self.ffn_norm(hidden_states)
        feed_forward_hidden_states = self.feed_forward(normed_ffn_input, deterministic=deterministic)
        hidden_states = hidden_states + feed_forward_hidden_states

        return (hidden_states,) + attn_outputs[1:]


class FlaxLLaMAPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LLaMAConfig
    base_model_prefix = "transformer"
    module_class: nnx.Module = None

    def __init__(
        self,
        config: LLaMAConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        rngs = nnx.Rngs(seed)

        # ✅ initialize model using nnx, with rngs
        module = self.module_class(
            config=config,
            rngs=rngs,
            dtype=dtype,
            **kwargs,
        )
        self.rngs = rngs

        # ✅ capture initial state
        state = nnx.state(module)
        self.state = state
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)


    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype=jnp.int32)
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(input_ids.shape[-1]), input_shape)

        rngs = nnx.Rngs(rng)

        outputs = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            position_ids,
            return_dict=False,
            init_cache=False,
        )

        random_params = outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        """Initialize cache for fast decoding."""
        input_ids = jnp.ones((batch_size, max_length), dtype=jnp.int32)
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(max_length)[None, :], input_ids.shape)

        rngs = nnx.Rngs(0)

        outputs = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            position_ids,
            return_dict=False,
            init_cache=True,
        )
        return outputs["cache"]

    @add_start_docstrings_to_model_forward("")
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxGPTJAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            input_ids,
            attention_mask,
            position_ids,
            deterministic=not train,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        if past_key_values is not None:
            outputs, updated_cache = outputs
            if return_dict:
                outputs["past_key_values"] = unfreeze(updated_cache["cache"])
                return outputs
            else:
                outputs = outputs[:1] + (unfreeze(updated_cache["cache"]),) + outputs[1:]

        return outputs

class FlaxLLaMABlockCollection(nnx.Module):
    def __init__(
        self,
        config: LLaMAConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        block_cls = FlaxLLaMABlock
        if config.gradient_checkpointing:
            block_cls = remat(block_cls, static_argnums=(3, 4, 5))

        self.blocks = []
        for i in range(config.num_hidden_layers):
            self.blocks.append(
                block_cls(
                    config=config,
                    rngs=rngs,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                )
            )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = [] if output_attentions else None
        all_hidden_states = [] if output_hidden_states else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_outputs = block(
                hidden_states,
                attention_mask,
                position_ids,
                deterministic,
                init_cache,
                output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions.append(layer_outputs[1])

        outputs = (hidden_states, tuple(all_hidden_states) if all_hidden_states is not None else None,
                   tuple(all_attentions) if all_attentions is not None else None)

        return outputs
        

class FlaxLLaMAModule(nnx.Module):
    def __init__(
        self,
        config: LLaMAConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.embed_dim = config.hidden_size

        init_fn = nnx.initializers.normal(stddev=self.config.initializer_range)

        # Embedding layer with sharding
        self.wte = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=nnx.with_partitioning(init_fn, ('mp', None)),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(rate=self.config.embd_pdrop)

        # Transformer blocks
        self.h = FlaxLLaMABlockCollection(
            config,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

        self.ln_f = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        input_embeds = self.wte(input_ids.astype("i4"))

        hidden_states = self.dropout(input_embeds, deterministic=deterministic)

        outputs = self.h(
            hidden_states,
            attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )

@add_start_docstrings("", "")
class FlaxLLaMAModel(FlaxLLaMAPreTrainedModel):
    module_class = FlaxLLaMAModule


class FlaxLLaMAForCausalLMModule(nnx.Module):
    def __init__(
        self,
        config: LLaMAConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        # Transformer body
        self.transformer = FlaxLLaMAModule(
            config=config,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

        # Language modeling head
        init_fn = nnx.initializers.normal(config.initializer_range)

        self.lm_head = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            kernel_init=nnx.with_partitioning(init_fn, ('model', None)),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # Tie embeddings if needed
        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.wte.embedding.value.T
            lm_logits = jnp.einsum('bld,vd->blv', hidden_states, shared_kernel)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs[1],
            attentions=outputs[2],
        )
    
@add_start_docstrings("", "")
class FlaxLLaMAForCausalLM(FlaxLLaMAPreTrainedModel):
    module_class = FlaxLLaMAForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jnp.ndarray] = None):
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)

        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = extended_attention_mask.at[:, :attention_mask.shape[1]].set(attention_mask)
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
