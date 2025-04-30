import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax_llama import FlaxLLaMAForCausalLM
from jax_llama.llama3_tokenizer import Tokenizer as LLaMA3Tokenizer
from transformers.generation import GenerationConfig
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import PyTree
from flax import struct
from typing import List, Optional, Union

def _shard(x, mesh, spec):
    if mesh is not None:
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, spec))
    return x

class LLaMA(struct.PyTreeNode):
    params: PyTree
    model: FlaxLLaMAForCausalLM = struct.field(pytree_node=False)
    tokenizer: LLaMA3Tokenizer = struct.field(pytree_node=False)
    mesh: Optional[Mesh] = struct.field(pytree_node=False, default=None)

    def __post_init__(self):
        # bind pjit-ed generation function on init
        self._pjit_generate = pjit(
            self._generate_fn,
            in_shardings=(P("dp", None), P("dp", None), None, None, None),
            out_shardings=P("dp", None),
        )

    def _generate_fn(self, tokens, attention_mask, max_gen_len, temperature, top_p):
        tokens = _shard(tokens, self.mesh, P("dp", None))
        attention_mask = _shard(attention_mask, self.mesh, P("dp", None))

        generations = self.model.generate(
            input_ids=tokens,
            attention_mask=attention_mask,
            params=self.params,
            generation_config=GenerationConfig(
                num_beams=1,
                do_sample=temperature != 0.0,
                max_length=max_gen_len + tokens.shape[1],
                pad_token_id=self.tokenizer.eos_id,
                eos_token_id=self.tokenizer.eos_id,
                temperature=temperature,
                top_p=top_p,
            ),
        )

        out_tokens = generations.sequences
        return _shard(out_tokens, self.mesh, P("dp", None))

    def generate(self, tokens: jnp.ndarray, attention_mask: jnp.ndarray, max_gen_len: int, temperature: float = 0.8, top_p: float = 0.95) -> jnp.ndarray:
        return self._pjit_generate(tokens, attention_mask, max_gen_len, temperature, top_p)

    def generate_from_str(self, prompts: List[str], max_gen_len: int, temperature: float = 0.8, top_p: float = 0.95):
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        max_prompt_size = max([len(t) for t in prompt_tokens])

        tokens = jnp.full((len(prompts), max_prompt_size), self.tokenizer.eos_id).astype(jnp.int32)
        for i, t in enumerate(prompt_tokens):
            tokens = tokens.at[i, -len(t):].set(t)
        attention_mask = (tokens != self.tokenizer.eos_id).astype(jnp.int32)

        out_tokens = self.generate(tokens, attention_mask, max_gen_len, temperature, top_p)

        decoded = []
        for i, t in enumerate(out_tokens.tolist()):
            t = t[t.index(self.tokenizer.bos_id):]
            t = t[:(len(prompt_tokens[i]) + max_gen_len)]
            try:
                t = t[:t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded
