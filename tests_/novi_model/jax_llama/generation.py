import jax
import jax.numpy as jnp
from jax_llama import FlaxLLaMAForCausalLM
from jax_llama.llama3_tokenizer import Tokenizer as LLaMA3Tokenizer
from transformers.generation import GenerationConfig
from jax.sharding import Mesh
from jaxtyping import PyTree
from flax import struct
from functools import partial
from typing import List, Optional, Union


class LLaMA(struct.PyTreeNode):
    params: PyTree
    model: FlaxLLaMAForCausalLM = struct.field(pytree_node=False)
    tokenizer: LLaMA3Tokenizer = struct.field(pytree_node=False)
    mesh: Optional[Mesh] = struct.field(pytree_node=False, default=None)

    @partial(jax.jit, static_argnums=(3, 4))
    def generate(
        self,
        tokens: jnp.ndarray,
        attention_mask: jnp.ndarray,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> jnp.ndarray:
        # Generation happens inside the mesh context
        with self.mesh:
            generations = self.model(
                input_ids=tokens,
                attention_mask=attention_mask,
            )
        return generations.sequences

    def generate_from_str(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.1,
        top_p: float = 0.99,
    ):
        # Tokenize input prompts
        prompt_tokens = [
            self.tokenizer.encode(
                x,
                bos=True,
                eos=False,
                allowed_special="all",
                disallowed_special=(),
            )
            for x in prompts
        ]

        max_prompt_size = max(len(t) for t in prompt_tokens)

        # Left pad tokens
        tokens = jnp.full((len(prompts), max_prompt_size), self.tokenizer.pad_id, dtype=jnp.int32)
        for i, t in enumerate(prompt_tokens):
            tokens = tokens.at[i, -len(t):].set(jnp.array(t))

        attention_mask = (tokens != self.tokenizer.pad_id).astype(jnp.int32)

        # Generate tokens
        out_tokens = self.generate(tokens, attention_mask, max_gen_len, temperature, top_p)

        # Decode outputs
        decoded = []
        for t in out_tokens.tolist():
            try:
                start_idx = t.index(self.tokenizer.bos_id)
            except ValueError:
                start_idx = 0
            t = t[start_idx:]
            decoded.append(self.tokenizer.decode(t))

        return decoded
