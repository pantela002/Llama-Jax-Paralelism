import os
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze
from jax_llama.model import FlaxLLaMAForCausalLM
from jax_llama.config import LLaMAConfig
from jax_llama.llama3_tokenizer import Tokenizer as LLaMA3Tokenizer
from jax_llama.convert_weights import convert_llama_weights  # assumes this exists in your setup
from transformers import AutoConfig

# === CONFIGURE THESE ===
ckpt_dir = "/root/tt/sw/llama3.1-8B/8B"
tokenizer_path = "/root/tt/sw/llama3.1-8B/original/tokenizer.model"

model_id = "meta-llama/Meta-Llama-3.1-8B"
config = LLaMAConfig(
    embd_pdrop = 0.0,
    resid_pdrop = 0.0,
    attn_pdrop = 0.0,
    gradient_checkpointing = False
)

# 2. Load checkpoint weights
tokenizer = LLaMA3Tokenizer(tokenizer_path)
jax_params, _ = convert_llama_weights(
    ckpt_dir=ckpt_dir,
    tokenizer=tokenizer,
)
params = freeze(jax.tree.map(jnp.asarray, jax_params))

# 3. Create model
model = FlaxLLaMAForCausalLM(config=config, dtype=jnp.bfloat16)

# 4. Create dummy input
input_ids = jnp.ones((1, 8), dtype=jnp.int32)
attention_mask = jnp.ones_like(input_ids)
position_ids = jnp.arange(8)[None, :]

# 5. Run inference
output = model.module.apply(
    {"params": params},
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    deterministic=True,
)

# 6. Output
print("✅ Logits shape (real weights, unsharded):", output.logits.shape)
print(output)

with open("output_unsharded_real_weights.txt", "w") as f:
    f.write(str(output.logits))
