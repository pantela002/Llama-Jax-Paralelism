import os
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze
from jax_llama.model import FlaxLLaMAForCausalLM
from jax_llama.config import LLaMAConfig
from jax_llama.llama3_tokenizer import Tokenizer as LLaMA3Tokenizer
from jax_llama.convert_weights import convert_llama_weights  # assumes this exists in your setup
from transformers import AutoConfig
import numpy as np
from jax_llama import config



# === CONFIGURE THESE ===
ckpt_dir = "/root/tt/sw/llama3.1-8B/8B"
tokenizer_path = "/root/tt/sw/llama3.1-8B/original/tokenizer.model"

model_id = "meta-llama/Meta-Llama-3.1-8B"
config = LLaMAConfig(
    embd_pdrop = 0.0,
    resid_pdrop = 0.0,
    attn_pdrop = 0.0,
    tie_word_embeddings = False,
    gradient_checkpointing = False,
    num_hidden_layers=1,  # 👈 force only 1 layer

)

# 2. Load checkpoint weights
tokenizer = LLaMA3Tokenizer(tokenizer_path)
jax_params, _ = convert_llama_weights(
    ckpt_dir=ckpt_dir,
    tokenizer=tokenizer,
)
model = FlaxLLaMAForCausalLM(config=config, dtype=jnp.float16)
params = freeze(jax.tree.map(jnp.asarray, jax_params))

# 4. Create dummy input
input_ids = jnp.ones((1, 8), dtype=jnp.int32)
attention_mask = jnp.ones_like(input_ids)
position_ids = jnp.arange(8)[None, :]


import psutil, os
process = psutil.Process(os.getpid())
print("🔍 After model init:", process.memory_info().rss / (1024 * 1024), "MB")


# 5. Run inference
output = model.module.apply(
    {"params": params},
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    deterministic=True,
)

# 6. Output
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# 6. Output
print("✅ Logits shape (real weights, unsharded):", output.logits.shape)
print(output.logits)

np.save("output_jax_unsharded.npy", output.logits)
