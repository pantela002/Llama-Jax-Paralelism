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
import os
from flax.traverse_util import flatten_dict


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
    num_hidden_layers=2,  # 👈 force only 1 layer
    max_sequence_length = 128


)

# 2. Load checkpoint weights
tokenizer = LLaMA3Tokenizer(tokenizer_path)
jax_params, _ = convert_llama_weights(
    ckpt_dir=ckpt_dir,
    tokenizer=tokenizer,
    max_seq_len=128,
    verbose=True
)
print(sorted(jax_params['transformer']['h'].keys()))

model = FlaxLLaMAForCausalLM(config=config, dtype=jnp.float16, _do_init=False)
params = freeze(jax.tree.map(jnp.asarray, jax_params))
print(sorted(params['transformer']['h'].keys()))

"""from flax.traverse_util import flatten_dict
flat = flatten_dict(params)
print("🔍 Inspecting loaded parameter stats (first 10 entries):")
for name, value in list(flat.items())[:10]:
    value_np = np.array(value)
    print(f"{name}: mean={value_np.mean():.4f}, std={value_np.std():.4f}")

exit(0)"""

"""
flat_params = flatten_dict(params, sep='.')
os.makedirs("jax_params", exist_ok=True)

with open("jax_params.txt", "w") as f:
    for key, value in flat_params.items():
        np_value = np.array(value)
        f.write(f"{key}: shape={np_value.shape}, dtype={np_value.dtype}\n")
        f.write(np.array2string(np_value, separator=', ', threshold=10, edgeitems=3))
        f.write("\n\n")
"""

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
    return_dict=True
)

# 6. Output
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# 6. Output
print("✅ Logits shape (real weights, unsharded):", output.logits.shape)
#print(output.logits)

np.save("output_jax_unsharded.npy", output.logits)
