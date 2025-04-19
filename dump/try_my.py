import jax.numpy as jnp
from flax.training.common_utils import shard
from jax import random
from model import FlaxLLaMAForCausalLM, LLaMAConfig  # <- adjust path
import numpy as np
# Load config + model
config = LLaMAConfig()
model = FlaxLLaMAForCausalLM(config=config)

# Load input ids and attention mask
input_ids = jnp.array(np.load("/root/tt/Jax_llama/jax_llama/hf_input_ids.npy"))  # shape: (1, seq_len)
attention_mask = jnp.array(np.load("/root/tt/Jax_llama/jax_llama/hf_attention_mask.npy"))
position_ids = jnp.arange(input_ids.shape[1])[None, :]

# Initialize params (or load pretrained if available)
params = model.init_weights(random.PRNGKey(0), input_ids.shape)

# Run model
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    params=params,
    train=False
)
jax_logits = outputs.logits
next_token_logits = jax_logits[:, -1, :]
next_token_id = jnp.argmax(next_token_logits, axis=-1)

# Decode
next_word = tokenizer.decode(np.array(next_token_id))
print("[JAX] Next word prediction:", next_word)
