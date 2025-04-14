# run_model.py

import jax
import jax.numpy as jnp
import equinox as eqx

from model import LlamaForCausalLM, LlamaConfig  # Import from model.py

# === Helper to print model structure ===
def print_model_structure(module, prefix=""):
    if isinstance(module, list):
        for i, m in enumerate(module):
            print_model_structure(m, prefix + f"[{i}].")
        return

    for name, attr in vars(module).items():
        if isinstance(attr, eqx.Module):
            print(f"{prefix}{name}: {type(attr).__name__}")
            print_model_structure(attr, prefix + f"{name}.")
        elif isinstance(attr, list) and all(isinstance(m, eqx.Module) for m in attr):
            print(f"{prefix}{name}: list[{len(attr)} modules]")
            for i, m in enumerate(attr):
                print_model_structure(m, prefix + f"{name}[{i}].")
        else:
            print(f"{prefix}{name}: {type(attr).__name__}")

# === Create dummy config ===
config = LlamaConfig(
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    vocab_size=128256,
    rms_norm_eps=1e-5,
    initializer_range=0.02,

)

# Properly split the key before passing into the model
key = jax.random.PRNGKey(0)
model_key, input_key = jax.random.split(key)

# Instantiate the model with correct key
model = LlamaForCausalLM(config, model_key)

dummy_input = jnp.ones((1, 128), dtype=jnp.int32)  # Dummy input for the model
dummy_attention_mask = jnp.ones((1, 128), dtype=jnp.int32)  # Dummy attention mask
dummy_position_ids = jnp.arange(128).reshape((1, 128))  # Dummy position IDs
dummy_input = (dummy_input, dummy_attention_mask, dummy_position_ids)

# === Forward pass (optional) ===
output = model(dummy_input, key)

# === Print structure ===
print("Model Structure:")
print_model_structure(model)
