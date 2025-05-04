import os

# Simulate 8 devices on CPU before importing jax!
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import numpy as np

from flax.core.frozen_dict import freeze
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from jax_llama.model import FlaxLLaMAForCausalLM
from jax_llama.config import LLaMAConfig
from jax_llama.partition import get_llama_param_partition_spec
from jax.experimental import pjit as old_pjit

# 1. Setup a fake mesh: 2 data × 4 model = 8 devices
devices = np.array(jax.devices()).reshape(2, 4) 
mesh = Mesh(devices, axis_names=("dp", "mp"))


# 2. Build config and model
config = LLaMAConfig(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=128,
    rms_norm_eps=1e-5,
)

model = FlaxLLaMAForCausalLM(config=config)
params = model.init_weights(jax.random.PRNGKey(0), input_shape=(1, 8))
params = freeze(params)

# 3. Get partition spec from your partition.py logic
param_spec = get_llama_param_partition_spec(params)

# 4. Shard params using pjit
shard_params = old_pjit.pjit(
    lambda x: x,
    None,                # in_axis_resources
    param_spec,          # out_axis_resources
    (),                  # static_argnums
    (),                  # donate_argnums
)

with mesh:
    sharded_params = shard_params(params)

    # 🔍 Visualize sharding of a few parameters
    from flax.traverse_util import flatten_dict
    import jax.debug

    print("\n🔍 Visualizing sharded parameter placements (first few):")
    flat = flatten_dict(sharded_params)
    for k, v in list(flat.items())[:5]:  # visualize 5 parameters
        print(f"🔹 Param {k} sharding:")
        jax.debug.visualize_array_sharding(v)

# 5. Create dummy inputs
input_ids = jnp.ones((1, 8), dtype=jnp.int32)
attention_mask = jnp.ones_like(input_ids)
position_ids = jnp.arange(8)[None, :]

# 6. Apply model using sharded params
with mesh:
    output = model.module.apply(
        {"params": sharded_params},
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        deterministic=True,
    )

print("✅ Logits shape:", output.logits.shape)

print(output)

#save txt output to file
with open("output.txt", "w") as f:
    f.write(str(output.logits))
