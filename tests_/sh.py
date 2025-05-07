"""import os

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


config = LLaMAConfig(
    embd_pdrop = 0.0,
    resid_pdrop = 0.0,
    attn_pdrop = 0.0,
    tie_word_embeddings = False,
    gradient_checkpointing = False,
    num_hidden_layers=1,  # 👈 force only 1 layer

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
np.save("output_sharded.npy", np.array(output.logits))


"""

import os

# Simulate 8 devices on CPU before importing JAX!
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import numpy as np

from flax.core.frozen_dict import freeze
from jax.experimental.pjit import pjit
from jax.sharding import Mesh
from flax.traverse_util import flatten_dict
import jax.debug

from jax_llama.model import FlaxLLaMAForCausalLM
from jax_llama.config import LLaMAConfig
from jax_llama.partition import get_llama_param_partition_spec
from jax_llama.convert_weights import convert_llama_weights
from jax_llama.llama3_tokenizer import Tokenizer as LLaMA3Tokenizer
from jax.experimental import pjit as old_pjit

# === CONFIGURE THESE ===
ckpt_dir = "/root/tt/sw/llama3.1-8B/8B"
tokenizer_path = "/root/tt/sw/llama3.1-8B/original/tokenizer.model"

# 1. Setup device mesh
devices = np.array(jax.devices()).reshape(2, 4)
mesh = Mesh(devices, axis_names=("dp", "mp"))

# 2. Load config and weights
config = LLaMAConfig(
    embd_pdrop=0.0,
    resid_pdrop=0.0,
    attn_pdrop=0.0,
    tie_word_embeddings=False,
    gradient_checkpointing=False,
    num_hidden_layers=1,
)

tokenizer = LLaMA3Tokenizer(tokenizer_path)
jax_params, _ = convert_llama_weights(ckpt_dir=ckpt_dir, tokenizer=tokenizer)
params = freeze(jax.tree.map(jnp.asarray, jax_params))

# 3. Init model and get partition spec
model = FlaxLLaMAForCausalLM(config=config, dtype=jnp.float16)
param_spec = get_llama_param_partition_spec(params)

# 4
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

# 5. Dummy input
input_ids = jnp.ones((1, 8), dtype=jnp.int32)
attention_mask = jnp.ones_like(input_ids)
position_ids = jnp.arange(8)[None, :]

# 6. Inference
with mesh:
    output = model.module.apply(
        {"params": sharded_params},
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        deterministic=True,
        return_dict=True,
    )

# 7. Save logits
print("✅ Logits shape:", output.logits.shape)
np.save("output_jax_sharded.npy", np.array(output.logits))
