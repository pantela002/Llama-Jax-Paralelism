import os

# Simulate 8 CPU devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import numpy as np
import jax
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from flax import nnx

print(f"Available devices: {jax.devices()}")

# Set up a 2x4 mesh
mesh = Mesh(devices=np.array(jax.devices()).reshape(2, 4), axis_names=('data', 'model'))
print("Mesh created:", mesh)

# Define a very tiny example model
class TinyModel(nnx.Module):
    def __init__(self, depth: int, rngs: nnx.Rngs):
        init_fn = nnx.initializers.lecun_normal()
        
        # Kernel with sharding annotations
        self.kernel = nnx.Param(
            init_fn(rngs.params(), (depth, depth)),
            sharding=(None, 'model')
        )

    def __call__(self, x: jax.Array):
        return jnp.dot(x, self.kernel.value)

# JIT-compile and shard the model
@nnx.jit
def create_sharded_model():
    model = TinyModel(depth=8, rngs=nnx.Rngs(0))  # Tiny depth=8 for fast testing
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model

# Create the sharded model inside the mesh
with mesh:
    model = create_sharded_model()

# Print the sharding
print("Model kernel sharding:", model.kernel.value.sharding)

# Visualize how it's split across devices
jax.debug.visualize_array_sharding(model.kernel.value)
