import numpy as np
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import flax.linen as nn
from flax.core import freeze, unfreeze


# Input x: shape (batch=2, seq_len=3, dim=4)
batch, seq, in_dim, out_dim = 2, 3, 4, 8  # Adjust as needed
x = jnp.array([
    [[1,2,3,4], [5,6,7,8], [9,10,11,12]],
    [[13,14,15,16], [17,18,19,20], [21,22,23,24]]
], dtype=jnp.float32)
x = jnp.array(x, dtype=jnp.float32)
custom_kernel = [[1, 2, 3, 4, 5, 6, 7, 8],
                 [9, 10, 11, 12, 13, 14, 15, 16],
                 [19, 20, 21, 22, 23, 24, 25, 26],
                 [31, 32, 33, 34, 35, 36, 37, 38]]
custom_kernel = jnp.array(custom_kernel, dtype=jnp.float32)

# Matrix multiplication over last dim of x and first dim of kernel
# Output shape: (2, 3, 8)
output = np.einsum('bsd,df->bsf', x, custom_kernel)

print("Output shape:", output.shape)
print(output)
