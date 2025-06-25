import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import flax.linen as nn
from flax.core import freeze, unfreeze
from jax import debug

# Setup mesh
devices = jax.devices()
print("Devices:", devices)
mesh = Mesh(devices, axis_names=('mp',))


class ParallelDense(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]
        out_dim = self.features
        local_shape = (in_dim, out_dim)
        
        
        kernel = self.param("kernel", nn.initializers.lecun_normal(), local_shape, self.param_dtype)

        
        def matmul_fn(x, k):
            axis_idx = jax.lax.axis_index("mp")
            debug.print("🔧 Device {}/{} running matmul: x.shape = {}, kernel.shape = {}", 
                        axis_idx, mesh.shape['mp'], x.shape, k.shape)
            
            local_out = jnp.einsum('bsd,df->bsf', x, k)

            full_out = jax.lax.all_gather(local_out, axis_name="mp", axis=0)

            return jnp.reshape(jnp.transpose(full_out, (1, 2, 0, 3)), (x.shape[0], x.shape[1], -1))


        # Note: we replicate x, shard only kernel
        return shard_map(
            matmul_fn,
            mesh=mesh,
            in_specs=(None, P(None, "mp")),     # x is replicated, kernel is sharded on output dim
            out_specs=P(None),            # output is sharded along output dim
            check_rep=False
        )(x, kernel)



batch, seq, in_dim, out_dim = 2, 3, 4, 8  # Adjust as needed
x = jnp.arange(batch * seq * in_dim).reshape((batch, seq, in_dim)).astype(jnp.float32)
custom_kernel = jnp.ones((in_dim, out_dim)).astype(jnp.float32)

model = ParallelDense(features=8)

params = model.init(jax.random.PRNGKey(0), x)
params_unfrozen = unfreeze(params)
params_unfrozen["params"]["kernel"] = custom_kernel
params = freeze(params_unfrozen)


y = model.apply(params, x)

print("Final output shape:", y.shape)
print(y)
