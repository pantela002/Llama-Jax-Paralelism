import time
import timeit
from jax import jit, random
import jax.numpy as jnp

# Define SELU function
def selu(x):
    return jnp.where(x > 0, 1.0507 * x, 1.0507 * 1.67326 * (jnp.exp(x) - 1))

# JIT compile it
selu_jit = jit(selu)

# Create input
key = random.key(0)
x = random.normal(key, (1_000_000,))

# -------------------------------
# 🟠 Measure FIRST CALL (compilation + execution)
start = time.time()
selu_jit(x).block_until_ready()
end = time.time()
first_call_ms = (end - start) * 1000
print(f"⏱️ First call (includes compile): {first_call_ms:.3f} ms")

# -------------------------------
# 🟢 Measure LATER CALLS (cached, fast)
def run():
    selu_jit(x).block_until_ready()

# Use timeit to repeat the run 100 times
total_time = timeit.timeit(run, number=100)
avg_time_ms = (total_time / 100) * 1000
print(f"⚡ JIT-compiled SELU average time: {avg_time_ms:.3f} ms")
