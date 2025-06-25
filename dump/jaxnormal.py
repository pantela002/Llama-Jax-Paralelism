import time
import timeit
from jax import random
import jax.numpy as jnp

# Define SELU function (no JIT)
def selu(x):
    return jnp.where(x > 0, 1.0507 * x, 1.0507 * 1.67326 * (jnp.exp(x) - 1))

# Create input
key = random.key(0)
x = random.normal(key, (1_000_000,))

# -------------------------------
# 🟠 Measure FIRST CALL
start = time.time()
selu(x).block_until_ready()
end = time.time()
first_call_ms = (end - start) * 1000
print(f"⏱️ Non-JIT first call: {first_call_ms:.3f} ms")

# -------------------------------
# 🟢 Measure LATER CALLS
def run():
    selu(x).block_until_ready()

# Time it over 100 runs
total_time = timeit.timeit(run, number=100)
avg_time_ms = (total_time / 100) * 1000
print(f"🐌 Non-JIT average time per call: {avg_time_ms:.3f} ms")
