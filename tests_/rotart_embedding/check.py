import numpy as np

jax = np.loadtxt("tests_/rotart_embedding/jax_sin.txt")
hf = np.loadtxt("tests_/rotart_embedding/hf_sin.txt")

print("Max difference:", np.max(np.abs(jax - hf)))
