import torch
import numpy as np
from transformers import LlamaForCausalLM
from jax import numpy as jnp
# Load LLaMA 3.1-8B model
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model.eval()

# Simulate a small input
batch_size = 1
seq_len = 6
head_dim = 128  # LLaMA 3.1-8B: hidden_size=4096, num_heads=32 → head_dim = 128

# Dummy tensor, shape = [batch, seq_len, head_dim]
#dummy_hidden = torch.zeros(batch_size, seq_len, head_dim)
dummy_hidden = torch.randn(batch_size, seq_len, head_dim)


# Position IDs = [0, 1, 2, ..., seq_len-1]
position_ids = torch.arange(seq_len).unsqueeze(0)

# Call rotary embedding module correctly
cos, sin = model.model.rotary_emb(dummy_hidden, position_ids)

print("cos shape:", cos.shape)  # Should be [1, 6, 128]
print("sin shape:", sin.shape)
print("cos[0, 0, :10]:", cos[0, 0, :100])

# Save for JAX comparison
np.save("tests_/rotart_embedding/hf_cos.npy", cos.detach().cpu().numpy())
np.save("tests_/rotart_embedding/hf_sin.npy", sin.detach().cpu().numpy())

hf_cos = jnp.array(np.load("hf_cos.npy"))
hf_sin = jnp.array(np.load("hf_sin.npy"))

print("Loaded shapes:", hf_cos.shape, hf_sin.shape)
print("hf_cos[0, 0, :10]:", hf_cos[0, 0, :10])
np.savetxt("tests_/rotart_embedding/hf_cos.txt", cos.view(-1, head_dim).cpu().numpy(), fmt="%.6f")
np.savetxt("tests_/rotart_embedding/hf_sin.txt", sin.view(-1, head_dim).cpu().numpy(), fmt="%.6f")