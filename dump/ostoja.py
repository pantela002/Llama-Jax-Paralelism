import torch

# Create a 3D tensor (Depth x Height x Width)
t = torch.randn(2, 3, dtype=torch.float32, device='cpu')

print("Tensor:", t)
print("Shape:", t.shape)
print("Dtype:", t.dtype)
print("Device:", t.device)
print("Strides:", t.stride())