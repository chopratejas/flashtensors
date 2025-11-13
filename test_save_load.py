#!/usr/bin/env python3
"""
Simple test to verify save/load mechanism
"""

import os
import torch
import sys
sys.path.insert(0, os.path.dirname(__file__))

from snacktensors.torch_storage import save_dict, load_dict

# Create a simple test tensor with known values
print("Creating test tensor...")
test_tensor = torch.arange(0, 1000, dtype=torch.float32).reshape(10, 100)
print(f"Original tensor: shape={test_tensor.shape}, dtype={test_tensor.dtype}")
print(f"First 10 values: {test_tensor.flatten()[:10]}")
print(f"Sum: {test_tensor.sum()}")

# Save it
save_path = "/tmp/test_tensor"
os.makedirs(save_path, exist_ok=True)
print(f"\nSaving to {save_path}...")
save_dict({"test": test_tensor}, save_path)
print("✅ Save completed")

# Load it back to CUDA
print(f"\nLoading from {save_path} to CUDA...")
device_map = {"": 0}  # Load to CUDA device 0
loaded_dict = load_dict(save_path, device_map, storage_path="/tmp")
loaded_tensor = loaded_dict["test"]

print(f"Loaded tensor: shape={loaded_tensor.shape}, dtype={loaded_tensor.dtype}, device={loaded_tensor.device}")
print(f"First 10 values: {loaded_tensor.flatten()[:10]}")
print(f"Sum: {loaded_tensor.sum()}")

# Compare (move loaded tensor back to CPU for comparison)
loaded_tensor_cpu = loaded_tensor.cpu()
if torch.allclose(test_tensor, loaded_tensor_cpu):
    print("\n✅ SUCCESS: Tensors match!")
else:
    print("\n❌ FAILURE: Tensors don't match!")
    print(f"Max diff: {(test_tensor - loaded_tensor_cpu).abs().max()}")
    print(f"Mean diff: {(test_tensor - loaded_tensor_cpu).abs().mean()}")
