#!/usr/bin/env python3
"""
Test save/load with CUDA tensor that's moved to CPU (like vLLM does)
"""

import os
import torch
import sys
sys.path.insert(0, os.path.dirname(__file__))

from snacktensors.torch_storage import save_dict, load_dict

# Create a tensor on CUDA
print("Creating test tensor on CUDA...")
test_tensor_cuda = torch.arange(0, 1000, dtype=torch.bfloat16, device='cuda').reshape(10, 100)
print(f"CUDA tensor: shape={test_tensor_cuda.shape}, dtype={test_tensor_cuda.dtype}, device={test_tensor_cuda.device}")
print(f"First 10 values: {test_tensor_cuda.flatten()[:10]}")
print(f"Sum: {test_tensor_cuda.sum()}")

# Move to CPU and make contiguous (like vLLM does)
print("\nMoving to CPU with .cpu().contiguous()...")
test_tensor_cpu = test_tensor_cuda.cpu().contiguous()
torch.cuda.synchronize()  # Make sure CUDA operations are done
print(f"CPU tensor: shape={test_tensor_cpu.shape}, dtype={test_tensor_cpu.dtype}, device={test_tensor_cpu.device}")
print(f"First 10 values: {test_tensor_cpu.flatten()[:10]}")
print(f"Sum: {test_tensor_cpu.sum()}")

# Save it
save_path = "/tmp/test_cuda_tensor"
os.makedirs(save_path, exist_ok=True)
print(f"\nSaving to {save_path}...")
save_dict({"test": test_tensor_cpu}, save_path)
print("✅ Save completed")

# Load it back to CUDA
print(f"\nLoading from {save_path} to CUDA...")
device_map = {"": 0}
loaded_dict = load_dict(save_path, device_map, storage_path="/tmp")
loaded_tensor = loaded_dict["test"]

print(f"Loaded tensor: shape={loaded_tensor.shape}, dtype={loaded_tensor.dtype}, device={loaded_tensor.device}")
print(f"First 10 values: {loaded_tensor.flatten()[:10]}")
print(f"Sum: {loaded_tensor.sum()}")

# Compare
if torch.allclose(test_tensor_cuda, loaded_tensor):
    print("\n✅ SUCCESS: Tensors match!")
else:
    print("\n❌ FAILURE: Tensors don't match!")
    diff = (test_tensor_cuda - loaded_tensor).abs()
    print(f"Max diff: {diff.max()}")
    print(f"Mean diff: {diff.mean()}")
    print(f"\nOriginal values: {test_tensor_cuda.flatten()[:20]}")
    print(f"Loaded values:   {loaded_tensor.flatten()[:20]}")
