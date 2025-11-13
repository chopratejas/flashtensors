#!/usr/bin/env python3
"""Simple test: load once, no swapping"""
import flashtensors as flash

flash.shutdown_server()
flash.configure(
    storage_path="/tmp/models",
    mem_pool_size=1024**3*30,
    chunk_size=1024**2*32,
    num_threads=4,
)
flash.activate_vllm_integration()

# Register
flash.register_model(model_id="Qwen/Qwen3-0.6B", backend="vllm", torch_dtype="bfloat16")

# Load once
print("Loading...")
llm = flash.load_model(model_id="Qwen/Qwen3-0.6B", backend="vllm", dtype="bfloat16")
print("✅ SUCCESS!")

# Cleanup
flash.cleanup_gpu()
print("✅ Cleanup done!")

flash.shutdown_server()
