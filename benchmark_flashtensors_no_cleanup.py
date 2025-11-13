#!/usr/bin/env python3
"""
FlashTensors Warm Swap Benchmark - NO CLEANUP VERSION

This version avoids the storage server crash by NOT calling cleanup_gpu()
between iterations. Instead, we just measure the raw load time repeatedly.

This is NOT a true warm swap test, but helps us understand performance.
"""

import time
import sys
import os
from vllm import SamplingParams

# Add flashtensors to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    import flashtensors as flash
except ImportError as e:
    print(f"Failed to import flashtensors: {e}")
    sys.exit(1)


def measure_model_load(model_id, iteration):
    """Load a model and measure time"""
    print(f"\n{'='*80}")
    print(f"  Loading Model: {model_id} (Iteration {iteration})")
    print(f"{'='*80}")

    load_start = time.time()

    try:
        llm = flash.load_model(
            model_id=model_id,
            backend="vllm",
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
        )
        load_time = time.time() - load_start

        print(f"  ✅ Model loaded in {load_time:.3f}s")

        # Quick inference test
        inference_start = time.time()
        sampling_params = SamplingParams(temperature=0.1, max_tokens=10)
        outputs = llm.generate(["Hello"], sampling_params)
        inference_time = time.time() - inference_start

        print(f"  ✅ Inference: {inference_time:.3f}s")

        # DO NOT CLEANUP - this is the key difference
        print(f"  ⚠️  NOT calling cleanup_gpu() - leaving model loaded")

        return {
            "iteration": iteration,
            "load_time": load_time,
            "inference_time": inference_time,
        }

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\n" + "="*80)
    print("  FLASHTENSORS LOAD BENCHMARK (NO CLEANUP)")
    print("="*80)
    print()
    print("  ⚠️  This is NOT a true warm swap test!")
    print("  We avoid cleanup_gpu() to work around storage server crash.")
    print()

    # Configure
    print("🔧 Configuring FlashTensors...")
    flash.shutdown_server()

    flash.configure(
        storage_path="/tmp/models",
        mem_pool_size=1024**3*30,
        chunk_size=1024**2*32,
        num_threads=4,
        gpu_memory_utilization=0.8,
        server_host="0.0.0.0",
        server_port=8073
    )

    flash.activate_vllm_integration()
    print("✅ FlashTensors configured")

    # Verify server
    print("🔍 Verifying storage server...")
    client = flash.storage_client.StorageClient()
    server_config = client.get_server_config(max_retries=15, retry_delay=2.0)
    if not server_config:
        print("❌ Storage server failed!")
        raise RuntimeError("Storage server not responding")
    print(f"✅ Storage server ready\n")

    model_id = "Qwen/Qwen3-0.6B"

    # Register model first
    print(f"🔄 Registering {model_id}...")
    models = flash.list_models()
    model_key = f"{model_id}:vllm"

    if model_key not in models:
        result = flash.register_model(
            model_id=model_id,
            backend="vllm",
            torch_dtype="bfloat16",
        )
        print(f"✅ Registered: {result['status']}\n")
    else:
        print(f"✅ Already registered\n")

    # Load 3 times without cleanup
    results = []
    for i in range(1, 4):
        result = measure_model_load(model_id, i)
        if result:
            results.append(result)
        time.sleep(1)

    # Summary
    print("\n" + "="*80)
    print("  RESULTS")
    print("="*80)
    for r in results:
        print(f"  Iteration {r['iteration']}: Load={r['load_time']:.3f}s, Inference={r['inference_time']:.3f}s")

    if len(results) > 1:
        avg_load = sum(r['load_time'] for r in results[1:]) / len(results[1:])
        print(f"\n  Average (iterations 2-{len(results)}): {avg_load:.3f}s")

    print("="*80)
    print()

    # Cleanup
    flash.cleanup_gpu()
    flash.shutdown_server()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        flash.shutdown_server()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        flash.shutdown_server()
        sys.exit(1)
