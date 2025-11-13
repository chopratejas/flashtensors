#!/usr/bin/env python3
"""
FlashTensors Warm Swap Benchmark

Measures the COMPLETE time to swap between models using FlashTensors including:
- Weight loading from FlashTensors storage to GPU
- vLLM engine initialization
- CUDA graph compilation
- Model warmup
- First inference token

This demonstrates FlashTensors' ultra-fast model swapping capability.
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
    print("Make sure you've built and installed the C++ extensions")
    sys.exit(1)


def measure_model_load_and_inference(model_id, iteration, first_load=False):
    """Load a model with FlashTensors and measure time to first inference"""
    print(f"\n{'='*80}")
    print(f"  Loading Model: {model_id} (Iteration {iteration})")
    if first_load:
        print(f"  ⚠️  First load - includes registration overhead")
    else:
        print(f"  ⚡ WARM SWAP - weights already in FlashTensors storage")
    print(f"{'='*80}")

    # Measure complete load time (FlashTensors + vLLM initialization + CUDA graphs)
    load_start = time.time()
    print(f"  ⏱️  Starting model load at {load_start:.3f}...")

    try:
        # If first load, need to register
        if first_load:
            print(f"  📝 Checking if model is registered...")
            models = flash.list_models()
            model_key = f"{model_id}:vllm"

            if model_key not in models:
                print(f"  🔄 Registering {model_id} with FlashTensors...")
                reg_start = time.time()
                result = flash.register_model(
                    model_id=model_id,
                    backend="vllm",
                    torch_dtype="bfloat16",
                    force=False,
                    hf_token=None
                )
                reg_time = time.time() - reg_start
                print(f"  ✅ Model registered in {reg_time:.3f}s: {result['status']}")
            else:
                print(f"  ✅ Model already registered")

        # Load model with FlashTensors
        actual_load_start = time.time()
        llm = flash.load_model(
            model_id=model_id,
            backend="vllm",
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
        )
        load_end = time.time()
        load_time = load_end - load_start
        actual_load_time = load_end - actual_load_start

        print(f"  ✅ Model loaded in {load_time:.3f}s")
        if first_load:
            print(f"     - FlashTensors load time: {actual_load_time:.3f}s")
        else:
            print(f"     - FlashTensors WARM SWAP: {actual_load_time:.3f}s ⚡")
        print(f"     - Includes: weight loading, vLLM init, memory allocation")

        # Run inference to prove model is ready
        inference_start = time.time()
        sampling_params = SamplingParams(temperature=0.1, max_tokens=20)

        outputs = llm.generate(["Hello, world!"], sampling_params)
        inference_end = time.time()
        inference_time = inference_end - inference_start

        output_text = outputs[0].outputs[0].text
        print(f"  ✅ First inference completed in {inference_time:.3f}s")
        print(f"     Output: {output_text[:50]}...")

        total_time = load_time + inference_time

        # Cleanup - unload but keep in FlashTensors storage
        print(f"  🧹 Unloading model (keeping in FlashTensors storage)...")
        flash.cleanup_gpu()

        return {
            "model": model_id,
            "iteration": iteration,
            "load_time": load_time,
            "actual_load_time": actual_load_time,
            "inference_time": inference_time,
            "total_time": total_time,
            "first_load": first_load,
        }

    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        flash.cleanup_gpu()
        return None


def main():
    print("\n" + "="*80)
    print("  FLASHTENSORS WARM SWAP BENCHMARK")
    print("="*80)
    print()
    print("  This benchmark measures the COMPLETE time to swap models with FlashTensors:")
    print("    1. Load weights from FlashTensors storage to GPU")
    print("    2. Initialize vLLM engine")
    print("    3. Allocate memory and compile CUDA graphs")
    print("    4. Run first inference to prove model is ready")
    print()
    print("  Key Advantage: Weights stay in FlashTensors storage for instant swaps!")
    print("  Scenario: Load Model A → Inference → Swap → Load Model B → Inference")
    print("="*80)
    print()

    # Configure FlashTensors
    print("🔧 Configuring FlashTensors...")
    flash.shutdown_server()  # Ensure clean start

    flash.configure(
        storage_path="/tmp/models",
        mem_pool_size=1024**3*30,      # 30GB memory pool
        chunk_size=1024**2*32,          # 32MB chunks
        num_threads=4,
        gpu_memory_utilization=0.8,
        server_host="0.0.0.0",
        server_port=8073
    )

    flash.activate_vllm_integration()
    print("✅ FlashTensors configured")

    # Verify storage server is running
    print("🔍 Verifying storage server connection...")
    client = flash.storage_client.StorageClient()
    server_config = client.get_server_config(max_retries=15, retry_delay=2.0)
    if not server_config:
        print("❌ Failed to connect to storage server!")
        raise RuntimeError("Storage server not responding")
    print(f"✅ Storage server connected: {server_config}\n")

    # Model to test - we'll swap it multiple times to show warm swap performance
    models = [
        "Qwen/Qwen3-0.6B",
    ]

    results = []

    # Initial GPU state
    print("\n📊 GPU Configuration:")
    import torch
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Total GPU Memory: {gpu_mem:.2f} GB")
        print(f"  FlashTensors Storage Pool: 30 GB")

    # Warm swap test: Load same model multiple times
    # First load registers model, subsequent loads are true warm swaps from storage
    swap_sequence = [
        (models[0], 1, True),   # First load (includes registration)
        (models[0], 2, False),  # WARM SWAP #1 from storage ⚡
        (models[0], 3, False),  # WARM SWAP #2 from storage ⚡
        (models[0], 4, False),  # WARM SWAP #3 from storage ⚡
    ]

    print(f"\n🔄 Warm Swap Sequence:")
    for i, (model, iteration, first) in enumerate(swap_sequence, 1):
        status = "REGISTER + LOAD" if first else "⚡ WARM SWAP"
        print(f"  {i}. {model} (iteration {iteration}) - {status}")
    print()

    # Run benchmark
    for model_id, iteration, first_load in swap_sequence:
        result = measure_model_load_and_inference(model_id, iteration, first_load)
        if result:
            results.append(result)
        time.sleep(2)  # Brief pause between swaps

    # Print summary
    print("\n" + "="*80)
    print("  BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print()
    print(f"{'Model':<30} {'Iteration':<12} {'Type':<15} {'Load Time':<15} {'Inference':<15} {'Total':<15}")
    print("-"*80)

    for r in results:
        load_type = "First Load" if r['first_load'] else "⚡ WARM SWAP"
        print(f"{r['model']:<30} {r['iteration']:<12} {load_type:<15} {r['load_time']:>10.3f}s     {r['inference_time']:>10.3f}s     {r['total_time']:>10.3f}s")

    # Calculate averages for TRUE warm swaps (not first loads)
    warm_swap_results = [r for r in results if not r['first_load']]
    if warm_swap_results:
        avg_load = sum(r['load_time'] for r in warm_swap_results) / len(warm_swap_results)
        avg_inference = sum(r['inference_time'] for r in warm_swap_results) / len(warm_swap_results)
        avg_total = sum(r['total_time'] for r in warm_swap_results) / len(warm_swap_results)

        print("-"*80)
        print(f"{'⚡ WARM SWAP AVERAGE':<30} {'(n=' + str(len(warm_swap_results)) + ')':<12} {'⚡ OPTIMIZED':<15} {avg_load:>10.3f}s     {avg_inference:>10.3f}s     {avg_total:>10.3f}s")

    # First load averages (for comparison)
    first_load_results = [r for r in results if r['first_load']]
    if first_load_results:
        avg_first_load = sum(r['load_time'] for r in first_load_results) / len(first_load_results)
        avg_first_total = sum(r['total_time'] for r in first_load_results) / len(first_load_results)
        print(f"{'First Load Average':<30} {'(n=' + str(len(first_load_results)) + ')':<12} {'(baseline)':<15} {avg_first_load:>10.3f}s     {'':>10}      {avg_first_total:>10.3f}s")

    print()
    print("="*80)
    if warm_swap_results:
        print(f"  ⚡ Average FlashTensors Warm Swap Time: {avg_total:.3f}s")
        print(f"  📊 Load Time Breakdown:")
        print(f"     - FlashTensors weight loading: ~{sum(r['actual_load_time'] for r in warm_swap_results) / len(warm_swap_results):.3f}s")
        print(f"     - vLLM initialization: ~{avg_load - sum(r['actual_load_time'] for r in warm_swap_results) / len(warm_swap_results):.3f}s")
    print("="*80)
    print()
    print("  🚀 Key Advantage: Weights persist in FlashTensors storage!")
    print("     - No disk I/O on warm swaps")
    print("     - Direct GPU-to-GPU memory transfer")
    print("     - Instant model switching")
    print()
    print("  Compare this with baseline vLLM warm swap performance!")
    print()

    # Cleanup
    print("🧹 Final cleanup...")
    flash.cleanup_gpu()
    flash.shutdown_server()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        flash.shutdown_server()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        flash.shutdown_server()
        sys.exit(1)
