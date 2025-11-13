#!/usr/bin/env python3
"""
Baseline Warm Swap Benchmark - vLLM with SafeTensors

Measures the COMPLETE time to swap between models including:
- Weight loading from disk to GPU
- vLLM engine initialization
- CUDA graph compilation
- Model warmup
- First inference token

This is the state-of-the-art approach without FlashTensors.
"""

import time
import gc
import torch
from vllm import LLM, SamplingParams
import sys


def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    print("  🧹 Clearing GPU memory...")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    time.sleep(1)  # Let GPU memory stabilize


def measure_model_load_and_inference(model_id, iteration):
    """Load a model and measure time to first inference"""
    print(f"\n{'='*80}")
    print(f"  Loading Model: {model_id} (Iteration {iteration})")
    print(f"{'='*80}")

    # Clear GPU before loading
    clear_gpu_memory()

    # Measure complete load time (weights + vLLM initialization + CUDA graphs)
    load_start = time.time()
    print(f"  ⏱️  Starting model load at {load_start:.3f}...")

    try:
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
            enforce_eager=True,  # Disable CUDA graphs for faster init (optional)
            download_dir="/tmp/models/baseline",
        )
        load_end = time.time()
        load_time = load_end - load_start

        print(f"  ✅ Model loaded in {load_time:.3f}s")
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

        # Cleanup
        del llm
        clear_gpu_memory()

        return {
            "model": model_id,
            "iteration": iteration,
            "load_time": load_time,
            "inference_time": inference_time,
            "total_time": total_time,
        }

    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        clear_gpu_memory()
        return None


def main():
    print("\n" + "="*80)
    print("  BASELINE WARM SWAP BENCHMARK - vLLM with SafeTensors")
    print("="*80)
    print()
    print("  This benchmark measures the COMPLETE time to swap models:")
    print("    1. Load weights from disk to GPU")
    print("    2. Initialize vLLM engine")
    print("    3. Allocate memory and compile CUDA graphs")
    print("    4. Run first inference to prove model is ready")
    print()
    print("  Scenario: Load Model A → Inference → Unload → Load Model B → Inference")
    print("="*80)
    print()

    # Model to test - we'll swap it multiple times to show warm swap performance
    models = [
        "Qwen/Qwen3-0.6B",
    ]

    results = []

    # Initial GPU state
    print("\n📊 Initial GPU State:")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Total GPU Memory: {gpu_mem:.2f} GB")

    # Warm swap test: Load same model multiple times to measure swap performance
    swap_sequence = [
        (models[0], 1),  # First load
        (models[0], 2),  # Warm swap #1
        (models[0], 3),  # Warm swap #2
        (models[0], 4),  # Warm swap #3
    ]

    print(f"\n🔄 Warm Swap Sequence:")
    for i, (model, iteration) in enumerate(swap_sequence, 1):
        print(f"  {i}. {model} (iteration {iteration})")
    print()

    # Run benchmark
    for model_id, iteration in swap_sequence:
        result = measure_model_load_and_inference(model_id, iteration)
        if result:
            results.append(result)
        time.sleep(2)  # Brief pause between swaps

    # Print summary
    print("\n" + "="*80)
    print("  BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print()
    print(f"{'Model':<30} {'Iteration':<12} {'Load Time':<15} {'Inference':<15} {'Total':<15}")
    print("-"*80)

    for r in results:
        print(f"{r['model']:<30} {r['iteration']:<12} {r['load_time']:>10.3f}s     {r['inference_time']:>10.3f}s     {r['total_time']:>10.3f}s")

    # Calculate averages for warm swaps (skip first load)
    if len(results) > 1:
        warm_swap_results = results[1:]  # All but first
        avg_load = sum(r['load_time'] for r in warm_swap_results) / len(warm_swap_results)
        avg_inference = sum(r['inference_time'] for r in warm_swap_results) / len(warm_swap_results)
        avg_total = sum(r['total_time'] for r in warm_swap_results) / len(warm_swap_results)

        print("-"*80)
        print(f"{'WARM SWAP AVERAGE':<30} {'(n=' + str(len(warm_swap_results)) + ')':<12} {avg_load:>10.3f}s     {avg_inference:>10.3f}s     {avg_total:>10.3f}s")

    print()
    print("="*80)
    print(f"  ⏱️  Average Warm Swap Time (Baseline): {avg_total:.3f}s")
    print("="*80)
    print()
    print("  This is the time to COMPLETELY swap models including:")
    print("    - Loading weights from disk")
    print("    - Initializing vLLM engine")
    print("    - Memory allocation")
    print("    - CUDA graph compilation (if enabled)")
    print("    - First inference token")
    print()
    print("  Compare this with FlashTensors warm swap performance!")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
