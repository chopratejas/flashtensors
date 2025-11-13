#!/usr/bin/env python3
"""
Warm Swap Comparison Script

Runs both baseline and FlashTensors benchmarks and compares results.
Shows the speedup achieved by FlashTensors for model swapping.
"""

import subprocess
import sys
import re
import time


def run_benchmark(script_name, description):
    """Run a benchmark script and capture output"""
    print("\n" + "="*80)
    print(f"  Running: {description}")
    print("="*80)
    print()

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        elapsed = time.time() - start

        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Extract average warm swap time from output
        avg_time = None
        if "WARM SWAP AVERAGE" in result.stdout or "Average Warm Swap Time" in result.stdout:
            # Try to find the average time
            for line in result.stdout.split('\n'):
                if "Average" in line or "WARM SWAP AVERAGE" in line:
                    # Look for time in format like "10.234s"
                    matches = re.findall(r'(\d+\.\d+)s', line)
                    if matches:
                        # Take the third number (total time column)
                        if len(matches) >= 3:
                            avg_time = float(matches[2])
                        elif len(matches) >= 1:
                            avg_time = float(matches[-1])
                        break

        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "avg_time": avg_time,
            "elapsed": elapsed,
        }

    except subprocess.TimeoutExpired:
        print(f"❌ Benchmark timed out after 10 minutes")
        return {"success": False, "avg_time": None, "elapsed": 600}
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "avg_time": None, "elapsed": time.time() - start}


def main():
    print("\n" + "="*90)
    print("  WARM SWAP BENCHMARK COMPARISON")
    print("  Baseline vLLM (SafeTensors) vs FlashTensors")
    print("="*90)
    print()
    print("  This comparison measures COMPLETE model swap time including:")
    print("    - Weight loading from disk/storage to GPU")
    print("    - vLLM engine initialization")
    print("    - Memory allocation and CUDA graph compilation")
    print("    - First inference token")
    print()
    print("  Scenario: Swapping between models multiple times (warm swaps)")
    print("="*90)
    print()

    print("\nStarting baseline benchmark in 3 seconds...")
    time.sleep(3)

    # Run baseline
    print("\n" + "🔵 "*40)
    print("  PHASE 1: BASELINE - vLLM with SafeTensors")
    print("🔵 "*40)
    baseline_result = run_benchmark(
        "benchmark_baseline_warm_swap.py",
        "Baseline vLLM with SafeTensors"
    )

    if not baseline_result["success"]:
        print("\n❌ Baseline benchmark failed!")
        return

    print("\n✅ Baseline benchmark completed!")
    print("\nStarting FlashTensors benchmark in 3 seconds...")
    time.sleep(3)

    # Run FlashTensors
    print("\n" + "⚡ "*40)
    print("  PHASE 2: FLASHTENSORS - Ultra-Fast Model Swapping")
    print("⚡ "*40)
    flashtensors_result = run_benchmark(
        "benchmark_flashtensors_warm_swap.py",
        "FlashTensors Fast Model Swapping"
    )

    if not flashtensors_result["success"]:
        print("\n❌ FlashTensors benchmark failed!")
        return

    print("\n✅ FlashTensors benchmark completed!")

    # Compare results
    print("\n\n" + "="*90)
    print("  📊 COMPARISON RESULTS")
    print("="*90)
    print()

    if baseline_result["avg_time"] and flashtensors_result["avg_time"]:
        baseline_time = baseline_result["avg_time"]
        flashtensors_time = flashtensors_result["avg_time"]
        speedup = baseline_time / flashtensors_time
        time_saved = baseline_time - flashtensors_time
        percent_faster = ((baseline_time - flashtensors_time) / baseline_time) * 100

        print(f"  Baseline (vLLM + SafeTensors):")
        print(f"    Average Warm Swap Time: {baseline_time:.3f}s")
        print()
        print(f"  FlashTensors:")
        print(f"    Average Warm Swap Time: {flashtensors_time:.3f}s")
        print()
        print("  " + "-"*86)
        print()
        print(f"  🚀 SPEEDUP: {speedup:.2f}x faster")
        print(f"  ⏱️  TIME SAVED: {time_saved:.3f}s per swap ({percent_faster:.1f}% faster)")
        print()

        # Real-world impact
        print("  " + "-"*86)
        print()
        print("  💡 Real-World Impact:")
        print()
        print(f"    10 model swaps:")
        print(f"      Baseline:      {baseline_time * 10:.1f}s")
        print(f"      FlashTensors:  {flashtensors_time * 10:.1f}s")
        print(f"      Time saved:    {time_saved * 10:.1f}s")
        print()
        print(f"    100 model swaps:")
        print(f"      Baseline:      {baseline_time * 100 / 60:.1f} minutes")
        print(f"      FlashTensors:  {flashtensors_time * 100 / 60:.1f} minutes")
        print(f"      Time saved:    {time_saved * 100 / 60:.1f} minutes")
        print()
        print(f"    1000 model swaps:")
        print(f"      Baseline:      {baseline_time * 1000 / 3600:.1f} hours")
        print(f"      FlashTensors:  {flashtensors_time * 1000 / 3600:.1f} hours")
        print(f"      Time saved:    {time_saved * 1000 / 3600:.1f} hours")
        print()

    print("="*90)
    print()
    print("  ✅ Benchmark comparison complete!")
    print()
    print("  Key Takeaways:")
    print("    - FlashTensors keeps model weights in fast storage")
    print("    - Warm swaps avoid disk I/O completely")
    print("    - Direct GPU memory transfers for instant switching")
    print("    - Perfect for scenarios with frequent model swapping")
    print()
    print("="*90)
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
