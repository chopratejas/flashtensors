#!/usr/bin/env python3
"""
Direct test of H2D transfer performance with multi-stream optimization.
"""

import torch
import time
import numpy as np


def test_h2d_transfer(size_gb=14, num_iterations=5):
    """Test H2D transfer speed simulating 7B model"""

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    print("=" * 70)
    print("  H2D TRANSFER BENCHMARK - Phase 1 Multi-Stream Optimization")
    print("=" * 70)
    print()

    # Convert GB to bytes
    size_bytes = int(size_gb * 1024**3)

    # Our chunk size (same as snacktensors)
    chunk_size = 32 * 1024**2  # 32MB
    num_chunks = size_bytes // chunk_size

    print(f"Transfer size: {size_gb}GB")
    print(f"Chunk size: {chunk_size / (1024**2):.0f}MB")
    print(f"Number of chunks: {num_chunks}")
    print()

    # Allocate pinned host memory (like snacktensors does)
    print("Allocating pinned host memory...")
    host_arrays = []
    for i in range(min(num_chunks, 16)):  # Limit to 16 chunks for test
        arr = torch.empty(chunk_size // 4, dtype=torch.float32, pin_memory=True)
        host_arrays.append(arr)

    actual_chunks = len(host_arrays)
    actual_size_gb = (actual_chunks * chunk_size) / (1024**3)

    print(f"✅ Allocated {actual_chunks} chunks ({actual_size_gb:.2f}GB)")
    print()

    # Test sequential transfers (baseline)
    print("[1/2] Testing SEQUENTIAL transfers (baseline)...")
    sequential_times = []

    for iter in range(num_iterations):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        start = time.perf_counter()

        for arr in host_arrays:
            gpu_tensor = arr.cuda(non_blocking=False)  # Synchronous

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        sequential_times.append(elapsed)

        print(f"  Iteration {iter+1}: {elapsed*1000:.1f}ms")

    seq_mean = np.mean(sequential_times) * 1000
    seq_std = np.std(sequential_times) * 1000
    seq_bandwidth = actual_size_gb / np.mean(sequential_times)

    print(f"  Mean: {seq_mean:.1f}ms ± {seq_std:.1f}ms")
    print(f"  Bandwidth: {seq_bandwidth:.2f} GB/s")
    print()

    # Test async transfers with streams (our optimization)
    print("[2/2] Testing ASYNC MULTI-STREAM transfers (Phase 1)...")
    num_streams = 8
    async_times = []

    for iter in range(num_iterations):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Create streams (like our C++ code does)
        streams = [torch.cuda.Stream() for _ in range(num_streams)]

        start = time.perf_counter()

        # Distribute chunks across streams
        for idx, arr in enumerate(host_arrays):
            stream = streams[idx % num_streams]
            with torch.cuda.stream(stream):
                gpu_tensor = arr.cuda(non_blocking=True)  # Async

        # Synchronize all streams (like our C++ code)
        for stream in streams:
            stream.synchronize()

        elapsed = time.perf_counter() - start
        async_times.append(elapsed)

        print(f"  Iteration {iter+1}: {elapsed*1000:.1f}ms")

    async_mean = np.mean(async_times) * 1000
    async_std = np.std(async_times) * 1000
    async_bandwidth = actual_size_gb / np.mean(async_times)

    print(f"  Mean: {async_mean:.1f}ms ± {async_std:.1f}ms")
    print(f"  Bandwidth: {async_bandwidth:.2f} GB/s")
    print()

    # Calculate speedup
    speedup = seq_mean / async_mean
    bandwidth_improvement = async_bandwidth / seq_bandwidth

    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print(f"Sequential (baseline):  {seq_mean:.1f}ms  ({seq_bandwidth:.2f} GB/s)")
    print(f"Multi-stream (Phase 1): {async_mean:.1f}ms  ({async_bandwidth:.2f} GB/s)")
    print()
    print(f"🚀 SPEEDUP: {speedup:.2f}x faster!")
    print(f"📈 BANDWIDTH: {bandwidth_improvement:.2f}x higher!")
    print()

    # Extrapolate to full 7B model
    full_7b_seq = (size_gb / actual_size_gb) * seq_mean
    full_7b_async = (size_gb / actual_size_gb) * async_mean

    print(f"Extrapolated to full {size_gb}GB (7B model):")
    print(f"  Sequential: {full_7b_seq:.0f}ms ({full_7b_seq/1000:.2f}s)")
    print(f"  Multi-stream: {full_7b_async:.0f}ms ({full_7b_async/1000:.2f}s)")
    print(f"  Savings: {full_7b_seq - full_7b_async:.0f}ms")
    print()

    return {
        'sequential': seq_mean,
        'async': async_mean,
        'speedup': speedup,
        'bandwidth_improvement': bandwidth_improvement
    }


if __name__ == "__main__":
    test_h2d_transfer(size_gb=14, num_iterations=5)
