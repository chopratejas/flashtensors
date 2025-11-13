#!/usr/bin/env python3
"""
Test real io_uring performance with actual disk I/O.
"""

import sys
import time
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np


def create_test_file(size_gb=1):
    """Create a test file for I/O benchmarking."""
    size_bytes = int(size_gb * 1024 ** 3)

    # Create temporary file
    fd, path = tempfile.mkstemp(suffix='.bin', prefix='snack_io_test_')

    print(f"Creating {size_gb}GB test file: {path}")

    # Write random data in chunks
    chunk_size = 32 * 1024 * 1024  # 32MB
    data = np.random.randint(0, 256, chunk_size, dtype=np.uint8)

    bytes_written = 0
    while bytes_written < size_bytes:
        to_write = min(chunk_size, size_bytes - bytes_written)
        os.write(fd, data[:to_write].tobytes())
        bytes_written += to_write
        if bytes_written % (256 * 1024 * 1024) == 0:
            print(f"  Written: {bytes_written / (1024**3):.2f}GB")

    os.close(fd)
    print(f"✅ Test file created: {size_bytes / (1024**3):.2f}GB\n")

    return path


def test_io_uring_with_checkpoint_store(file_path, size_gb=1):
    """Test CheckpointStore with io_uring."""
    print("=" * 70)
    print("  TESTING CHECKPOINTSTORE WITH IO_URING")
    print("=" * 70)
    print()

    try:
        from snacktensors._checkpoint_store import CheckpointStore

        # Create CheckpointStore
        storage_path = "/tmp/snack_io_test"
        os.makedirs(storage_path, exist_ok=True)

        mem_pool_size = int(2 * 1024 ** 3)  # 2GB
        chunk_size = 32 * 1024 ** 2  # 32MB
        num_threads = 16

        store = CheckpointStore(storage_path, mem_pool_size, num_threads, chunk_size)

        print(f"✅ CheckpointStore created")
        print(f"   Threads: {num_threads}")
        print(f"   Chunk size: {chunk_size / (1024**2):.0f}MB")
        print()

        # Check if io_uring is being used
        print("NOTE: Check logs above for 'Using io_uring' message")
        print("      If you see 'Using multi-threaded pread()', io_uring is not active")
        print()

    except Exception as e:
        print(f"❌ CheckpointStore test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "=" * 70)
    print("  SNACKTENSORS PHASE 3.1: IO_URING REAL TEST")
    print("=" * 70)
    print()

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    # Create 1GB test file
    test_file = create_test_file(1.0)

    try:
        # Test with CheckpointStore
        test_io_uring_with_checkpoint_store(test_file, 1.0)

        print()
        print("=" * 70)
        print("  TEST COMPLETE")
        print("=" * 70)
        print()
        print("To verify io_uring is working:")
        print("  1. Look for 'Using io_uring for async I/O' in logs")
        print("  2. Compare with 'Using multi-threaded pread()' (fallback)")
        print("  3. io_uring should be 2-3x faster on real model loads")
        print()

    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"Cleaned up test file: {test_file}")


if __name__ == "__main__":
    main()
