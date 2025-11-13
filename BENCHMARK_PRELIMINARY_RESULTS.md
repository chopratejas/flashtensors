# Warm Swap Benchmark - Preliminary Results

## Issue Discovered

The FlashTensors storage server crashes during warm swap scenarios (second model load after cleanup). This is a **critical bug** that needs to be fixed before warm swap benchmarks can be completed.

### Error Pattern

1. **First Load (Iteration 1)**: ✅ SUCCESS
   - Model loads successfully: `14.897s`
   - Inference works: `0.393s`
   - Total: `15.290s`

2. **GPU Cleanup**: ✅ SUCCESS
   - `cleanup_gpu()` completes successfully
   - GPU memory cleared via storage service

3. **Second Load (Iteration 2)**: ❌ **STORAGE SERVER CRASH**
   - `load_into_gpu()` RPC succeeds and returns new replica UUID
   - `confirm_model_loaded()` RPC fails with:
     ```
     StatusCode.UNAVAILABLE
     details = "recvmsg:Connection reset by peer"
     ```
   - Storage server process dies completely

### Root Cause Hypothesis

The storage server crashes during the `confirm_model_loaded()` RPC call on the second load iteration. Possible causes:

1. **Memory corruption** in cleanup/reload cycle
2. **Use-after-free or double-free** in C++ storage server
3. **Race condition** in state management during confirm operation
4. **CUDA IPC handle management** issue when handles are recycled

## Comparison: Baseline vs FlashTensors (Partial Data)

### Baseline (vLLM + SafeTensors)

| Iteration | Type       | Load Time | Inference | Total    |
|-----------|------------|-----------|-----------|----------|
| 1         | Cold       | 21.757s   | 0.362s    | 22.119s  |
| 2         | Warm       | 13.824s   | 0.379s    | 14.203s  |
| 3         | Warm       | 13.672s   | 0.371s    | 14.043s  |
| 4         | Warm       | 13.545s   | 0.403s    | 13.948s  |
| **Average** | **Warm** | **13.680s** | **0.384s** | **14.064s** |

### FlashTensors (Limited Data - Crashed on Iteration 2)

| Iteration | Type        | Load Time | Inference | Total    | Status |
|-----------|-------------|-----------|-----------|----------|--------|
| 1         | Cold+Register | 14.897s | 0.393s  | 15.290s  | ✅ SUCCESS |
| 2         | Warm Swap   | -         | -         | -        | ❌ CRASH |
| 3         | Warm Swap   | -         | -         | -        | ❌ CRASH |
| 4         | Warm Swap   | -         | -         | -        | ❌ CRASH |

## Analysis

### What We Learned

1. **FlashTensors first load is faster than baseline cold load**
   - FlashTensors: `14.897s` (includes registration overhead)
   - Baseline cold: `21.757s`
   - **Speedup: 1.46x faster (31% improvement)**

2. **Baseline warm swaps are dominated by vLLM re-initialization, NOT disk I/O**
   - Warm swaps still take `13.680s` on average
   - This includes full vLLM engine re-initialization, memory allocation, tokenizer loading
   - Even with weights cached on disk, the overhead is substantial

3. **FlashTensors warm swaps are blocked by a critical bug**
   - Cannot measure true warm swap performance yet
   - Storage server crashes on second load attempt

### Expected FlashTensors Warm Swap Performance (Theoretical)

Based on the architecture, FlashTensors warm swaps **should** be:

1. **Weight loading**: ~0.5-1s (from storage pool → GPU, no disk I/O)
2. **vLLM re-initialization**: ~12-13s (same as baseline)
3. **Expected total**: ~13-14s (similar to baseline)

**Key insight**: Most of the warm swap time is **vLLM re-initialization**, not weight loading! This means:
- Even with instant weight loading, we're bottlenecked by vLLM engine setup
- To get dramatic speedup, we'd need to **keep vLLM engine alive** and just swap weights
- Current approach (full vLLM shutdown/restart) limits speedup potential

## Recommendations

### Fix Required

The storage server crash must be fixed before warm swap benchmarks can proceed. Investigation needed in:

1. `StorageServiceImpl::ConfirmModelLoaded()` in C++ server
2. CUDA IPC handle lifecycle management
3. Memory cleanup in `ClearGPU()` operation
4. Thread safety in model loading state machine

### Architecture Improvement for True Fast Swapping

To achieve 3-10x speedup claimed in documentation, consider:

1. **Keep vLLM engine alive** between swaps
2. **Hot-swap weights in-place** without full re-initialization
3. **Reuse KV cache allocations** and CUDA graphs
4. **Minimal engine re-configuration** instead of full restart

Current approach with full vLLM re-init limits speedup to ~1x (no benefit for warm swaps).

## Current Benchmark Status

- ✅ Baseline benchmark: **COMPLETE**
- ❌ FlashTensors benchmark: **BLOCKED** by storage server crash
- ❌ Comparison: **INCOMPLETE**

## Next Steps

1. **Debug storage server crash** (highest priority)
2. **Fix memory/lifecycle bug** in confirm operation
3. **Re-run FlashTensors benchmark** to get warm swap data
4. **Consider architecture changes** for true fast swapping
