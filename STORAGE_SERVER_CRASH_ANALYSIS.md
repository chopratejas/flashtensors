# Storage Server Crash Bug Analysis

## Problem Statement

The FlashTensors storage server crashes during warm swap scenarios (second model load after cleanup). This completely blocks the core use case of FlashTensors.

### Error Pattern

```
Iteration 1: load_into_gpu() → confirm_model_loaded() ✅ SUCCESS (14.891s)
Cleanup: cleanup_gpu() ✅ SUCCESS
Iteration 2: load_into_gpu() ✅ SUCCESS → confirm_model_loaded() ❌ CRASH

Error: StatusCode.UNAVAILABLE - "recvmsg:Connection reset by peer"
```

The server **dies completely** during the `confirm_model_loaded()` RPC call on iteration 2.

## Root Cause Analysis

### Initial Hypothesis (WRONG)

I initially thought the bug was in `ClearMem()` clearing `model_map_` which destroyed Model objects while GPU replicas were still active.

**Fix attempted**: Removed `model_map_.clear()` to keep models registered.
**Result**: ❌ Still crashes!

### Current Hypothesis (INVESTIGATING)

The real issue appears to be:

1. **Each iteration creates a NEW GPU replica with a different UUID**:
   - Iteration 1: `replica_uuid = "2cb5dff4-92ca-4c24-b97b-bd30dd2d779c"`
   - Iteration 2: `replica_uuid = "9fa0fae4-cef6-489b-9eb3-9cc1261ff525"`

2. **GPU replicas accumulate in the Model object** (never cleaned up)

3. **Possible conflicts**:
   - Multiple replicas trying to use the same CUDA IPC handles
   - Race condition in async `ToGpu()` task
   - State machine issue in `WaitInGpu()` when multiple replicas exist

### Code Flow Analysis

#### Iteration 1 (SUCCESS)
```
1. RegisterModel() → Model object created, stored in model_map_
2. LoadModelFromMemAsync() → Async task starts loading to GPU
3. WaitInGpu() → Waits for async task, returns 0 (success)
4. cleanup_gpu() → Calls ClearMem() which calls FreeHost()
   - GPU replica #1 remains in gpu_replicas_ map (NOT freed)
```

#### Iteration 2 (CRASH)
```
1. Model already registered (reused from iteration 1)
2. LoadModelFromMemAsync() → Creates GPU replica #2 with NEW UUID
3. WaitInGpu() → **CRASHES** during confirm
   - Now has TWO GPU replicas (#1 and #2) in gpu_replicas_ map
   - Something goes wrong in the async loading or confirm operation
```

## Fixes Attempted

### Fix #1: Don't Clear model_map_
```cpp
// BEFORE (WRONG):
model_map_.clear();  // Destroys Model objects!

// AFTER:
// model_map_.clear();  // Keep models registered
```
**Result**: ❌ Still crashes

### Fix #2: Explicitly Free GPU Replicas
```cpp
for (const auto &uuid : replica_uuids) {
  model->FreeGpu(uuid);  // Erase from gpu_replicas_ map
}
```
**Result**: ❌ Still crashes (may have made it worse!)

### Fix #3: Don't Touch GPU Replicas at All
```cpp
// Just call FreeHost(), let client handle IPC cleanup
model->FreeHost();
// GPU replicas persist in gpu_replicas_ map
```
**Result**: ❌ Still crashes (tested, not yet confirmed)

## The Real Issue

Looking at `Model::FreeGpu()`:
```cpp
int Model::FreeGpu(const std::string &replica_uuid) {
  std::unique_lock<std::mutex> lock(mutex_);
  // ... checks ...
  gpu_replicas_.erase(replica_uuid);  // Just erases from map!
  return 0;
}
```

**It doesn't actually FREE anything!** It just erases the replica from the map. The CUDA IPC handles and GPU memory are never explicitly freed.

## Potential Root Causes

### 1. CUDA IPC Handle Leakage
- `cudaIpcOpenMemHandle()` is called in `GetDevicePtrsFromMemHandles()` (checkpoint_store.cpp:372)
- There's NO corresponding `cudaIpcCloseMemHandle()` anywhere!
- Multiple replicas may be opening the SAME IPC handles, causing conflicts

### 2. Async Task State Corruption
- `LoadModelFromMemAsync()` creates an async task that may still be running
- If cleanup happens while async task is active, could cause use-after-free
- The crash in `WaitInGpu()` suggests the async task is in a bad state

### 3. GpuReplica State Machine Issue
- Each GpuReplica has a state (UNINITIALIZED, LOADING, LOADED, INTERRUPTED)
- With multiple replicas, the state machine may get confused
- `WaitInGpu()` waits on condition variables that may deadlock or crash

## Recommended Fix Strategy

### Short-term: Add CUDA IPC Handle Cleanup

```cpp
// In Model::FreeGpu() or GpuReplica destructor:
for (auto &ptr : gpu_memory_ptrs) {
  cudaIpcCloseMemHandle(ptr);
}
```

### Medium-term: Prevent Multiple GPU Replicas Per Model

The current design allows unlimited GPU replicas per model, but for the warm swap use case, we should:

1. **Reuse existing GPU replica** if one exists
2. **Free old replica** before creating new one
3. **Only allow ONE replica per model** for simplicity

### Long-term: Redesign Warm Swap Architecture

Current architecture does full vLLM teardown between swaps, which is slow. Better approach:

1. Keep vLLM engine alive between swaps
2. Hot-swap weights in-place
3. Reuse KV cache and CUDA graphs
4. Would achieve true 3-10x speedup claim

## Files Modified

- `/root/flashtensors/csrc/store/checkpoint_store.cpp` - ClearMem() fixes
- `/root/flashtensors/csrc/store/model.h` - Added GetGpuReplicas() getter

## Testing Status

- ✅ Baseline benchmark: COMPLETE (13.680s warm swap)
- ❌ FlashTensors benchmark: **BLOCKED by crash**
- ✅ First load works: 14.897s (31% faster than baseline!)
- ❌ Second load crashes server

## Next Steps

1. Add explicit `cudaIpcCloseMemHandle()` calls
2. Implement replica reuse or cleanup strategy
3. Add detailed logging to async loading tasks
4. Consider disabling multiple replicas per model
5. Run under valgrind/gdb to catch exact crash location

## Impact

**CRITICAL**: This bug completely blocks the core value proposition of FlashTensors (fast warm swapping). Without this fix, FlashTensors can only do single loads, not repeated swaps.
