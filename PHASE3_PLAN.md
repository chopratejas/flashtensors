# Phase 3: io_uring + GPU Direct Storage

**Goal:** 3-4x total speedup by optimizing disk I/O (currently 86% of time)

**Date:** 2025-11-13
**Status:** Planning → Implementation

---

## Current Bottleneck

**From benchmark:** 7B load = 4.05s
```
Disk I/O:        3.5s (86%) ← TARGET THIS!
H2D transfer:    0.48s (12%)
Sync/tensor:     0.07s (2%)
```

**Current I/O implementation** (`model.cpp:75-212`):
- Multi-threaded (16 threads)
- Synchronous `pread()` per chunk
- O_DIRECT enabled ✅
- **Problem**: Each thread blocks on I/O

---

## Phase 3.1: io_uring Async I/O

### What is io_uring?

Linux kernel interface for async I/O:
- Zero-copy between kernel and userspace
- Batch submission (queue depth 128-256)
- Kernel reorders for optimal disk access
- No thread blocking
- **2-3x faster than threaded pread()**

### Implementation Plan

**1. Add liburing dependency:**

`CMakeLists.txt`:
```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBURING REQUIRED liburing)

target_link_libraries(checkpoint_store
    ...
    ${LIBURING_LIBRARIES}
)
```

**2. Create io_uring helper class:**

`csrc/store/io_uring_reader.h`:
```cpp
class IoUringReader {
public:
    IoUringReader(int queue_depth = 256);
    ~IoUringReader();

    // Submit read operation
    int submit_read(int fd, void* buf, size_t size, off_t offset, void* user_data);

    // Wait for completions
    int wait_completions(int min_complete = 1);

    // Get completed operations
    std::vector<CompletedOp> get_completed();

private:
    struct io_uring ring_;
    int queue_depth_;
};
```

**3. Replace pread() in model.cpp:**

Current (blocking):
```cpp
for (chunk_idx ...) {
    ssize_t ret = pread(fd, buf, size, offset);  // BLOCKS!
    // ...
}
```

New (async):
```cpp
// Submit all reads
for (chunk_idx ...) {
    uring.submit_read(fd, buf, size, offset, chunk_idx);
}

// Wait for all completions
uring.wait_completions(num_chunks);

// Process results
for (auto &op : uring.get_completed()) {
    // Handle completion
}
```

**4. Error handling:**
- Check io_uring_queue_init success
- Handle partial reads
- Fall back to pread() if io_uring unavailable

---

## Phase 3.2: GPU Direct Storage (GDS)

### What is GDS?

NVIDIA GPUDirect Storage:
- Direct NVMe → GPU transfers
- Bypasses host memory and CPU
- Requires:
  - NVIDIA GPU with GPUDirect support
  - nvme driver with GDS support
  - cuFile API

### Implementation Plan

**1. Check GDS availability:**

```cpp
bool CheckGDSSupport() {
    CUfileError_t status;
    CUfileDriver_t driver_props;
    status = cuFileDriverOpen(&driver_props);
    if (status.err != CU_FILE_SUCCESS) {
        return false;
    }
    cuFileDriverClose();
    return true;
}
```

**2. Direct NVMe → GPU reads:**

```cpp
// Register GPU buffer with GDS
CUfileDescr_t cf_descr;
cuFileHandleRegister(&cf_handle, &cf_descr);

// Direct read from file to GPU
cuFileBufRegister(device_ptr, size, flags);
cuFileRead(cf_handle, device_ptr, size, offset, 0);
```

**3. Hybrid approach:**
```cpp
if (GDS_available && device_supports_gds) {
    // Direct NVMe → GPU (FASTEST!)
    use_gds_path();
} else if (io_uring_available) {
    // Async NVMe → Host → GPU (FAST)
    use_io_uring_path();
} else {
    // Fallback: pread → Host → GPU (CURRENT)
    use_traditional_path();
}
```

---

## Phase 3.3: safetensors Format Support

### Why safetensors?

- Faster loading than pickle
- Memory-mapped friendly
- No arbitrary code execution
- Growing adoption (HuggingFace, etc.)

### Implementation

**1. Add safetensors parsing:**

```cpp
// Parse safetensors header (JSON)
struct SafetensorsHeader {
    std::unordered_map<std::string, TensorMetadata> tensors;
    size_t total_size;
};

SafetensorsHeader ParseSafetensorsHeader(int fd);
```

**2. Memory-mapped loading:**

```cpp
// mmap the file
void* mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

// Zero-copy tensor creation
for (auto &[name, meta] : header.tensors) {
    torch::Tensor tensor = torch::from_blob(
        (void*)((char*)mapped + meta.offset),
        meta.shape,
        meta.dtype
    );
}
```

**3. Integration:**

```python
# snacktensors API
st.save_model(model, "my-model", format="safetensors")
model = st.load_model("my-model")  # Auto-detect format
```

---

## Expected Performance

### Component Speedups:

```
Disk I/O (current):     3.5s
  with io_uring:        1.2s  (2.9x faster)
  with GDS:             0.4s  (8.8x faster)

H2D transfer:           0.48s (Phase 1+2 optimized)
Sync/tensor:            0.07s (Phase 1+2 optimized)
```

### Total Impact:

**Current (Phase 1+2):**
```
Total: 4.05s
```

**With io_uring (Phase 3.1):**
```
Disk: 1.2s + H2D: 0.48s + Other: 0.07s = 1.75s
Speedup: 4.05s → 1.75s (2.3x)
```

**With GDS (Phase 3.2):**
```
Disk→GPU: 0.4s + Other: 0.07s = 0.47s
Speedup: 4.05s → 0.47s (8.6x!)
H2D transfer eliminated!
```

---

## Implementation Order

### Week 1: io_uring (Phase 3.1)

1. ✅ Research io_uring API
2. ⬜ Add liburing dependency
3. ⬜ Create IoUringReader class
4. ⬜ Replace pread() in ToHost()
5. ⬜ Test and benchmark
6. ⬜ Error handling and fallback

**Target:** 2.3x total speedup (4.05s → 1.75s)

### Week 2: GDS (Phase 3.2)

1. ⬜ Check GDS availability
2. ⬜ Implement cuFile integration
3. ⬜ Create hybrid path selection
4. ⬜ Test on GDS-capable hardware
5. ⬜ Benchmark and optimize

**Target:** 8.6x total speedup (4.05s → 0.47s)

### Week 3: safetensors (Phase 3.3)

1. ⬜ Add safetensors parsing
2. ⬜ Implement mmap loading
3. ⬜ Python API integration
4. ⬜ Format auto-detection
5. ⬜ Benchmark vs pickle

**Target:** Additional 10-20% improvement

---

## Technical Challenges

### io_uring:

**Challenge 1:** O_DIRECT alignment requirements
- **Solution:** Already handled in current code

**Challenge 2:** Queue depth tuning
- **Solution:** Start with 256, benchmark to find optimal

**Challenge 3:** Error handling for partial reads
- **Solution:** Resubmit failed operations

### GDS:

**Challenge 1:** Hardware compatibility
- **Solution:** Runtime detection with fallback

**Challenge 2:** cuFile setup complexity
- **Solution:** Initialize once, reuse handles

**Challenge 3:** Buffer registration overhead
- **Solution:** Register buffers at allocation time

---

## Fallback Strategy

```
┌─────────────────────────────────────┐
│ Try GDS (if available & supported)  │
│ ├─ Success: Use GDS path (FASTEST)  │
│ └─ Fail: Continue                   │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ Try io_uring (if available)         │
│ ├─ Success: Use io_uring (FAST)     │
│ └─ Fail: Continue                   │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ Use pread() (current implementation)│
│ Always works (BASELINE)              │
└─────────────────────────────────────┘
```

No regression - always have fallback!

---

## Testing Plan

### io_uring Tests:

```bash
# Test 1: Small model (100 chunks)
python3 test_io_uring_small.py

# Test 2: Large model (7B, 448 chunks)
python3 test_io_uring_7b.py

# Test 3: Stress test (multiple models)
python3 test_io_uring_stress.py
```

### GDS Tests:

```bash
# Test 1: GDS detection
python3 test_gds_detection.py

# Test 2: Direct GPU load
python3 test_gds_load.py

# Test 3: Compare GDS vs non-GDS
python3 test_gds_comparison.py
```

---

## Success Metrics

### Must Have:
- ✅ io_uring implementation working
- ✅ 2x faster disk I/O
- ✅ No regression (fallback works)
- ✅ Error handling complete

### Nice to Have:
- ✅ GDS working on compatible hardware
- ✅ 8x faster end-to-end
- ✅ safetensors format support

### Stretch Goals:
- Sub-500ms 7B model loads
- 10x total speedup from baseline
- Production deployment

---

## Dependencies

**System packages:**
```bash
# io_uring
apt-get install liburing-dev

# GDS (on compatible systems)
apt-get install nvidia-gds
```

**Python packages:**
```python
# safetensors
pip install safetensors
```

---

## Next Steps

1. **Start with Phase 3.1 (io_uring)**
   - Lowest risk
   - Highest guaranteed return (2-3x)
   - No hardware requirements

2. **Add Phase 3.2 (GDS) if available**
   - Check hardware compatibility
   - Massive gains (8x+) if supported

3. **Phase 3.3 (safetensors) as polish**
   - Additional 10-20% improvement
   - Better format for future

---

**Ready to implement Phase 3.1 (io_uring) now!** 🚀
