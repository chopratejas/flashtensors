// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------

#include "model.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#ifdef HAVE_LIBURING
#include <liburing.h>
#endif

#include <algorithm>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Third-party library headers
#include <cuda_runtime.h>
#include <glog/logging.h>

#include "error_handling.h"

// CUDA graphs removed - reverted to simpler multi-stream approach
// CUDA graphs caused cache invalidation issues when device pointers were closed

int Model::Initialize(const std::filesystem::path storage_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ != MemoryState::UNINITIALIZED) {
    return 0;
  }
  model_size_ = 0;
  partition_sizes_.clear();
  partition_paths_.clear();
  // Attempt to read from 0 until the file is not found
  for (int partition_id = 0;; ++partition_id) {
    auto tensor_path = storage_path / model_path_ /
                       ("tensor.data_" + std::to_string(partition_id));
    if (access(tensor_path.c_str(), F_OK) == -1) {
      LOG(INFO) << "Tensor file " << tensor_path << " does not exist";
      break;
    }
    struct stat st;
    if (stat(tensor_path.c_str(), &st) != 0) {
      LOG(ERROR) << "Failed to get file size of " << tensor_path;
      return -1;
    }
    model_size_ += st.st_size;
    partition_sizes_.push_back(st.st_size);
    partition_paths_.push_back(tensor_path);
  }
  if (model_size_ == 0) {
    LOG(ERROR) << "Model " << model_path_ << " does not exist";
    return -1;
  }
  state_ = MemoryState::UNALLOCATED;

  return 0;
}

int Model::ToHost(int num_threads) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (state_ != MemoryState::ALLOCATED) {
    if (state_ == MemoryState::LOADING || state_ == MemoryState::LOADED) {
      return 0;
    } else {
      LOG(ERROR) << "Model " << model_path_ << " is at state " << state_;
      return -1;
    }
  }

  std::vector<int> file_descriptors;
  // Attempt to read from 0 until the file is not found
  for (int partition_id = 0; partition_id < partition_sizes_.size();
       ++partition_id) {
    auto tensor_path = partition_paths_[partition_id];
    if (access(tensor_path.c_str(), F_OK) == -1) {
      LOG(ERROR) << "File " << tensor_path << " does not exist";
      return -1;
    }

    // Open file
    int fd = open(tensor_path.c_str(), O_DIRECT | O_RDONLY);
    if (fd < 0) {
      std::string err = "open() failed for file: " + tensor_path.string() +
                        ", error: " + strerror(errno);
      LOG(ERROR) << err;
      return -1;
    }

    // OPTIMIZATION: Hint kernel for sequential reads and read-ahead
    posix_fadvise(fd, 0, partition_sizes_[partition_id],
                  POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);

    file_descriptors.push_back(fd);
  }

  LOG(INFO) << "Loading model " << model_path_ << " size " << model_size_
            << " to host";
  if (!pinned_mem_ || pinned_mem_->num_chunks() == 0) {
    LOG(ERROR) << "CPU memory not allocated";
    return 1;
  }

  auto host_buffers = pinned_mem_->get();
  size_t num_chunks = pinned_mem_->num_chunks();
  size_t chunk_size = pinned_mem_->chunk_size();
  host_ptr_vector_ = std::make_shared<BatchVector>();
  host_ptr_vector_->init("queue_name", num_chunks);

  state_ = MemoryState::LOADING;
  lock.unlock();

#ifdef HAVE_LIBURING
  // OPTIMIZATION: io_uring async disk I/O for 2-3x faster reads
  struct io_uring ring;
  // OPTIMIZATION: Use larger ring for big models (>1000 chunks ≈ 2GB+)
  int max_ring_size = (num_chunks > 1000) ? 8192 : 4096;
  int ring_size = std::min(num_chunks, (size_t)max_ring_size);
  int ret = io_uring_queue_init(ring_size, &ring, 0);
  if (ret < 0) {
    LOG(ERROR) << "io_uring_queue_init failed: " << strerror(-ret);
    lock.lock();
    state_ = MemoryState::INTERRUPTED;
    pinned_mem_.reset();
    state_ = MemoryState::UNALLOCATED;
    return -1;
  }

  LOG(INFO) << "Loading model " << model_path_ << " to host with io_uring, "
            << num_chunks << " chunks, " << chunk_size << " chunk size";

  // Prepare all read requests
  struct ReadRequest {
    size_t chunk_idx;
    size_t size;
    int fd;
    size_t file_offset;
    size_t partition_id;
    size_t buffer_offset;  // Offset within the chunk buffer for split reads
  };
  std::vector<ReadRequest> read_requests;

  // Calculate cumulative partition sizes for efficient boundary detection
  std::vector<size_t> cumulative_partition_sizes;
  size_t cumulative = 0;
  for (size_t partition_size : partition_sizes_) {
    cumulative += partition_size;
    cumulative_partition_sizes.push_back(cumulative);
  }
  LOG(INFO) << "Model has " << partition_sizes_.size() << " partitions, cumulative sizes: ";
  for (size_t i = 0; i < cumulative_partition_sizes.size(); ++i) {
    LOG(INFO) << "  Partition " << i << ": " << partition_sizes_[i] / MB << "MB (cumulative: " << cumulative_partition_sizes[i] / MB << "MB)";
  }

  size_t global_offset = 0;
  for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    size_t size = std::min(chunk_size, model_size_ - chunk_idx * chunk_size);

    if (host_buffers[chunk_idx] == nullptr) {
      LOG(ERROR) << "Host buffer not allocated";
#ifdef HAVE_LIBURING
      io_uring_queue_exit(&ring);
#endif
      lock.lock();
      state_ = MemoryState::INTERRUPTED;
      pinned_mem_.reset();
      state_ = MemoryState::UNALLOCATED;
      return -1;
    }

    // Find which partition this chunk belongs to using cumulative sizes
    // Handle chunks that may span partition boundaries by splitting them
    size_t remaining_size = size;
    size_t buffer_offset = 0;
    
    while (remaining_size > 0) {
      // Find the partition containing the current global offset
      size_t partition_id = 0;
      size_t partition_offset = global_offset;
      bool found_partition = false;
      
      for (size_t i = 0; i < cumulative_partition_sizes.size(); ++i) {
        if (global_offset < cumulative_partition_sizes[i]) {
          partition_id = i;
          // Calculate offset within this partition
          if (i > 0) {
            partition_offset = global_offset - cumulative_partition_sizes[i - 1];
          }
          found_partition = true;
          break;
        }
      }
      
      // Safety check: if we're beyond all partitions, something is wrong
      if (!found_partition) {
        LOG(ERROR) << "Chunk " << chunk_idx << " global_offset " << global_offset / MB 
                   << "MB exceeds model size " << model_size_ / MB << "MB";
        lock.lock();
        state_ = MemoryState::INTERRUPTED;
        pinned_mem_.reset();
        state_ = MemoryState::UNALLOCATED;
        io_uring_queue_exit(&ring);
        return -1;
      }
      
      // Bounds check: ensure partition_id is valid
      if (partition_id >= partition_sizes_.size()) {
        LOG(ERROR) << "Invalid partition_id " << partition_id << " (max: " 
                   << (partition_sizes_.size() - 1) << ") for chunk " << chunk_idx 
                   << " at global_offset " << global_offset / MB << "MB";
        lock.lock();
        state_ = MemoryState::INTERRUPTED;
        pinned_mem_.reset();
        state_ = MemoryState::UNALLOCATED;
        io_uring_queue_exit(&ring);
        return -1;
      }
      
      // Calculate how much we can read from this partition
      size_t remaining_in_partition = partition_sizes_[partition_id] - partition_offset;
      size_t read_size = std::min(remaining_size, remaining_in_partition);
      
      if (chunk_idx < 10 || chunk_idx % 1000 == 0 || remaining_size != size) {
        LOG(INFO) << "Chunk " << chunk_idx << " (buffer_offset=" << buffer_offset / MB << "MB): "
                  << "global_offset=" << global_offset / MB << "MB, "
                  << "partition_id=" << partition_id << ", "
                  << "partition_offset=" << partition_offset / MB << "MB, "
                  << "read_size=" << read_size / MB << "MB, "
                  << "remaining=" << remaining_size / MB << "MB";
      }
      
      if (remaining_size != size) {
        LOG(WARNING) << "Chunk " << chunk_idx << " spans partition boundary - splitting read: "
                     << "first_part=" << (size - remaining_size) / MB << "MB, "
                     << "second_part=" << read_size / MB << "MB from partition " << partition_id;
      }
      
      read_requests.push_back({
          chunk_idx, read_size, file_descriptors[partition_id], partition_offset, partition_id, buffer_offset});
      
      global_offset += read_size;
      remaining_size -= read_size;
      buffer_offset += read_size;
    }
  }

  LOG(INFO) << "Prepared " << read_requests.size() << " io_uring read requests";

  // Submit all read requests in batches
  size_t submitted = 0;
  bool error = false;

  while (submitted < read_requests.size() && !error) {
    // Submit a batch of requests
    size_t batch_size = std::min((size_t)ring_size, read_requests.size() - submitted);

    for (size_t i = 0; i < batch_size; ++i) {
      auto &req = read_requests[submitted + i];
      struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
      if (!sqe) {
        LOG(ERROR) << "io_uring_get_sqe failed";
        error = true;
        break;
      }

      // Calculate the correct buffer pointer, accounting for buffer_offset for split reads
      void* read_buffer = host_buffers[req.chunk_idx];
      if (req.buffer_offset > 0) {
        read_buffer = static_cast<char*>(read_buffer) + req.buffer_offset;
      }
      io_uring_prep_read(sqe, req.fd, read_buffer,
                         req.size, req.file_offset);
      io_uring_sqe_set_data(sqe, (void *)(uintptr_t)req.chunk_idx);
    }

    if (error) break;

    // Submit the batch
    int submit_ret = io_uring_submit(&ring);
    if (submit_ret < 0) {
      LOG(ERROR) << "io_uring_submit failed: " << strerror(-submit_ret);
      error = true;
      break;
    }

    // Wait for completions for this batch
    for (size_t i = 0; i < batch_size; ++i) {
      struct io_uring_cqe *cqe;
      ret = io_uring_wait_cqe(&ring, &cqe);
      if (ret < 0) {
        LOG(ERROR) << "io_uring_wait_cqe failed: " << strerror(-ret);
        error = true;
        break;
      }

      size_t chunk_idx = (size_t)io_uring_cqe_get_data(cqe);
      int res = cqe->res;
      io_uring_cqe_seen(&ring, cqe);

      if (res < 0) {
        LOG(ERROR) << "io_uring read failed for chunk " << chunk_idx
                   << ": " << strerror(-res);
        error = true;
        break;
      }

      // Look up the request by chunk_idx (io_uring can complete out of order)
      auto &req = read_requests[chunk_idx];
      if ((size_t)res != req.size) {
        LOG(ERROR) << "io_uring read size mismatch for chunk " << chunk_idx
                   << ": expected " << req.size << ", got " << res;
        error = true;
        break;
      }

      // Enqueue completed chunk
      host_ptr_vector_->enqueue(chunk_idx, Batch{chunk_idx, (size_t)res});
    }

    submitted += batch_size;
  }

  io_uring_queue_exit(&ring);
  LOG(INFO) << "Completed " << submitted << " io_uring read operations";

#else
  // FALLBACK: Multi-threaded pread() implementation (original code)
  std::vector<std::future<int>> futures;
  size_t chunk_per_thread = (num_chunks + num_threads - 1) / num_threads;
  LOG(INFO) << "Loading model " << model_path_ << " to host with "
            << num_threads << " threads, " << num_chunks << " chunks, "
            << chunk_size << " chunk size, " << chunk_per_thread
            << " chunks per thread";

  // Calculate cumulative partition sizes for efficient boundary detection
  std::vector<size_t> cumulative_partition_sizes;
  size_t cumulative = 0;
  for (size_t partition_size : partition_sizes_) {
    cumulative += partition_size;
    cumulative_partition_sizes.push_back(cumulative);
  }

  for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    futures.emplace_back(std::async(std::launch::async, [&, thread_idx, cumulative_partition_sizes]() {
      size_t start_chunk = thread_idx * chunk_per_thread;
      size_t end_chunk = std::min((thread_idx + 1) * chunk_per_thread, num_chunks);
      
      if (start_chunk >= num_chunks) {
        LOG(INFO) << "Thread " << thread_idx << " early exits (no chunks)";
        return 0;
      }

      size_t global_offset = start_chunk * chunk_size;
      
      // Find starting partition
      size_t partition_id = 0;
      size_t partition_offset = global_offset;
      bool found_partition = false;
      for (size_t i = 0; i < cumulative_partition_sizes.size(); ++i) {
        if (global_offset < cumulative_partition_sizes[i]) {
          partition_id = i;
          if (i > 0) {
            partition_offset = global_offset - cumulative_partition_sizes[i - 1];
          }
          found_partition = true;
          break;
        }
      }
      
      // Safety check: ensure we found a valid partition
      if (!found_partition || partition_id >= partition_sizes_.size()) {
        LOG(ERROR) << "Thread " << thread_idx << " invalid partition " << partition_id 
                   << " for start_chunk " << start_chunk << " at global_offset " 
                   << global_offset / MB << "MB";
        return -1;
      }

      LOG(INFO) << "Thread " << thread_idx << " starting from chunk " << start_chunk
                << " partition " << partition_id << " offset " << partition_offset / MB << "MB";

      for (size_t chunk_idx = start_chunk; chunk_idx < end_chunk; ++chunk_idx) {
        size_t size = std::min(chunk_size, model_size_ - chunk_idx * chunk_size);
        
        if (host_buffers[chunk_idx] == nullptr) {
          LOG(ERROR) << "Host buffer not allocated";
          return -1;
        }

        if (state_ == MemoryState::CANCELLED) {
          LOG(INFO) << "Loading from disk for model " << model_path_ << " is cancelled";
          return 0;
        }

        // Handle partition boundaries with improved logic
        size_t remaining_size = size;
        size_t buffer_offset = 0;
        
        while (remaining_size > 0) {
          // Find current partition
          size_t current_partition_id = partition_id;
          size_t current_partition_offset = partition_offset;
          
          // Bounds check: ensure partition_id is valid
          if (current_partition_id >= partition_sizes_.size()) {
            LOG(ERROR) << "Thread " << thread_idx << " invalid partition_id " << current_partition_id 
                       << " (max: " << (partition_sizes_.size() - 1) << ") for chunk " << chunk_idx 
                       << " at global_offset " << global_offset / MB << "MB";
            return -1;
          }
          
          // Check if we need to move to next partition
          size_t remaining_in_partition = partition_sizes_[current_partition_id] - current_partition_offset;
          size_t read_size = std::min(remaining_size, remaining_in_partition);
          
          int fd = file_descriptors[current_partition_id];
          void* read_buffer = static_cast<char*>(host_buffers[chunk_idx]) + buffer_offset;
          
          ssize_t ret = pread(fd, read_buffer, read_size, current_partition_offset);
          
          if (ret < 0) {
            auto tensor_path = partition_paths_[current_partition_id];
            LOG(ERROR) << "pread() failed for file: " << tensor_path
                       << ", error: " << strerror(errno);
            return -1;
          } else if ((size_t)ret != read_size) {
            auto tensor_path = partition_paths_[current_partition_id];
            LOG(ERROR) << "Failed to read file: " << tensor_path
                       << " read: " << ret << " expected: " << read_size;
            return -1;
          }

          remaining_size -= read_size;
          buffer_offset += read_size;
          
          // Update offsets for next iteration
          partition_offset += read_size;
          if (partition_offset >= partition_sizes_[current_partition_id] && 
              current_partition_id + 1 < partition_sizes_.size()) {
            partition_id = current_partition_id + 1;
            partition_offset = 0;
          }
        }

        host_ptr_vector_->enqueue(chunk_idx, Batch{chunk_idx, size});
      }

      return 0;
    }));
  }

  bool error = false;
  for (auto &future : futures) {
    int ret = future.get();
    if (ret != 0) {
      LOG(ERROR) << "Error reading from disk, ret " << ret;
      error = true;
    }
  }
#endif

  // close file
  for (int fd : file_descriptors) {
    close(fd);
  }

  lock.lock();
  if (error) {
    state_ = MemoryState::INTERRUPTED;
    // Deal with gpu replicas
    for (auto &[replica_uuid, gpu_replica] : gpu_replicas_) {
      if (gpu_replica->state_ == MemoryState::LOADING) {
        gpu_replica->state_ = MemoryState::CANCELLED;
        gpu_replica->cv_.notify_all();
      }
      // wait for gpu replicas to finish
      gpu_replica->cv_.wait(lock, [&gpu_replica] {
        return gpu_replica->state_ == MemoryState::LOADED ||
               gpu_replica->state_ == MemoryState::INTERRUPTED;
      });
      // Note: gpu replicas will be handled by the caller
    }
    // release pinned memory
    pinned_mem_.reset();
    state_ = MemoryState::UNALLOCATED;

    return -1;
  }

  state_ = MemoryState::LOADED;
  cv_.notify_all();  // Wake up waiting threads (e.g., ToGpu)
  LOG(INFO) << "Finished loading model " << model_path_ << " from disk";

  return 0;
}

int Model::ToGpu(
    const std::string &replica_uuid, const MemPtrListMap &device_ptrs,
    const std::unordered_map<int, MemCopyChunkList> &mem_copy_chunks,
    const std::unordered_map<int, MemCopyHandleList> &mem_copy_handles) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (state_ == MemoryState::UNINITIALIZED) {
    LOG(ERROR) << "Model " << model_path_ << " is not initialized";
    return -1;
  }

  if (gpu_replicas_.find(replica_uuid) != gpu_replicas_.end()) {
    LOG(ERROR) << "Replica " << replica_uuid << " already exists";
    return -1;
  }
  LOG(INFO) << "Creating replica " << replica_uuid;
  gpu_replicas_.emplace(replica_uuid, std::make_shared<GpuReplica>());
  GpuReplicaPtr gpu_replica = gpu_replicas_.at(replica_uuid);
  for (const auto &[device_id, _] : device_ptrs) {
    LOG(INFO) << "Creating queue for device " << device_id;
    gpu_replica->gpu_loading_queue_.emplace(device_id,
                                            std::make_shared<BatchQueue>());
  }
  gpu_replica->device_ptrs_ = device_ptrs;
  gpu_replica->state_ = MemoryState::LOADING;
  LOG(INFO) << "Created replica " << replica_uuid;
  cv_.notify_all();
  lock.unlock();

  // Start a dispatcher first
  auto dispatch_future =
      std::async(std::launch::async, [this, gpu_replica, mem_copy_chunks,
                                      mem_copy_handles]() {
        return DispatchToGpu(gpu_replica, mem_copy_chunks, mem_copy_handles);
      });

  LOG(INFO) << "Dispatcher started for model " << model_path_;

  std::unordered_map<int, std::future<int>> futures;
  for (auto &[device_id, device_ptr_list] : device_ptrs) {
    futures.emplace(
        device_id, std::async(std::launch::async, [this, gpu_replica, device_id,
                                                   device_ptr_list]() {
          auto gpu_loading_queue =
              gpu_replica->gpu_loading_queue_.at(device_id);
          if (!pinned_mem_ || pinned_mem_->num_chunks() == 0) {
            LOG(ERROR) << "CPU memory not allocated";
            return 1;
          }

          cudaError_t err = cudaSetDevice(device_id);
          if (err != cudaSuccess) {
            LOG(ERROR) << "Error setting device " << cudaGetErrorString(err);
            return 1;
          }

          auto &host_buffers = pinned_mem_->get();

          // Collect all transfer operations first
          struct TransferOp {
            size_t chunk_id;
            size_t chunk_offset;
            size_t size;
            size_t gpu_offset;
            int handle_idx;
          };
          std::vector<TransferOp> transfers;

          size_t loaded_size = 0;
          while (true) {
            auto [chunk_id, chunk_offset, size, gpu_offset, handle_idx] =
                gpu_loading_queue->dequeue();
            if (size == 0) {
              break;
            }
            if (gpu_replica->state_ == MemoryState::CANCELLED) {
              LOG(INFO) << "Loading from mem for model " << model_path_
                        << " is cancelled, chunk " << chunk_id;
              // Clean up streams before returning (will be created below)
              return 0;
            }

            transfers.push_back({chunk_id, chunk_offset, size, gpu_offset, handle_idx});
            loaded_size += size;
          }

          LOG(INFO) << "Starting " << transfers.size()
                    << " async transfers across 4 streams for device " << device_id;

          // OPTIMIZATION: Multi-stream async transfers for 2-3x speedup
          const int num_streams = 4;  // 4 concurrent streams for overlap
          std::vector<cudaStream_t> streams(num_streams);
          for (int i = 0; i < num_streams; ++i) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]),
                      "cudaStreamCreate Error");
          }

          // Issue all async transfers across multiple streams
          for (size_t i = 0; i < transfers.size(); ++i) {
            int stream_idx = i % num_streams;  // Round-robin across streams
            auto& op = transfers[i];

            CUDA_CHECK(
                cudaMemcpyAsync(
                    (void *)((char *)device_ptr_list[op.handle_idx] + op.gpu_offset),
                    (void *)(host_buffers[op.chunk_id] + op.chunk_offset),
                    op.size,
                    cudaMemcpyHostToDevice,
                    streams[stream_idx]),
                "cudaMemcpyAsync Error");
          }

          // CRITICAL: Synchronize ALL streams before proceeding
          LOG(INFO) << "Synchronizing " << num_streams << " CUDA streams...";
          for (int i = 0; i < num_streams; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]),
                      "cudaStreamSynchronize Error");
          }

          // Clean up streams
          for (auto& stream : streams) {
            cudaStreamDestroy(stream);
          }

          LOG(INFO) << "Finished loading " << loaded_size
                    << " bytes to device " << device_id
                    << " using multi-stream transfers";

          return 0;
        }));
  }

  LOG(INFO) << "Waiting for model " << model_path_ << " num tasks "
            << futures.size() << " state " << gpu_replica->state_;
  dispatch_future.wait();
  bool error = false;
  for (auto &[device_id, future] : futures) {
    int ret = future.get();
    if (ret != 0) {
      LOG(ERROR) << "Error copying to device " << device_id;
      error = true;
    }
  }

  lock.lock();
  futures.clear();

  if (error) {
    LOG(ERROR) << "Failed to load model " << model_path_;
    gpu_replica->state_ = MemoryState::INTERRUPTED;
  } else {
    gpu_replica->state_ = MemoryState::LOADED;
  }
  gpu_replica->cv_.notify_all();

  // TODO: move to background thread
  for (auto &[device_id, device_ptr_list] : gpu_replica->device_ptrs_) {
    cudaSetDevice(device_id);
    for (auto device_ptr : device_ptr_list) {
      cudaError_t err = cudaIpcCloseMemHandle(device_ptr);
      if (err != cudaSuccess) {
        LOG(ERROR) << "Failed to close memory handle for device " << device_id
                   << " error: " << cudaGetErrorString(err);
      }
    }
  }

  if (gpu_replica->state_ == MemoryState::INTERRUPTED) {
    LOG(ERROR) << "Model " << model_path_ << " replica " << replica_uuid
               << " is interrupted";
    return -1;
  }

  return 0;
}

int Model::WaitInHost() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (state_ < MemoryState::LOADED) {
    cv_.wait(lock, [this] {
      return state_ == MemoryState::LOADED ||
             state_ == MemoryState::INTERRUPTED;
    });
  }

  if (state_ >= MemoryState::INTERRUPTED) {
    LOG(INFO) << "Model " << model_path_ << " is interrupted";
    return 1;
  }

  return 0;
}

int Model::WaitInGpu(const std::string &replica_uuid) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (gpu_replicas_.find(replica_uuid) == gpu_replicas_.end()) {
    cv_.wait(lock, [this, replica_uuid] {
      return gpu_replicas_.find(replica_uuid) != gpu_replicas_.end();
    });
  }

  auto &gpu_replica = gpu_replicas_.at(replica_uuid);

  if (gpu_replica->state_ < MemoryState::LOADED) {
    gpu_replica->cv_.wait(lock, [&gpu_replica] {
      return gpu_replica->state_ == MemoryState::LOADED ||
             gpu_replica->state_ == MemoryState::INTERRUPTED;
    });
  }

  if (gpu_replica->state_ >= MemoryState::INTERRUPTED) {
    LOG(INFO) << "Model " << model_path_ << " is interrupted";
    return 1;
  }

  return 0;
}

int Model::FreeGpu(const std::string &replica_uuid) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (gpu_replicas_.find(replica_uuid) == gpu_replicas_.end()) {
    LOG(ERROR) << "Model " << model_path_ << " replica " << replica_uuid
               << " is not registered";
    return -1;
  }
  
  auto &gpu_replica = gpu_replicas_.at(replica_uuid);
  if (gpu_replica->state_ == MemoryState::UNINITIALIZED) {
    LOG(WARNING) << "Model " << model_path_ << " replica " << replica_uuid
                 << " is not initialized";
    gpu_replicas_.erase(replica_uuid);
    return 0;
  }

  if (gpu_replica->state_ == MemoryState::LOADING) {
    LOG(INFO) << "Waiting for model " << model_path_ << " replica "
              << replica_uuid << " to be loaded";
    gpu_replica->cv_.wait(lock, [&gpu_replica] {
      return gpu_replica->state_ == MemoryState::LOADED ||
             gpu_replica->state_ == MemoryState::INTERRUPTED;
    });
  }

  gpu_replicas_.erase(replica_uuid);
  return 0;
}

int Model::FreeHost() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (state_ == MemoryState::UNINITIALIZED) {
    LOG(WARNING) << "Model " << model_path_ << " is not initialized";
    return 1;
  }

  if (state_ == MemoryState::UNALLOCATED) {
    LOG(WARNING) << "Model " << model_path_ << " is not allocated";
    return 1;
  }

  if (state_ == MemoryState::LOADING) {
    LOG(INFO) << "Waiting for model " << model_path_ << " to be loaded";
    cv_.wait(lock, [this] {
      return state_ == MemoryState::LOADED ||
             state_ == MemoryState::INTERRUPTED;
    });
  }

  // make sure no gpu replicas are loading
  for (auto &[replica_uuid, gpu_replica] : gpu_replicas_) {
    if (gpu_replica->state_ == MemoryState::LOADING) {
      LOG(INFO) << "Waiting for replica " << replica_uuid << " to be loaded";
      gpu_replica->cv_.wait(lock, [&gpu_replica] {
        return gpu_replica->state_ == MemoryState::LOADED ||
               gpu_replica->state_ == MemoryState::INTERRUPTED;
      });
    }
  }

  // free pinned memory
  int freed_chunks = pinned_mem_->num_chunks();
  pinned_mem_.reset();
  state_ = MemoryState::UNALLOCATED;

  return 0;
}

int Model::TryFreeHost() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (state_ == MemoryState::UNINITIALIZED) {
    LOG(WARNING) << "Model " << model_path_ << " is not initialized";
    return 0;
  }

  if (state_ == MemoryState::UNALLOCATED) {
    LOG(WARNING) << "Model " << model_path_ << " is not allocated";
    return 0;
  }

  if (state_ == MemoryState::LOADING) {
    return -1;
  }

  // make sure no gpu replicas are loading
  for (auto &[replica_uuid, gpu_replica] : gpu_replicas_) {
    if (gpu_replica->state_ == MemoryState::LOADING) {
      return -1;
    }
  }

  // free pinned memory
  int freed_chunks = pinned_mem_->num_chunks();
  pinned_mem_.reset();
  state_ = MemoryState::UNALLOCATED;

  return freed_chunks;
}

int Model::DispatchToGpu(
    const std::shared_ptr<GpuReplica> &gpu_replica,
    const std::unordered_map<int, MemCopyChunkList> &mem_copy_chunks,
    const std::unordered_map<int, MemCopyHandleList> &mem_copy_handles) {
  // device_id, chunk_offset, size, gpu_offset

  size_t num_chunks = pinned_mem_->num_chunks();
  std::vector<std::vector<GpuChunk>> chunk_id_to_gpu_chunks(num_chunks);
  for (const auto &[device_id, mem_copy_chunk_list] : mem_copy_chunks) {
    const auto &device_handles = mem_copy_handles.at(device_id);
    std::vector<size_t> handle_offsets(device_handles.size(), 0);

    for (auto [host_offset, size, gpu_offset, handle_idx] :
         mem_copy_chunk_list) {
      handle_offsets[handle_idx] = gpu_offset;

      std::vector<std::tuple<int, size_t, size_t>> chunks =
          MapDataToChunks(host_offset, size, pinned_mem_->chunk_size());
      for (const auto &[chunk_id, chunk_offset, size] : chunks) {
        chunk_id_to_gpu_chunks[chunk_id].push_back(
            std::make_tuple(device_id, chunk_offset, size,
                            handle_offsets[handle_idx], handle_idx));
        handle_offsets[handle_idx] += size;
      }
    }
  }

  for (int i = 0; i < host_ptr_vector_->capacity(); i++) {
    auto data_chunk = host_ptr_vector_->dequeue(i);
    auto chunk_id = data_chunk.chunk_id_;
    auto &gpu_chunks = chunk_id_to_gpu_chunks[chunk_id];
    for (const auto &[device_id, chunk_offset, size, gpu_offset, handle_idx] :
         gpu_chunks) {
      auto &gpu_loading_queue = gpu_replica->gpu_loading_queue_.at(device_id);
      // LOG(INFO) << "Enqueueing chunk " << chunk_id << " offset " <<
      // chunk_offset
      //           << " size " << size << " to device " << device_id;
      gpu_loading_queue->enqueue(
          GpuBatch{chunk_id, chunk_offset, size, gpu_offset, handle_idx});
    }
  }

  // notify end of loading
  for (auto &[device_id, gpu_loading_queue] : gpu_replica->gpu_loading_queue_) {
    gpu_loading_queue->enqueue(GpuBatch{});
  }

  return 0;
}

std::vector<std::tuple<int, size_t, size_t>>
Model::MapDataToChunks(size_t offset, size_t size, size_t chunk_size) {
  int start_chunk = offset / chunk_size;
  size_t offset_in_start_chunk = offset % chunk_size;
  size_t remaining_data = size;
  std::vector<std::tuple<int, size_t, size_t>> output;

  for (int chunk_id = start_chunk; remaining_data > 0; ++chunk_id) {
    const size_t chunk_data_size =
        (chunk_id == start_chunk)
            ? std::min(chunk_size - offset_in_start_chunk, remaining_data)
            : std::min(chunk_size, remaining_data);
    output.emplace_back(chunk_id,
                        chunk_id == start_chunk ? offset_in_start_chunk : 0,
                        chunk_data_size);
    remaining_data -= chunk_data_size;
  }

  return output;
}

int Model::AllocatePinnedMemory(std::shared_ptr<PinnedMemoryPool> pool) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ == MemoryState::UNINITIALIZED) {
    LOG(ERROR) << "Model " << model_path_ << " is not initialized";
    return -1;
  }
  if (state_ != MemoryState::UNALLOCATED) {
    return 0;
  }
  pinned_mem_ = std::make_shared<PinnedMemory>();
  int ret = pinned_mem_->Allocate(model_size_, pool);
  if (ret < 0) {
    LOG(ERROR) << "Error allocating CPU memory for model " << model_path_;
    return ret;
  } else if (ret > 0) {
    LOG(WARNING) << "Not enough memory for model " << model_path_;
    return ret;
  } else if (!pinned_mem_ || pinned_mem_->num_chunks() == 0) {
    LOG(ERROR) << "CPU memory not allocated";
    return -1;
  }

  state_ = MemoryState::ALLOCATED;
  return 0;
};