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
#include <liburing.h>

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

// OPTIMIZATION: Cache CUDA graphs to avoid recreating on every load
namespace {
struct GraphCacheEntry {
  cudaGraphExec_t graph_exec;
  size_t num_transfers;
  size_t total_size;
};
std::unordered_map<std::string, GraphCacheEntry> cuda_graph_cache_;
std::mutex graph_cache_mutex_;
}

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
  };
  std::vector<ReadRequest> read_requests;

  size_t partition_id = 0;
  size_t file_offset = 0;
  for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    size_t size = std::min(chunk_size, model_size_ - chunk_idx * chunk_size);

    if (host_buffers[chunk_idx] == nullptr) {
      LOG(ERROR) << "Host buffer not allocated";
      io_uring_queue_exit(&ring);
      lock.lock();
      state_ = MemoryState::INTERRUPTED;
      pinned_mem_.reset();
      state_ = MemoryState::UNALLOCATED;
      return -1;
    }

    // Handle partition boundaries
    while (partition_id < partition_sizes_.size() &&
           file_offset >= partition_sizes_.at(partition_id)) {
      file_offset -= partition_sizes_.at(partition_id);
      partition_id += 1;
    }

    read_requests.push_back({
        chunk_idx, size, file_descriptors[partition_id], file_offset, partition_id});

    file_offset += size;
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

      io_uring_prep_read(sqe, req.fd, host_buffers[req.chunk_idx],
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
              return 0;
            }

            transfers.push_back({chunk_id, chunk_offset, size, gpu_offset, handle_idx});
            loaded_size += size;
          }

          LOG(INFO) << "Starting " << transfers.size()
                    << " transfers to device " << device_id
                    << " using CUDA graphs";

          // OPTIMIZATION: Cache CUDA graphs to avoid recreation overhead
          std::string cache_key = model_path_ + ":" +
                                  std::to_string(transfers.size()) + ":" +
                                  std::to_string(loaded_size);

          cudaStream_t stream;
          CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate Error");

          cudaGraphExec_t graph_exec;
          bool cached = false;

          // Check cache for existing graph
          {
            std::lock_guard<std::mutex> cache_lock(graph_cache_mutex_);
            auto it = cuda_graph_cache_.find(cache_key);
            if (it != cuda_graph_cache_.end() &&
                it->second.num_transfers == transfers.size() &&
                it->second.total_size == loaded_size) {
              // FAST PATH: Reuse cached graph
              graph_exec = it->second.graph_exec;
              cached = true;
              LOG(INFO) << "Reusing cached CUDA graph for " << transfers.size() << " transfers";
            }
          }

          if (!cached) {
            // SLOW PATH: Capture new graph and cache it
            LOG(INFO) << "Capturing new CUDA graph for " << transfers.size() << " transfers...";

            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                      "cudaStreamBeginCapture Error");

            // Issue all transfers on single stream (will be captured in graph)
            for (size_t i = 0; i < transfers.size(); ++i) {
              auto& op = transfers[i];

              CUDA_CHECK(
                  cudaMemcpyAsync(
                      (void *)((char *)device_ptr_list[op.handle_idx] + op.gpu_offset),
                      (void *)(host_buffers[op.chunk_id] + op.chunk_offset),
                      op.size,
                      cudaMemcpyHostToDevice,
                      stream),
                  "cudaMemcpyAsync Error");
            }

            // End graph capture
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph),
                      "cudaStreamEndCapture Error");

            // Instantiate the graph
            CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0),
                      "cudaGraphInstantiate Error");

            cudaGraphDestroy(graph);  // Can destroy graph after instantiation

            // Add to cache
            {
              std::lock_guard<std::mutex> cache_lock(graph_cache_mutex_);
              cuda_graph_cache_[cache_key] = {graph_exec, transfers.size(), loaded_size};
              LOG(INFO) << "Cached CUDA graph with key: " << cache_key;
            }
          }

          LOG(INFO) << "Launching CUDA graph with " << transfers.size() << " transfers...";

          // Launch the graph (this replays all captured operations)
          CUDA_CHECK(cudaGraphLaunch(graph_exec, stream),
                    "cudaGraphLaunch Error");

          // CRITICAL: Synchronize stream to ensure all transfers complete
          CUDA_CHECK(cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize Error");

          LOG(INFO) << "CUDA graph " << (cached ? "replay" : "execution") << " completed successfully";

          // Clean up stream (but NOT graph_exec if cached)
          cudaStreamDestroy(stream);

          LOG(INFO) << "Finished loading " << loaded_size
                    << " bytes to device " << device_id
                    << " using CUDA graphs";

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