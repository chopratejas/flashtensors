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

// Phase 3.1: io_uring async I/O
#ifdef HAVE_LIBURING
#include "io_uring_reader.h"
using snacktensors::IoUringReader;
using snacktensors::CompletedOp;
#endif

#include "error_handling.h"

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
  std::vector<std::future<int>> futures;
  size_t chunk_per_thread = (num_chunks + num_threads - 1) / num_threads;
  LOG(INFO) << "Loading model " << model_path_ << " to host with "
            << num_threads << " threads, " << num_chunks << " chunks, "
            << chunk_size << " chunk size, " << chunk_per_thread
            << " chunks per thread";

  state_ = MemoryState::LOADING;
  lock.unlock();

#ifdef HAVE_LIBURING
  // Phase 3.1: Use io_uring for async I/O (2-3x faster!)
  bool use_io_uring = IoUringReader::is_available();
  if (use_io_uring) {
    LOG(INFO) << "Using io_uring for async I/O (Phase 3.1)";

    try {
      // Create io_uring reader with optimal queue depth
      int queue_depth = std::min(256, static_cast<int>(num_chunks));
      IoUringReader reader(queue_depth);

      // Submit all read operations asynchronously
      for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        size_t size = std::min(chunk_size, model_size_ - chunk_idx * chunk_size);

        // Calculate which partition and offset
        size_t partition_id = 0;
        size_t file_offset = chunk_idx * chunk_size;
        while (partition_id < partition_sizes_.size() &&
               file_offset >= partition_sizes_.at(partition_id)) {
          file_offset -= partition_sizes_.at(partition_id);
          partition_id += 1;
        }

        if (partition_id >= partition_sizes_.size()) {
          LOG(ERROR) << "Invalid partition for chunk " << chunk_idx;
          use_io_uring = false;  // Fall back to pread
          break;
        }

        int fd = file_descriptors[partition_id];

        // Submit async read (non-blocking)
        if (reader.submit_read(fd, (void*)host_buffers[chunk_idx], size,
                               file_offset, (void*)(uintptr_t)chunk_idx) < 0) {
          LOG(WARNING) << "Failed to submit read for chunk " << chunk_idx
                       << ", falling back to pread()";
          use_io_uring = false;
          break;
        }
      }

      if (use_io_uring) {
        // All reads submitted successfully, wait for completions
        LOG(INFO) << "Submitted " << num_chunks << " async reads via io_uring";

        size_t completed = 0;
        while (completed < num_chunks) {
          // Wait for at least one completion
          int num_completed = reader.wait_completions(1);
          if (num_completed < 0) {
            LOG(ERROR) << "io_uring wait failed";
            use_io_uring = false;
            break;
          }

          // Process completed operations
          auto ops = reader.get_completed();
          for (const auto& op : ops) {
            size_t chunk_idx = (size_t)(uintptr_t)op.user_data;

            if (op.is_error) {
              LOG(ERROR) << "I/O error for chunk " << chunk_idx
                         << ": " << strerror(op.error_code);
              use_io_uring = false;
              break;
            }

            // Enqueue completed chunk
            size_t size = std::min(chunk_size, model_size_ - chunk_idx * chunk_size);
            host_ptr_vector_->enqueue(chunk_idx, Batch{chunk_idx, size});
            completed++;
          }

          if (!use_io_uring) break;
        }

        if (use_io_uring) {
          LOG(INFO) << "io_uring async I/O completed successfully ("
                    << completed << " chunks)";
        }
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "io_uring exception: " << e.what()
                   << ", falling back to pread()";
      use_io_uring = false;
    }
  }

  // Fallback to pread() if io_uring not available or failed
  if (!use_io_uring)
#endif
  {
    LOG(INFO) << "Using multi-threaded pread() for I/O";

    for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
      futures.emplace_back(std::async(std::launch::async, [&, thread_idx]() {
        size_t partition_id = 0;
        size_t file_offset = thread_idx * chunk_per_thread * chunk_size;
        while (partition_id < partition_sizes_.size() &&
               file_offset >= partition_sizes_.at(partition_id)) {
          file_offset -= partition_sizes_.at(partition_id);
          partition_id += 1;
        }
        if (partition_id >= partition_sizes_.size()) {
          LOG(INFO) << "Thread " << thread_idx << " early exits";
          return 0;
        }
        LOG(INFO) << "Thread " << thread_idx << " starting from partition "
                  << partition_id << " offset " << file_offset;
        for (size_t chunk_idx = thread_idx * chunk_per_thread;
             chunk_idx < (thread_idx + 1) * chunk_per_thread &&
             chunk_idx < num_chunks;
             ++chunk_idx) {
          size_t size =
              std::min(chunk_size, model_size_ - chunk_idx * chunk_size);
          if (host_buffers[chunk_idx] == nullptr) {
            LOG(ERROR) << "Host buffer not allocated";
            return -1;
          }

          if (state_ == MemoryState::CANCELLED) {
            LOG(INFO) << "Loading from disk for model " << model_path_
                      << " is cancelled";
            return 0;
          }

          int fd = file_descriptors[partition_id];
          ssize_t ret =
              pread(fd, (void *)host_buffers[chunk_idx], size, file_offset);
          if (ret < 0) {
            auto tensor_path = partition_paths_[partition_id];
            LOG(ERROR) << "pread() failed for file: " << tensor_path
                       << ", error: " << strerror(errno);
            return -1;
          } else if (ret != size) {
            if (ret < size && partition_id + 1 < file_descriptors.size()) {
              partition_id += 1;
              file_offset = 0;
              size_t remaining_size = size - ret;
              int fd = file_descriptors[partition_id];
              ret = pread(fd, (void *)(host_buffers[chunk_idx] + ret),
                          remaining_size, file_offset);
              if (ret != remaining_size) {
                auto tensor_path = partition_paths_[partition_id];
                LOG(ERROR) << "Failed to read file: " << tensor_path
                           << " read: " << ret << " expected: " << remaining_size;
                return -1;
              }
            } else {
              auto tensor_path = partition_paths_[partition_id];
              LOG(ERROR) << "Failed to read file: " << tensor_path
                         << " read: " << ret << " expected: " << size;
              return -1;
            }
          }
          file_offset += ret;

          host_ptr_vector_->enqueue(chunk_idx, Batch{chunk_idx, size});
        }

        return 0;
      }));
    }
  }

  bool error = false;
  for (auto &future : futures) {
    int ret = future.get();
    if (ret != 0) {
      LOG(ERROR) << "Error reading from disk, ret " << ret;
      error = true;
    }
  }

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

          // Phase 2: CUDA Graphs - Collect transfer operations first
          struct TransferOp {
            int chunk_id;
            size_t chunk_offset;
            size_t size;
            size_t gpu_offset;
            int handle_idx;
          };
          std::vector<TransferOp> transfer_ops;

          // Collect all transfer operations
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
            transfer_ops.push_back({chunk_id, chunk_offset, size, gpu_offset, handle_idx});
          }

          LOG(INFO) << "Collected " << transfer_ops.size()
                    << " transfer operations for device " << device_id;

          // Create multiple CUDA streams for parallel async transfers
          const int num_streams = 8;  // Optimal for most GPUs
          std::vector<cudaStream_t> streams(num_streams);
          for (int i = 0; i < num_streams; ++i) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]),
                      "cudaStreamCreate Error");
          }

          // Phase 2: CUDA Graph capture/replay
          // Check if we have a cached graph for this model+device
          static std::unordered_map<std::string, cudaGraph_t> graph_cache;
          static std::unordered_map<std::string, cudaGraphExec_t> graph_exec_cache;
          static std::mutex graph_mutex;

          std::string graph_key = model_path_ + "_dev" + std::to_string(device_id);
          bool use_graph = false;
          bool is_first_load = false;

          {
            std::lock_guard<std::mutex> lock(graph_mutex);
            use_graph = (graph_exec_cache.find(graph_key) != graph_exec_cache.end());
            is_first_load = !use_graph && transfer_ops.size() > 100; // Only for models with many chunks
          }

          size_t loaded_size = 0;

          if (use_graph) {
            // GRAPH REPLAY PATH - Fast! ⚡
            LOG(INFO) << "🚀 Using CUDA graph for device " << device_id
                      << " (" << transfer_ops.size() << " ops)";

            std::lock_guard<std::mutex> lock(graph_mutex);
            cudaGraphExec_t graph_exec = graph_exec_cache[graph_key];

            // Launch the entire graph with one call!
            CUDA_CHECK(cudaGraphLaunch(graph_exec, streams[0]),
                      "cudaGraphLaunch Error");
            CUDA_CHECK(cudaStreamSynchronize(streams[0]),
                      "Graph sync Error");

            // Calculate total size
            for (const auto &op : transfer_ops) {
              loaded_size += op.size;
            }

            LOG(INFO) << "✅ Graph replay complete: " << loaded_size << " bytes";

          } else if (is_first_load) {
            // GRAPH CAPTURE PATH - First time for this model
            LOG(INFO) << "📸 Capturing CUDA graph for device " << device_id
                      << " (" << transfer_ops.size() << " ops)";

            cudaGraph_t graph;
            cudaGraphExec_t graph_exec;

            // Begin graph capture on stream 0
            CUDA_CHECK(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal),
                      "cudaStreamBeginCapture Error");

            // Execute all transfers (they get recorded into the graph)
            for (size_t i = 0; i < transfer_ops.size(); ++i) {
              const auto &op = transfer_ops[i];

              CUDA_CHECK(
                  cudaMemcpyAsync(
                      (void *)((char *)device_ptr_list[op.handle_idx] + op.gpu_offset),
                      (void *)(host_buffers[op.chunk_id] + op.chunk_offset),
                      op.size,
                      cudaMemcpyHostToDevice,
                      streams[0]),  // All on same stream for capture
                  "cudaMemcpyAsync Error (capture)");

              loaded_size += op.size;
            }

            // End capture
            CUDA_CHECK(cudaStreamEndCapture(streams[0], &graph),
                      "cudaStreamEndCapture Error");

            // Instantiate the graph for replay
            CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0),
                      "cudaGraphInstantiate Error");

            // Store in cache
            {
              std::lock_guard<std::mutex> lock(graph_mutex);
              graph_cache[graph_key] = graph;
              graph_exec_cache[graph_key] = graph_exec;
            }

            LOG(INFO) << "✅ Graph captured and cached: " << loaded_size << " bytes";

          } else {
            // NORMAL PATH - For small models or when graphs disabled
            LOG(INFO) << "Executing normal multi-stream transfers for device "
                      << device_id << " (" << transfer_ops.size() << " ops)";

            for (size_t i = 0; i < transfer_ops.size(); ++i) {
              const auto &op = transfer_ops[i];

              // Use async memcpy with round-robin stream distribution
              CUDA_CHECK(
                  cudaMemcpyAsync(
                      (void *)((char *)device_ptr_list[op.handle_idx] + op.gpu_offset),
                      (void *)(host_buffers[op.chunk_id] + op.chunk_offset),
                      op.size,
                      cudaMemcpyHostToDevice,
                      streams[i % num_streams]),
                  "cudaMemcpyAsync Error");

              loaded_size += op.size;
            }
          }

          // Synchronize all streams to ensure all transfers complete
          for (int i = 0; i < num_streams; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]),
                      "cudaStreamSynchronize Error");
            CUDA_CHECK(cudaStreamDestroy(streams[i]),
                      "cudaStreamDestroy Error");
          }

          LOG(INFO) << "Finished loading tensor from memory to device "
                    << device_id;

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