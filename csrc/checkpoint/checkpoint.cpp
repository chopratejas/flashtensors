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

#include "checkpoint.h"

#include <ATen/cuda/CUDABlas.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include <fcntl.h>
#include <nvml.h>
#include <sys/stat.h>
#include <torch/extension.h>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "progress_bar.h"
#include "tensor_writer.h"

#define BUFFER_SIZE 1 << 30

std::unordered_map<std::string, uint64_t> SaveTensors(
    std::vector<std::string> tensor_names,
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> &tensor_data,
    const std::string &path) {
  std::string tensor_filename = std::filesystem::path(path) / "tensor.data";
  std::cerr << "SaveTensors: path=" << path << ", tensor_filename=" << tensor_filename
            << ", num_tensors=" << tensor_names.size() << std::endl;

  // make a tensor writer
  TensorWriter writer(tensor_filename);
  // make a tensor index
  std::unordered_map<std::string, uint64_t> tensor_offsets;
  // Some tensors may share the same data, so we need to record the data to
  // avoid duplication
  std::unordered_map<const char *, std::string> data_record;

  int total = tensor_names.size();
  int count = 0;
  size_t total_written = 0;

  for (const auto &name : tensor_names) {
    const auto &[base, size] = tensor_data[name];
    const char *data_ptr = reinterpret_cast<const char *>(base);
    if (data_record.find(data_ptr) != data_record.end()) {
      tensor_offsets[name] = tensor_offsets[data_record[data_ptr]];
      continue;
    }
    data_record[data_ptr] = name;

    if (count == 0) {
      std::cerr << "First tensor: name=" << name << ", base=" << base
                << ", size=" << size << ", data_ptr=" << (void*)data_ptr << std::endl;
    }

    uint64_t offset = writer.writeRecord(data_ptr, size);
    tensor_offsets[name] = offset;
    total_written += size;

    // Update progress bar
    count++;
    float progress = static_cast<float>(count) / total;
    showProgressBar(progress, "Saving tensors: ");
  }

  std::cerr << "SaveTensors: wrote " << count << " tensors, total_written="
            << total_written / 1024 / 1024 << "MB" << std::endl;

  return tensor_offsets;
}

// Function to print the binary array in hexadecimal format
void printBinaryArrayInHex(const unsigned char *data, size_t size) {
  std::cout << "Data in Hex: ";
  for (size_t i = 0; i < size; ++i) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(data[i]) << " ";
  }
  std::cout << std::dec
            << std::endl; // Switch back to decimal for any future output
}

// Mapping from string to at::ScalarType
at::ScalarType stringToScalarType(const std::string &dtype_str) {
  static const std::unordered_map<std::string, at::ScalarType> dtype_map = {
      {"torch.float16", torch::kFloat16},  {"torch.float32", torch::kFloat32},
      {"torch.float64", torch::kFloat64},  {"torch.int16", torch::kInt16},
      {"torch.int32", torch::kInt32},      {"torch.int64", torch::kInt64},
      {"torch.uint8", torch::kUInt8},      {"torch.int8", torch::kInt8},
      {"torch.bfloat16", torch::kBFloat16}};

  auto it = dtype_map.find(dtype_str);
  if (it != dtype_map.end()) {
    return it->second;
  } else {
    throw std::invalid_argument("Unknown dtype string: " + dtype_str);
  }
}

std::unordered_map<std::string, torch::Tensor> RestoreTensors(
    const std::unordered_map<
        std::string,
        std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::string>>
        &meta_state_dict,
    const std::unordered_map<int, void *> &memory_base_address,
    const std::unordered_map<int, std::unordered_map<std::string, uint64_t>>
        &tensor_device_offsets) {

  // Pre-calculate total number of tensors for better allocation
  size_t total_tensors = 0;
  for (const auto &[device, tensor_offset] : tensor_device_offsets) {
    total_tensors += tensor_offset.size();
  }

  // Reserve capacity upfront to avoid rehashing
  std::unordered_map<std::string, torch::Tensor> state_dict;
  state_dict.reserve(total_tensors);

  std::unordered_set<void *> handled_memory;
  handled_memory.reserve(memory_base_address.size());

  // Pre-create TensorOptions objects to avoid repeated construction
  std::unordered_map<int, torch::TensorOptions> device_options_cache;
  for (const auto &[device, _] : memory_base_address) {
    device_options_cache[device] = torch::TensorOptions()
        .device(torch::Device(torch::kCUDA, device));
  }

  for (const auto &[device, tensor_offset] : tensor_device_offsets) {
    // Check device existence once per device
    auto mem_it = memory_base_address.find(device);
    if (mem_it == memory_base_address.end()) {
      std::cerr << "Cannot find device " << device << std::endl;
      exit(1);
    }

    void *base_address = mem_it->second;
    const auto &base_options = device_options_cache.at(device);

    for (const auto &[name, tensor_offset_value] : tensor_offset) {
      uint64_t offset = reinterpret_cast<uint64_t>(base_address) + tensor_offset_value;

      // Use const reference to avoid copy
      const auto &meta = meta_state_dict.at(name);
      const auto &sizes = std::get<0>(meta);
      const auto &strides = std::get<1>(meta);
      at::ScalarType dtype = stringToScalarType(std::get<2>(meta));

      // Create tensor options with dtype
      auto tensor_options = base_options.dtype(dtype);

      if (tensor_offset_value == 0 &&
          handled_memory.find(base_address) == handled_memory.end()) {
        // First tensor owns the memory with cudaFree deleter
        torch::Tensor real_tensor = torch::from_blob(
            reinterpret_cast<void *>(offset),
            c10::makeArrayRef(sizes),
            c10::makeArrayRef(strides),
            [](void *ptr) { cudaFree(ptr); },
            tensor_options);
        state_dict.emplace(name, std::move(real_tensor));
        handled_memory.insert(base_address);
      } else {
        // Other tensors are views with no-op deleter
        torch::Tensor real_tensor = torch::from_blob(
            reinterpret_cast<void *>(offset),
            sizes,
            strides,
            [](void *ptr) {},
            tensor_options);
        state_dict.emplace(name, std::move(real_tensor));
      }
    }
  }
  return state_dict;
}

std::unordered_map<std::string, int> GetGpuUUID() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount); // Get the number of CUDA devices
  std::unordered_map<std::string, int> uuidToDeviceIdMap;

  for (int devId = 0; devId < deviceCount; ++devId) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, devId); // Get properties for each device

    // Convert UUID bytes to string with unsigned char casting
    char uuidStr[80];
    snprintf(
        uuidStr, sizeof(uuidStr),
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        (unsigned char)props.uuid.bytes[0], (unsigned char)props.uuid.bytes[1],
        (unsigned char)props.uuid.bytes[2], (unsigned char)props.uuid.bytes[3],
        (unsigned char)props.uuid.bytes[4], (unsigned char)props.uuid.bytes[5],
        (unsigned char)props.uuid.bytes[6], (unsigned char)props.uuid.bytes[7],
        (unsigned char)props.uuid.bytes[8], (unsigned char)props.uuid.bytes[9],
        (unsigned char)props.uuid.bytes[10],
        (unsigned char)props.uuid.bytes[11],
        (unsigned char)props.uuid.bytes[12],
        (unsigned char)props.uuid.bytes[13],
        (unsigned char)props.uuid.bytes[14],
        (unsigned char)props.uuid.bytes[15]);

    uuidToDeviceIdMap[std::string(uuidStr)] = devId;
  }

  return uuidToDeviceIdMap;
}

std::unordered_map<int, void *>
AllocateCudaMemory(const std::unordered_map<int, size_t> &tensor_sizes) {
  std::unordered_map<int, void *> memory_ptrs;
  for (const auto &p : tensor_sizes) {
    int device = p.first;
    size_t size = p.second;
    void *ptr = nullptr;
    cudaSetDevice(device);
    cudaMalloc(&ptr, size);
    memory_ptrs[device] = ptr;
  }
  return memory_ptrs;
}

std::unordered_map<int, std::string>
GetCudaMemoryHandles(const std::unordered_map<int, void *> &memory_ptrs) {
  std::unordered_map<int, std::string> memory_handles;
  for (const auto &p : memory_ptrs) {
    int device = p.first;
    void *ptr = p.second;
    cudaIpcMemHandle_t handle;
    cudaSetDevice(device);
    cudaIpcGetMemHandle(&handle, ptr);
    memory_handles[device] = std::string(
        reinterpret_cast<const char *>(&handle), sizeof(cudaIpcMemHandle_t));
  }
  return memory_handles;
}

std::unordered_map<int, std::vector<std::string>> GetCudaMemoryHandles(
    const std::unordered_map<int, std::vector<void *>> &memory_ptrs) {
  std::unordered_map<int, std::vector<std::string>> memory_handles;
  for (const auto &p : memory_ptrs) {
    auto device = p.first;
    const auto &ptrs = p.second;
    cudaIpcMemHandle_t handle;
    cudaSetDevice(device);

    std::vector<std::string> handles;
    for (const auto &ptr : ptrs) {
      cudaIpcGetMemHandle(&handle, ptr);
      handles.push_back(std::string(reinterpret_cast<const char *>(&handle),
                                    sizeof(cudaIpcMemHandle_t)));
    }
    memory_handles[device] = handles;
  }
  return memory_handles;
}

std::unordered_map<int, std::string> GetDeviceUuidMap() {
  std::unordered_map<std::string, int> gpu_uuid = GetGpuUUID();
  std::unordered_map<int, std::string> device_uuid_map;
  for (const auto &p : gpu_uuid) {
    if (device_uuid_map.find(p.second) != device_uuid_map.end()) {
      std::cerr << "Duplicate device id: " << p.second << std::endl;
      exit(1);
    }
    device_uuid_map[p.second] = p.first;
  }
  return device_uuid_map;
}