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
#include "tensor_writer.h"

#include <iostream>

TensorWriter::TensorWriter(const std::string &filename) : filename_(filename) {}

TensorWriter::~TensorWriter() {}

uint64_t TensorWriter::writeRecord(const char *data, size_t size) {
  uint64_t start_offset = offset_;
  
  // CRITICAL FIX: Handle tensors that span partition boundaries
  // We need to split the write across partitions if necessary
  size_t remaining_size = size;
  const char* current_data = data;
  size_t padding_needed = (size % 8) ? (8 - size % 8) : 0;
  
  while (remaining_size > 0) {
    // Check if we need a new partition
    // Account for padding that will be added at the end of the tensor (only on last chunk)
    size_t total_needed = remaining_size + (remaining_size <= padding_needed + 8 ? padding_needed : 0);
    
    if (partition_idx_ == -1 || partition_size_ + total_needed > kPartitionMaxSize) {
      // Check if we can fit at least some data in current partition
      if (partition_idx_ >= 0 && partition_size_ < kPartitionMaxSize) {
        // Current partition has space but not enough for entire tensor + padding
        // We need to split: write what fits, then move to next partition
        size_t space_in_partition = kPartitionMaxSize - partition_size_;
        
        // Write as much as we can to current partition (without padding - padding goes at end)
        if (space_in_partition > 0) {
          size_t write_to_current = std::min(remaining_size, space_in_partition);
          size_t written = buffer_->writeData(current_data, write_to_current);
          offset_ += written;
          partition_size_ += written;
          remaining_size -= written;
          current_data += written;
        }
      }
      
      // Create new partition
      partition_idx_++;
      partition_size_ = 0;
      std::string partition_filename =
          filename_ + "_" + std::to_string(partition_idx_);
      buffer_ = std::make_unique<AlignedBuffer>(partition_filename);
    }
    
    // Write remaining data (or all data if first partition)
    size_t written = buffer_->writeData(current_data, remaining_size);
    
    // Add padding only at the very end of the complete tensor
    // This happens when we've written all remaining_size bytes
    if (padding_needed > 0 && written == remaining_size) {
      written += buffer_->writePadding(padding_needed);
    }
    
    offset_ += written;
    partition_size_ += written;
    remaining_size -= written;
    current_data += written;
  }
  
  return start_offset;
}
