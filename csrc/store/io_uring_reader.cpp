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
#include "io_uring_reader.h"

#include <glog/logging.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

namespace snacktensors {

IoUringReader::IoUringReader(int queue_depth)
    : queue_depth_(queue_depth), pending_ops_(0), total_completed_(0) {

    // Initialize io_uring with specified queue depth
    int ret = io_uring_queue_init(queue_depth_, &ring_, 0);
    if (ret < 0) {
        LOG(ERROR) << "io_uring_queue_init failed: " << strerror(-ret);
        throw std::runtime_error("Failed to initialize io_uring");
    }

    LOG(INFO) << "io_uring initialized with queue depth " << queue_depth_;

    // Reserve space for completed operations
    completed_ops_.reserve(queue_depth_);
}

IoUringReader::~IoUringReader() {
    // Clean up io_uring resources
    io_uring_queue_exit(&ring_);
    LOG(INFO) << "io_uring cleaned up. Total ops completed: " << total_completed_;
}

int IoUringReader::submit_read(int fd, void* buf, size_t size, off_t offset,
                               void* user_data) {
    // Get a submission queue entry (SQE)
    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) {
        // Queue is full, need to submit first
        int submitted = submit_pending();
        if (submitted < 0) {
            LOG(ERROR) << "Failed to submit pending operations";
            return -1;
        }

        // Try again
        sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {
            LOG(ERROR) << "Failed to get SQE even after submit";
            return -1;
        }
    }

    // Prepare read operation
    io_uring_prep_read(sqe, fd, buf, size, offset);

    // Set user data for completion identification
    io_uring_sqe_set_data(sqe, user_data);

    pending_ops_++;

    return 0;
}

int IoUringReader::submit_pending() {
    if (pending_ops_ == 0) {
        return 0;  // Nothing to submit
    }

    // Submit all pending operations to kernel
    int submitted = io_uring_submit(&ring_);
    if (submitted < 0) {
        LOG(ERROR) << "io_uring_submit failed: " << strerror(-submitted);
        return -1;
    }

    LOG(INFO) << "Submitted " << submitted << " io_uring operations";
    return submitted;
}

int IoUringReader::wait_completions(int min_complete) {
    // Submit any pending operations first
    if (pending_ops_ > 0) {
        int submitted = submit_pending();
        if (submitted < 0) {
            return -1;
        }
    }

    if (min_complete <= 0) {
        min_complete = 1;
    }

    // Wait for completions
    struct io_uring_cqe* cqe;
    int completed = 0;

    while (completed < min_complete && pending_ops_ > 0) {
        int ret = io_uring_wait_cqe(&ring_, &cqe);
        if (ret < 0) {
            LOG(ERROR) << "io_uring_wait_cqe failed: " << strerror(-ret);
            return -1;
        }

        // Process completion
        CompletedOp op;
        op.user_data = io_uring_cqe_get_data(cqe);
        op.result = cqe->res;

        if (cqe->res < 0) {
            // Error occurred
            op.is_error = true;
            op.error_code = -cqe->res;
            LOG(WARNING) << "I/O operation failed: " << strerror(-cqe->res);
        } else {
            // Success
            op.is_error = false;
            op.error_code = 0;
        }

        completed_ops_.push_back(op);
        completed++;
        pending_ops_--;
        total_completed_++;

        // Mark this CQE as seen
        io_uring_cqe_seen(&ring_, cqe);
    }

    // Check for additional completions that are already ready
    while (pending_ops_ > 0) {
        int ret = io_uring_peek_cqe(&ring_, &cqe);
        if (ret < 0) {
            // No more ready completions
            break;
        }

        // Process completion
        CompletedOp op;
        op.user_data = io_uring_cqe_get_data(cqe);
        op.result = cqe->res;

        if (cqe->res < 0) {
            op.is_error = true;
            op.error_code = -cqe->res;
        } else {
            op.is_error = false;
            op.error_code = 0;
        }

        completed_ops_.push_back(op);
        completed++;
        pending_ops_--;
        total_completed_++;

        io_uring_cqe_seen(&ring_, cqe);
    }

    LOG(INFO) << "Processed " << completed << " completions, "
              << pending_ops_ << " still pending";

    return completed;
}

std::vector<CompletedOp> IoUringReader::get_completed() {
    // Return all completed operations and clear internal buffer
    std::vector<CompletedOp> result;
    result.swap(completed_ops_);
    return result;
}

bool IoUringReader::is_available() {
    // Try to create a small io_uring instance
    struct io_uring test_ring;
    int ret = io_uring_queue_init(2, &test_ring, 0);

    if (ret < 0) {
        LOG(INFO) << "io_uring not available: " << strerror(-ret);
        return false;
    }

    // Clean up test ring
    io_uring_queue_exit(&test_ring);
    return true;
}

}  // namespace snacktensors
