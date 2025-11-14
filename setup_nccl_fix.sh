#!/bin/bash
# Setup script to fix NCCL compatibility for flashtensors
# This should be sourced before running Python scripts

export LD_LIBRARY_PATH=/apps/default-python/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/tmp/libnccl_stub.so:$LD_PRELOAD

echo "✅ NCCL compatibility fix applied"
echo "   LD_LIBRARY_PATH includes torch's NCCL"
echo "   LD_PRELOAD includes NCCL stub library"
