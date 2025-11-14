#!/bin/bash
# Activate NCCL fix for flashtensors
# Source this before running: source activate_nccl_fix.sh

export LD_LIBRARY_PATH=/apps/default-python/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/root/flashtensors/libnccl_stub.so:$LD_PRELOAD

echo "✅ NCCL compatibility fix activated"
