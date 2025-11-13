#!/usr/bin/env python3
"""
Simple inference test for Qwen model with the prompt "What is 2+2?"
"""

import sys
import os
import time
from vllm import SamplingParams

# Add the flashtensors directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    import flashtensors as flash
except ImportError as e:
    print(f"Failed to import flashtensors: {e}")
    print("Make sure you've built and installed the C++ extensions")
    sys.exit(1)


def main():
    print("=" * 70)
    print("  Qwen Model Inference Test: What is 2+2?")
    print("=" * 70)
    print()

    # Step 1: Configure FlashEngine
    print("🔧 Configuring SnackTensors...")

    flash.shutdown_server()  # Ensure clean start

    flash.configure(
        storage_path="/tmp/models",
        mem_pool_size=1024**3*30,           # 30GB memory pool
        chunk_size=1024**2*32,              # 32MB chunks
        num_threads=4,
        gpu_memory_utilization=0.8,         # Use 80% of GPU memory
        server_host="0.0.0.0",
        server_port=8073
    )

    flash.activate_vllm_integration()
    print("✅ SnackTensors configured\n")

    # Step 2: Model ID
    model_id = "Qwen/Qwen3-0.6B"

    # Check if model is already registered
    print(f"📋 Checking if {model_id} is already registered...")
    models = flash.list_models()
    model_key = f"{model_id}:vllm"

    if model_key not in models:
        print(f"🔄 Model not found. Registering {model_id}...")
        result = flash.register_model(
            model_id=model_id,
            backend="vllm",
            torch_dtype="bfloat16",
            force=False,
            hf_token=None
        )
        print(f"✅ Model registered: {result['status']}\n")
    else:
        print(f"✅ Model already registered\n")

    # Step 3: Load model
    print(f"⚡ Loading model {model_id}...")
    load_start = time.time()

    llm = flash.load_model(
        model_id=model_id,
        backend="vllm",
        dtype="bfloat16",
        gpu_memory_utilization=0.8
    )

    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.2f}s\n")

    # Step 4: Run inference with "What is 2+2?"
    print("🤖 Running inference...")
    print("-" * 70)

    prompts = ["What is 2+2?"]

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=100
    )

    inference_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    inference_time = time.time() - inference_start

    print(f"📝 Inference Results (took {inference_time:.2f}s):")
    print("-" * 70)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Answer: {generated_text}")
        print()

    # Step 5: Cleanup
    print("🧹 Cleaning up...")
    flash.cleanup_gpu()
    print("✅ Done!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        flash.shutdown_server()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        flash.shutdown_server()
        sys.exit(1)
