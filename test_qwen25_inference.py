#!/usr/bin/env python3
"""
Test the Qwen2.5-0.5B model that was registered earlier
"""

import sys
import os
import time
from vllm import SamplingParams

sys.path.insert(0, os.path.dirname(__file__))

try:
    import snacktensors as flash
except ImportError as e:
    print(f"Failed to import snacktensors: {e}")
    sys.exit(1)


def main():
    print("=" * 70)
    print("  Qwen 2.5-0.5B Model Test (registered earlier)")
    print("=" * 70)
    print()

    flash.shutdown_server()
    flash.configure(
        storage_path="/tmp/models",
        mem_pool_size=1024**3*30,
        chunk_size=1024**2*32,
        num_threads=4,
        gpu_memory_utilization=0.8,
        server_host="0.0.0.0",
        server_port=8073
    )
    flash.activate_vllm_integration()

    model_id = "Qwen/Qwen2.5-0.5B"

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

    print("🤖 Running inference...")
    print("-" * 70)

    prompts = ["What is 2+2?"]
    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=100)

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

    flash.cleanup_gpu()
    print("✅ Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        flash.shutdown_server()
        sys.exit(1)
