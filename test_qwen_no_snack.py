#!/usr/bin/env python3
"""
Test Qwen model WITHOUT SnackTensors to verify the model works correctly.
"""

import sys
import time
from vllm import LLM, SamplingParams

def main():
    print("=" * 70)
    print("  Qwen Model Test WITHOUT SnackTensors")
    print("=" * 70)
    print()

    # Load model directly with vLLM (no SnackTensors)
    model_id = "Qwen/Qwen3-0.6B"

    print(f"⚡ Loading model {model_id} with standard vLLM...")
    load_start = time.time()

    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        disable_log_stats=True
    )

    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.2f}s\n")

    # Run inference
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

    print("✅ Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
