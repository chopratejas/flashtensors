#!/usr/bin/env python3
"""
Test Qwen32B model - delete existing, pull fresh, and run inference
to check if the "!!!" output issue is fixed.
"""

import sys
import os
import time
import shutil

# Add the flashtensors directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Try to find and add CUDA libraries to LD_LIBRARY_PATH
def setup_cuda_libs():
    """Find and add CUDA libraries to LD_LIBRARY_PATH"""
    try:
        import torch
        torch_path = os.path.dirname(torch.__file__)
        
        # Common CUDA library locations
        cuda_paths = [
            os.path.join(torch_path, "lib"),
            os.path.join(os.path.dirname(torch_path), "lib"),
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/lib",
            "/usr/lib/x86_64-linux-gnu",
        ]
        
        found_paths = []
        for path in cuda_paths:
            if os.path.exists(path):
                # Check for any CUDA runtime library
                try:
                    files = os.listdir(path)
                    for f in files:
                        if "libcudart" in f.lower():
                            if path not in os.environ.get("LD_LIBRARY_PATH", ""):
                                current_ld = os.environ.get("LD_LIBRARY_PATH", "")
                                os.environ["LD_LIBRARY_PATH"] = f"{path}:{current_ld}" if current_ld else path
                                found_paths.append(path)
                                print(f"✅ Added CUDA library path: {path} (found {f})")
                                break
                except:
                    pass
        
        return found_paths[0] if found_paths else None
    except Exception as e:
        print(f"⚠️  Could not auto-detect CUDA libraries: {e}")
    return None

cuda_path = setup_cuda_libs()
if not cuda_path:
    print("⚠️  No CUDA libraries found - vLLM may not work, but transformers backend should be fine")

try:
    import torch
    import flashtensors as flash
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"Failed to import flashtensors: {e}")
    print("Make sure you've built and installed the C++ extensions")
    sys.exit(1)


def main():
    print("=" * 70)
    print("  Qwen32B Model Test - Delete, Pull, and Run Inference")
    print("=" * 70)
    print()

    # Step 1: Configure FlashEngine
    print("🔧 Configuring FlashTensors...")
    flash.shutdown_server()  # Ensure clean start

    # Use /tmp/models if /workspace doesn't exist
    storage_path = "/workspace" if os.path.exists("/workspace") else "/tmp/models"
    if not os.path.exists(storage_path):
        os.makedirs(storage_path, exist_ok=True)
        print(f"✅ Created storage directory: {storage_path}")
    
    # Try to configure - if vLLM import fails, we can still use transformers backend
    try:
        flash.configure(
            storage_path=storage_path,
            mem_pool_size=1024**3*80,           # 80GB memory pool (Qwen32B needs ~61GB, so 80GB is safe)
            chunk_size=1024**2*32,              # 32MB chunks
            num_threads=4,
            gpu_memory_utilization=0.8,         # Use 80% of GPU memory
            server_host="0.0.0.0",
            server_port=8073
        )
        print("✅ FlashTensors configured\n")
    except (ImportError, RuntimeError) as e:
        error_str = str(e).lower()
        if "libcudart" in error_str or "vllm" in error_str or "cannot open shared object" in error_str:
            print(f"⚠️  Warning: vLLM import failed ({e})")
            print("   This is OK for transformers backend - continuing...")
            # Manually configure storage path without vLLM
            from flashtensors.config import update_config, get_config
            from flashtensors.server_manager import ensure_server_running
            
            # Update config manually
            try:
                update_config(
                    storage_path=storage_path,
                    mem_pool_size=1024**3*80,  # 80GB memory pool for Qwen32B
                    chunk_size=1024**2*32,
                    num_threads=4,
                    gpu_memory_utilization=0.8,
                    server_host="0.0.0.0",
                    server_port=8073
                )
            except RuntimeError as re:
                # Server might already be running, that's OK
                if "server is currently running" not in str(re):
                    raise
            
            # Start server with longer timeout (memory pool creation can take 30-60 seconds)
            print("   Starting storage server (this may take 30-60 seconds for memory pool initialization)...")
            import time as time_module
            start_time = time_module.time()
            timeout = 90  # Increase timeout to 90 seconds for large memory pools
            
            # Wait longer for server to initialize
            server_started = False
            for i in range(18):  # Check every 5 seconds for 90 seconds
                if ensure_server_running():
                    server_started = True
                    break
                if i < 17:  # Don't print on last iteration
                    print(f"   Waiting for server to initialize... ({i*5}s)")
                time_module.sleep(5)
            
            if server_started:
                print("✅ Storage server started (without vLLM)\n")
            else:
                # Check logs for more info
                error_log = "/tmp/flashtensors_storage_server_error.log"
                if os.path.exists(error_log):
                    print(f"\n❌ Server failed to start. Last error log entries:")
                    with open(error_log, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-10:]:
                            print(f"   {line.rstrip()}")
                print("❌ Failed to start storage server")
                print("   Checking if server is actually running...")
                # Wait a bit more and check again
                import time as time_module
                time_module.sleep(5)
                from flashtensors.config import is_server_running
                if is_server_running():
                    print("✅ Server is actually running (health check passed)\n")
                else:
                    print("   Server health check failed - checking logs...")
                    error_log = "/tmp/flashtensors_storage_server_error.log"
                    if os.path.exists(error_log):
                        with open(error_log, 'r') as f:
                            lines = f.readlines()
                            print("   Last 20 lines of error log:")
                            for line in lines[-20:]:
                                print(f"   {line.rstrip()}")
                    print("\n   Please check server logs manually:")
                    print("   - /tmp/flashtensors_storage_server_error.log")
                    print("   - /tmp/flashtensors_storage_server.log")
                    sys.exit(1)
        else:
            raise

    # Step 2: Check for existing Qwen32B models
    model_id = "Qwen/Qwen2.5-32B-Instruct"  # Common Qwen32B model name
    print(f"📋 Checking for existing {model_id} models...")
    models = flash.list_models()
    
    # Check for any Qwen32B variants
    qwen32b_models = [k for k in models.keys() if "32" in k.upper() and "QWEN" in k.upper()]
    
    if qwen32b_models:
        print(f"✅ Found {len(qwen32b_models)} existing Qwen32B model(s):")
        for model_key in qwen32b_models:
            print(f"  - {model_key}")
        print("   Using existing model (skipping re-pull)\n")
    else:
        print("⚠️  No existing Qwen32B models found")
        print(f"🔄 Registering {model_id}...")
        result = flash.register_model(
            model_id=model_id,
            backend="transformers",
            torch_dtype="float16",
            force=False,  # Don't force re-download
            hf_token=None
        )
        
        if result['status'] != 'success':
            print(f"❌ Failed to register model: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        
        print(f"✅ Model registered: {result['status']}\n")

    # Step 4: Load model
    print(f"⚡ Loading model {model_id}...")
    load_start = time.time()

    try:
        model = flash.load_model(
            model_id=model_id,
            backend="transformers",
            torch_dtype="float16",
            device_map="auto",
            hf_model_class="AutoModelForCausalLM"
        )
        
        # Load tokenizer
        model_path = os.path.join(storage_path, "transformers", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        load_time = time.time() - load_start
        print(f"✅ Model loaded in {load_time:.2f}s\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 5: Run inference
    print("🤖 Running inference...")
    print("-" * 70)

    prompt = "What is 2+2? Please explain your answer."
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, 'to'):
            # Move inputs to the same device as the model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        inference_start = time.time()
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        inference_time = time.time() - inference_start
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"📝 Inference Results (took {inference_time:.2f}s):")
        print("-" * 70)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print()
        
        # Check for the "!!!" issue
        if "!!!" in generated_text:
            print("⚠️  WARNING: Output contains '!!!' - this may indicate the tensor splitting issue!")
        else:
            print("✅ Output looks good - no '!!!' detected")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 6: Cleanup
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

