#!/usr/bin/env python3
import subprocess
import sys
import os

def run_experiment(n_runs=5):
    """
    Run the experiment n times with different output files.
    
    Args:
        n_runs (int): Number of times to run the experiment
    """
    
    for i in range(1, n_runs + 1):
        print(f"\n{'='*60}")
        print(f"Running experiment {i}/{n_runs}")
        print(f"{'='*60}")
        
        # Run Llama experiment
        llama_cmd = [
            "uv", "run", "main.py",
            "--model", "llama3.2:latest",
            "--defense-mechanism", "noop",
            "--modification", "noop",
            "--dataset-size", "100",
            "--output-path", f"llama-undefended-{i}"
        ]
        
        print(f"Running: {' '.join(llama_cmd)}")
        result = subprocess.run(llama_cmd, text=True)
        
        if result.returncode == 0:
            print("✅ Llama experiment completed successfully")
        else:
            print(f"❌ Llama experiment failed with error: {result.stderr}")
            continue
        
        # Run Qwen experiment
        qwen_cmd = [
            "uv", "run", "main.py",
            "--model", "qwen2.5:7b",
            "--defense-mechanism", "noop",
            "--modification", "noop",
            "--dataset-size", "100",
            "--output-path", f"qwen-undefended-{i}"
        ]
        
        print(f"Running: {' '.join(qwen_cmd)}")
        result = subprocess.run(qwen_cmd, text=True)
        
        if result.returncode == 0:
            print("✅ Qwen experiment completed successfully")
        else:
            print(f"❌ Qwen experiment failed with error: {result.stderr}")
    
    print(f"\n{'='*60}")
    print(f"Completed {n_runs} experiment runs!")
    print(f"{'='*60}")
    
    # List generated files
    print("\nGenerated files:")
    for i in range(1, n_runs + 1):
        llama_file = f"llama-undefended-{i}.jsonl"
        qwen_file = f"qwen-undefended-{i}.jsonl"
        
        if os.path.exists(llama_file):
            print(f"  ✅ {llama_file}")
        else:
            print(f"  ❌ {llama_file} (missing)")
            
        if os.path.exists(qwen_file):
            print(f"  ✅ {qwen_file}")
        else:
            print(f"  ❌ {qwen_file} (missing)")

if __name__ == "__main__":
    # Get number of runs from command line argument, default to 5
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_experiment(n) 