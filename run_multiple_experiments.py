#!/usr/bin/env python3
import subprocess
import sys
import os

def run_experiment(model_name, n_runs=5):
    """
    Run the experiment n times with different output files for a specific model.
    
    Args:
        model_name (str): Name of the model to run experiments for
        n_runs (int): Number of times to run the experiment
    """
    
    # Create a safe filename from the model name
    safe_model_name = model_name.replace(":", "-").replace("/", "-")
    
    for i in range(1, n_runs + 1):
        print(f"\n{'='*60}")
        print(f"Running experiment {i}/{n_runs} for model: {model_name}")
        print(f"{'='*60}")
        
        # Run experiment for the specified model
        cmd = [
            "uv", "run", "main.py",
            "--model", model_name,
            "--defense-mechanism", "noop",
            "--modification", "noop",
            "--dataset-size", "100",
            "--output-path", f"{safe_model_name}-undefended-{i}"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, text=True)
        
        if result.returncode == 0:
            print(f"✅ {model_name} experiment {i} completed successfully")
        else:
            print(f"❌ {model_name} experiment {i} failed with error: {result.stderr}")
    
    print(f"\n{'='*60}")
    print(f"Completed {n_runs} experiment runs!")
    print(f"{'='*60}")
    
    # List generated files
    print("\nGenerated files:")
    for i in range(1, n_runs + 1):
        output_file = f"{safe_model_name}-undefended-{i}.jsonl"
        
        if os.path.exists(output_file):
            print(f"  ✅ {output_file}")
        else:
            print(f"  ❌ {output_file} (missing)")

if __name__ == "__main__":
    # Get model name and number of runs from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_multiple_experiments.py <model_name> [n_runs]")
        print("Example: python run_multiple_experiments.py deepseek-r1:7b 10")
        sys.exit(1)
    
    model_name = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"Running {n_runs} experiments for model: {model_name}")
    run_experiment(model_name, n_runs) 