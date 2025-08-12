#!/usr/bin/env python3
import subprocess
import sys
import os
import argparse
from pathlib import Path
import re

def translate_model_name(model_name: str, server_type: str) -> str:
    """Translate model names between HuggingFace and Ollama formats using regex."""
    
    if server_type == "ollama":
        # Handle Qwen models (e.g., Qwen/Qwen3-8B -> qwen3:8b)
        qwen_match = re.match(r'Qwen/Qwen(\d+)-(\d+)B', model_name)
        if qwen_match:
            version, size = qwen_match.groups()
            return f"qwen{version}:{size}b"
            
        # Handle Llama models (e.g., meta-llama/Llama-2-7B-Instruct -> llama2:7b)
        llama_match = re.match(r'meta-llama/Llama-(\d+(?:\.\d+)?)-(\d+)B(?:-Instruct)?', model_name)
        if llama_match:
            version, size = llama_match.groups()
            return f"llama{version}:{size}b"
            
        # Handle Phi models with hardcoded match
        if model_name.startswith("microsoft/Phi-4-mini-instruct"):
            return "phi4-mini:3.8b"
    
    return model_name

def generate_output_path(model: str, cluster_id: int, tool_index: int, server_type: str) -> str:
    """Generate a structured output path based on experiment parameters."""
    # Create base directory if it doesn't exist
    base_dir = Path("vllm" if server_type == "vllm" else "ollama")
    base_dir.mkdir(exist_ok=True)
    
    # Extract model name for filename
    model_short_name = model.split('/')[-1].lower()
    
    # Format: {server_type}/cluster-{id}-tool-{index}-{model}-q0-100
    filename = f"cluster-{cluster_id}-tool-{tool_index}-{model_short_name}-q0-100"
    
    return str(base_dir / filename)

def run_experiment(model: str, cluster_id: int, tool_index: int, server_type: str, server_port: int, attacker_llm_model: str = None, defense_mechanism: str = "none", debug: bool = False):
    """Run a single experiment with the given parameters."""
    # Fixed parameters
    data_path = "data/clusters/bias_dataset_bfcl_format.jsonl"
    question_start = 0
    question_end = 100 if not debug else 5
    attack_mode = "cluster-attack"
    attack_modification = "both"
    max_attempts = 10 if not debug else 3
    
    # Translate model name if needed
    model_name = translate_model_name(model, server_type)
    
    # Generate output path
    output_path = generate_output_path(model, cluster_id, tool_index, server_type)
    
    # Construct command
    cmd = [
        "python", "main.py",
        "--attack-mode", attack_mode,
        "--model", model_name,
        "--data-path", data_path,
        "--question-start", str(question_start),
        "--question-end", str(question_end),
        "--attack-modification", attack_modification,
        "--cluster-id", str(cluster_id),
        "--target-tool-index", str(tool_index),
        "--defense-mechanism", str(defense_mechanism),
        "--max-attempts", str(max_attempts),
        "--output-path", output_path,
        "--server-type", server_type,
        "--server-port", str(server_port),
    ]

    if attacker_llm_model is not None:
        cmd.append("--attacker-llm-model")
        cmd.append(translate_model_name(attacker_llm_model, server_type))
    
    
    print(f"\nRunning experiment for tool index {tool_index}")
    print(f"Output path: {output_path}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment for tool index {tool_index}: {e}", file=sys.stderr)
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Run tool selection experiments for all tool indices")
    
    # Required arguments
    parser.add_argument("--model", required=True, help="Model name (HuggingFace format)")
    parser.add_argument("--cluster-id", type=int, required=True, help="Cluster ID (1-10)")
    
    # Optional arguments
    parser.add_argument("--server-type", default="ollama", choices=["vllm", "ollama"], help="Server type")
    parser.add_argument("--server-port", type=int, default=8000, help="Starting port number for server")
    parser.add_argument("--attacker-llm-model", default=None, help="Attacker model name (HuggingFace or Ollama format)")
    parser.add_argument("--defense-mechanism", default="none", choices=["none", "objective", "reword"], help="Defense mechanism to apply to the tool description.")
    parser.add_argument("--debug", action="store_true", help="Run a small number of trials for debugging")
    args = parser.parse_args()
    
    # Run experiments for all tool indices (0-4)
    for tool_index in range(5):
        success = run_experiment(
            model=args.model,
            cluster_id=args.cluster_id,
            tool_index=tool_index,
            server_type=args.server_type,
            server_port=args.server_port,
            attacker_llm_model=args.attacker_llm_model,
            defense_mechanism=args.defense_mechanism,
            debug=args.debug
        )
        if not success:
            print(f"Stopping after failure on tool index {tool_index}")
            sys.exit(1)

if __name__ == "__main__":
    main()

