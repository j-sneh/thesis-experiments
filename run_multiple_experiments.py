#!/usr/bin/env python3
import subprocess
import sys
import os
import argparse
from pathlib import Path
import re
import time
import json
from core.llm_clients import OpenAIClient
from core.utils import spawn_server

def sanitize_model_name(model_name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '_', model_name)

def translate_model_name(model_name: str, server_type: str) -> str:
    """Translate model names between HuggingFace and Ollama formats using regex."""
    # Remove colons 
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

def generate_base_dir(model: str, server_type: str, cluster_id: int, defense_mechanism: str) -> Path:
    """Generate a base directory for the experiment."""
    model_short_name = sanitize_model_name(model.lower())
    timestamp = int(time.time())

    bd =  Path("vllm" if server_type == "vllm" else "ollama") / model_short_name /  str(timestamp)
    return bd

def generate_output_path(model: str, cluster_id: int, tool_index: int, server_type: str, base_dir: Path) -> str:
    """Generate a structured output path based on experiment parameters."""    
    model_short_name = sanitize_model_name(model.lower())
    # Format: {server_type}/cluster-{id}-tool-{index}-{model}-q0-100
    filename = f"cluster-{cluster_id}-tool-{tool_index}-{model_short_name}-q0-100"
    
    return str(base_dir / filename)

def run_experiment(model: str, cluster_id: int, tool_index: int, server_type: str, server_port: int, url: str, attacker_llm_model: str = None, defense_mechanism: str = "none", debug: bool = False, base_dir: Path = None):
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
    output_path = generate_output_path(model, cluster_id, tool_index, server_type, base_dir)
    
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
        "--model-url", url,
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
    parser.add_argument("--cluster-id", type=int, nargs="+", required=True, help="Cluster ID (1-10) can either be a single number 1-10 for a single trial, or a range (2 numbers)")
    
    # Optional arguments
    # parser.add_argument("--server-type", default="ollama", choices=["vllm", "ollama"], help="Server type")
    parser.add_argument("--server-port", type=int, default=11434, help="Starting port number for server")
    parser.add_argument("--attacker-llm-model", default=None, help="Attacker model name (HuggingFace or Ollama format)")
    parser.add_argument("--defense-mechanism", default="none", choices=["none", "objective", "reword"], help="Defense mechanism to apply to the tool description.")
    parser.add_argument("--debug", action="store_true", help="Run a small number of trials for debugging")
    parser.add_argument("--tool-index", type=int, nargs="+", help="Tool index (0-4) can either be a single number 0-4 for a single trial, or a range (2 numbers). Will be all 4 tools if not specified.")
    args = parser.parse_args()

    cluster_ids = None
    if len(args.cluster_id) == 1:
        cluster_ids = range(args.cluster_id[0], args.cluster_id[0] + 1)
    elif len(args.cluster_id) == 2:
        min_val, max_val = args.cluster_id
        cluster_ids = range(min_val, max_val + 1)
        if min_val > max_val:
            raise ValueError("Min value must be less than max value")
    else:
        raise ValueError("Cluster ID must be a single number or a range of 2 numbers")

    tool_indices = None
    if args.tool_index is None:
        tool_indices = range(5)
    elif len(args.tool_index) == 1:
        tool_indices = range(args.tool_index[0], args.tool_index[0] + 1)
    elif len(args.tool_index) == 2:
        min_val, max_val = args.tool_index
        tool_indices = range(min_val, max_val + 1)
        if min_val > max_val:
            raise ValueError("Min value must be less than max value")
    else:
        raise ValueError("Tool index must be a single number or a range of 2 numbers")

    # Spawn the server for the main model
    base_dir = generate_base_dir(args.model, "ollama", args.cluster_id, args.defense_mechanism)
    url, process, log_handle = spawn_server(args.model, args.server_port, "ollama", base_dir)
    print(f"Spawned server for {args.model} at {url}")

    # Wait for the server to start
    client = OpenAIClient(args.model, url)
    client.wait_for_server_to_start()
    print(f"Server for {args.model} started")

    
    try:
    
        # Record command run
        base_dir.mkdir(exist_ok=True, parents=True)
        with open(base_dir / "args.json", "w") as f:
            json.dump(vars(args), f)
        
        # Run experiments for all tool indices (0-4)
        for cluster_id in cluster_ids:
            for tool_index in tool_indices:
                success = run_experiment(
                    model=args.model,
                    cluster_id=cluster_id,
                    tool_index=tool_index,
                    server_type="ollama",
                    server_port=args.server_port,
                    url=url,
                    attacker_llm_model=args.attacker_llm_model,
                    defense_mechanism=args.defense_mechanism,
                    debug=args.debug,
                    base_dir=base_dir
                )
                if not success:
                    print(f"Stopping after failure on tool index {tool_index}")
                    sys.exit(1)
    finally:
        # Kill the server
        process.terminate()
        log_handle.close()
        print(f"Killed server")

if __name__ == "__main__":
    main()

