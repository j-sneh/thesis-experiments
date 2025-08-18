#!/usr/bin/env python3
import subprocess
import sys
import os
import argparse
from pathlib import Path
import re
import time
import json
from typing import List, Dict, Any
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

def generate_base_dir(model: str, server_type: str, mode: str) -> Path:
    """Generate a base directory for the experiment."""
    model_short_name = sanitize_model_name(model.lower())
    timestamp = int(time.time())

    bd =  Path(server_type) / mode / model_short_name /  str(timestamp)
    return bd

def generate_output_path(model: str, cluster_id: int, tool_index: int, server_type: str, base_dir: Path) -> str:
    """Generate a structured output path based on experiment parameters."""    
    model_short_name = sanitize_model_name(model.lower())
    # Format: {server_type}/cluster-{id}-tool-{index}-{model}-q0-100
    filename = f"cluster-{cluster_id}-tool-{tool_index}-{model_short_name}-q0-100"
    
    return str(base_dir / filename)

def find_improvement_file(eval_dir: str, cluster_id: int, tool_index: int) -> str:
    """
    Find improvement history file for a specific cluster and tool combination.
    
    Returns:
        Path to the improvement history file, or None if not found
    """
    
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")
    
    # Pattern to match improvement history files for this specific cluster/tool
    pattern = re.compile(rf'cluster-{cluster_id}-tool-{tool_index}-[^-]+-q\d+-\d+-improvement_history\.jsonl$')
    
    for file_path in eval_path.rglob("*-improvement_history.jsonl"):
        if pattern.search(file_path.name):
            return str(file_path)
    
    return None

def run_experiment(model: str, cluster_id: int, tool_index: int, server_type: str, server_port: int, url: str, attacker_llm_model: str = None, defender_llm_model: str = None, defense_mechanism: str = "none", debug: bool = False, base_dir: Path = None, seed: int = 42, eval_mode: bool = False, eval_config: str = None, eval_attempt: int = None, modification: str = None):
    """Run a single experiment with the given parameters."""
    # Fixed parameters
    data_path = "data/clusters/bias_dataset_bfcl_format.jsonl"
    question_start = 0
    question_end = 100 if not debug else 5
    attack_mode = "cluster-attack"
    attack_modification = "both"
    max_attempts = 1 if eval_mode else (10 if not debug else 3)
    
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
        "--seed", str(seed),
    ]

    # Add eval mode parameters if specified
    if eval_mode:
        cmd.append("--eval-mode")
        if eval_config is not None:
            cmd.extend(["--eval-config", eval_config])
        if eval_attempt is not None:
            cmd.extend(["--eval-attempt", str(eval_attempt)])
    
    # Add modification if specified
    if modification is not None:
        cmd.extend(["--modification", modification])

    if attacker_llm_model is not None:
        cmd.append("--attacker-llm-model")
        cmd.append(translate_model_name(attacker_llm_model, server_type))
    
    if defender_llm_model is not None:
        cmd.append("--defender-llm-model")
        cmd.append(translate_model_name(defender_llm_model, server_type))
    
    # if we are spawning a server, we need to pass the url to the experiment
    # this is in the case of ollama (or vllm) and not hflocal
    if url is not None:
        cmd.append("--model-url")
        cmd.append(url)
        cmd.append("--attacker-url")
        cmd.append(url)
        cmd.append("--defender-url")
        cmd.append(url)
    
    print(f"\nRunning experiment for tool index {tool_index}")
    print(f"Output path: {output_path}")
    print(cmd)
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
    parser.add_argument("--server-type", default="hflocal", choices=["vllm", "ollama", "hflocal"], help="Server type")
    parser.add_argument("--server-port", type=int, default=11434, help="Starting port number for server")
    parser.add_argument("--attacker-llm-model", default=None, help="Attacker model name (HuggingFace or Ollama format)")
    parser.add_argument("--defender-llm-model", default=None, help="Defender model name (HuggingFace or Ollama format)")
    parser.add_argument("--defense-mechanism", default="none", choices=["none", "objective", "reword"], help="Defense mechanism to apply to the tool description.")
    parser.add_argument("--debug", action="store_true", help="Run a small number of trials for debugging")
    parser.add_argument("--tool-index", type=int, nargs="+", help="Tool index (0-4) can either be a single number 0-4 for a single trial, or a range (2 numbers). Will be all 4 tools if not specified.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results (default: 42).")
    
    # Eval mode arguments (mutually exclusive with baseline mode)
    parser.add_argument("--eval-mode", action="store_true", help="Enable eval mode: replay attacks from improvement history files in a directory")
    parser.add_argument("--eval-dir", type=str, help="Directory containing *-improvement_history.jsonl files to replay")
    parser.add_argument("--eval-attempt", type=int, help="Specific attempt number to replay (if not specified, uses best performing)")
    
    # Baseline mode arguments (mutually exclusive with eval mode)
    parser.add_argument("--baseline-mode", action="store_true", help="Enable baseline mode: test modifications on original tools")
    parser.add_argument("--modification", default="none", type=str, choices=["assertive_cue", "active_maintenance", "none", "noop"], help="Modification to apply in baseline mode")
    
    args = parser.parse_args()

    # Validate mutually exclusive modes
    if args.eval_mode and args.baseline_mode:
        parser.error("--eval-mode and --baseline-mode are mutually exclusive")
    
    if args.eval_mode and not args.eval_dir:
        parser.error("--eval-mode requires --eval-dir")
        
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

    server_type = args.server_type
    model = translate_model_name(args.model, server_type)
    attacker_llm_model = translate_model_name(args.attacker_llm_model, server_type) if args.attacker_llm_model else None
    defender_llm_model = translate_model_name(args.defender_llm_model, server_type) if args.defender_llm_model else None

    # Determine base directory based on mode
    if args.eval_mode:
        base_dir = generate_base_dir(model, server_type, "eval")
    elif args.baseline_mode:
        base_dir = generate_base_dir(model, server_type, "baseline")
    else:
        base_dir = generate_base_dir(model, server_type, "attack")

    model_processes = {}
    if server_type == "ollama":
    # Spawn the server for the main model
        url, process, log_handle = spawn_server(model, args.server_port, server_type, base_dir / "PLACEHOLDER_FILE_TO_REMOVE")
        print(f"Spawned server for {model} at {url}")

        # Wait for the server to start
        client = OpenAIClient(model, url)
        client.wait_for_server_to_start()
        print(f"Server for {model} started")
        model_processes["ollama"] = (url, process, log_handle)

    
    try:
        # Record command run
        base_dir.mkdir(exist_ok=True, parents=True)
        with open(base_dir / "args.json", "w") as f:
            json.dump(vars(args), f)
        
        # Run experiments for all cluster/tool combinations
        for cluster_id in cluster_ids:
            for tool_index in tool_indices:
                if args.eval_mode:
                    # Find the corresponding improvement history file
                    eval_config = find_improvement_file(args.eval_dir, cluster_id, tool_index)
                    if eval_config is None:
                        print(f"No improvement history file found for cluster {cluster_id}, tool {tool_index}")
                        continue
                    
                    print(f"Running eval experiment for cluster {cluster_id}, tool {tool_index}")
                    print(f"Eval config: {eval_config}")
                    
                    success = run_experiment(
                        model=model,
                        cluster_id=cluster_id,
                        tool_index=tool_index,
                        server_type=server_type,
                        server_port=args.server_port,
                        attacker_llm_model=None,  # Not used in eval mode
                        defender_llm_model=defender_llm_model,
                        defense_mechanism=args.defense_mechanism,
                        debug=args.debug,
                        base_dir=base_dir,
                        url=url if server_type == "ollama" else None,
                        seed=args.seed,
                        eval_mode=True,
                        eval_config=eval_config,
                        eval_attempt=args.eval_attempt
                    )
                elif args.baseline_mode:
                    print(f"Running baseline experiment for cluster {cluster_id}, tool {tool_index}")
                    print(f"Modification: {args.modification}")
                    
                    success = run_experiment(
                        model=model,
                        cluster_id=cluster_id,
                        tool_index=tool_index,
                        server_type=server_type,
                        server_port=args.server_port,
                        attacker_llm_model=None,  # Not used in baseline mode
                        defender_llm_model=defender_llm_model,
                        defense_mechanism=args.defense_mechanism,
                        debug=args.debug,
                        base_dir=base_dir,
                        url=url if server_type == "ollama" else None,
                        seed=args.seed,
                        eval_mode=True,  # Use eval mode for single attempt
                        modification=args.modification
                    )
                else:  # standard mode
                    success = run_experiment(
                        model=model,
                        cluster_id=cluster_id,
                        tool_index=tool_index,
                        server_type=server_type,
                        server_port=args.server_port,
                        attacker_llm_model=attacker_llm_model,
                        defender_llm_model=defender_llm_model,
                        defense_mechanism=args.defense_mechanism,
                        debug=args.debug,
                        base_dir=base_dir,
                        url=url if server_type == "ollama" else None,
                        seed=args.seed
                    )
                
                if not success:
                    print(f"Stopping after failure on cluster {cluster_id}, tool {tool_index}")
                    sys.exit(1)
    finally:
        # Kill the server
        for _, process, log_handle in model_processes.values():
            process.terminate()
            log_handle.close()
        print(f"Killed server")

if __name__ == "__main__":
    main()

