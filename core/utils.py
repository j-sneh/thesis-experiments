import json
import subprocess
import os
import time
from typing import List, Dict, Any, Tuple
import io

OLLAMA_PATH = "pkg/bin/ollama"

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    result = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            result.append(json.loads(line))
    return result

def load_cluster_data(file_path: str, cluster_id: int, target_tool_index: int, question_start: int, question_end: int) -> Dict[str, Any]:
    """
    Load specific cluster data with selected tool and question range.
    
    Args:
        file_path: Path to the cluster data file
        cluster_id: Cluster ID (1-10)
        target_tool_index: Index of the tool to attack (0-4)
        question_start: Start index for questions (inclusive)
        question_end: End index for questions (exclusive)
        
    Returns:
        Dictionary containing the cluster data with selected tool and questions
    """
    clusters = load_data(file_path)
    
    # Find the target cluster
    target_cluster = None
    for cluster in clusters:
        if cluster['id'] == f"bias-{cluster_id}":
            target_cluster = cluster
            break
    
    if target_cluster is None:
        raise ValueError(f"Cluster bias-{cluster_id} not found")
    
    # Validate tool index
    if target_tool_index < 0 or target_tool_index >= len(target_cluster['function']):
        raise ValueError(f"Tool index {target_tool_index} out of range. Available tools: 0-{len(target_cluster['function'])-1}")
    
    # Validate question range
    if question_start < 0 or question_end > len(target_cluster['question']) or question_start >= question_end:
        raise ValueError(f"Invalid question range {question_start}-{question_end}. Available questions: 0-{len(target_cluster['question'])-1}")
    
    # Extract the target tool and questions
    target_tool = target_cluster['function'][target_tool_index]
    selected_questions = target_cluster['question'][question_start:question_end]
    
    return {
        'cluster_id': cluster_id,
        'target_tool_index': target_tool_index,
        'original_target_tool': target_tool,
        'all_tools': target_cluster['function'],
        'questions': selected_questions,
        'question_range': (question_start, question_end)
    }

def save_results(file_path: str, results: List[Dict[str, Any]]):
    """Save results to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

def spawn_server(model_name: str, port: int = 8000, server_type: str = "ollama", output_file_path: str = None) -> Tuple[str, subprocess.Popen, io.TextIOWrapper]:
    """Spawn a server for a given model.
    
    Args:
        model_name: Name of the model to serve
        port: Port number to serve on
        server_type: Type of server ("ollama" or "vllm")
        output_file_path: Path to the output file, used to create unique log file names
        
    Returns:
        Tuple containing:
        - The base URL of the server
        - The server process (to kill it later)
        - The log file handle
    """

    if server_type == "ollama":
        port = 11434

    print(f"Spawning {server_type} server on port {port}")
    # Create log file path with output identifier
    output_identifier = os.path.splitext(os.path.basename(output_file_path))[0] if output_file_path else "default"
    log_file = f"{server_type}_server_{port}_{output_identifier}.log"
    print(f"Logging server output to: {log_file}")
    
    # Open log file and redirect both stdout and stderr to it
    log_handle = open(log_file, 'w', encoding="utf-8")
    env = os.environ
    base_url = f"http://localhost:{port}/v1"
    
    if server_type == "vllm":
        commands = [
            "vllm",
            "serve",
            model_name,
            "--port",
            str(port),
            "--enable-auto-tool-choice"
        ]

        if model_name.startswith("Qwen/Qwen3") or model_name.startswith("NousResearch/Hermes"):
            commands.extend(['--tool-call-parser', 'hermes'])
            commands.extend(['--reasoning-parser', 'deepseek_r1'])
        elif model_name.startswith("microsoft/Phi-4"):
            commands.extend([
                "--tool-call-parser",
                "phi4_mini_json",
                "--trust-remote-code",
                "--chat-template",
                "templates/phi4-basic.jinja"
            ])
        elif model_name.startswith("meta-llama/Llama-3.2"):
            commands.extend([
                "--tool-call-parser",
                "llama3_json"
            ])
        else:
            raise ValueError(f"Unsupported model for vllm: {model_name}")
            
    else:  # ollama
        env["OLLAMA_DEBUG"] = "2"
        env["OLLAMA_NUM_PARALLEL"] = "4"

        commands = [
            OLLAMA_PATH,
            "serve"
        ]

    print(f"Running commands: {' '.join(commands)}")
    
    # Start the process
    process = subprocess.Popen(commands, env=env, stdout=log_handle, stderr=log_handle)
    
    # Give the process a moment to start and check if it's still running
    # time.sleep(7)
    if process.poll() is not None:
        # Process has terminated
        log_handle.flush()
        with open(log_file, 'r') as f:
            error_output = f.read()
        raise RuntimeError(f"Server failed to start. Process exited with code {process.returncode}. Output:\n{error_output}")
    
    return base_url, process, log_handle

def parse_json_inside_string(s):
    # parses first json inside a string which includes other text
    s = s[next(idx for idx, c in enumerate(s) if c in "{["):]
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        return json.loads(s[:e.pos])