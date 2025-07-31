import json
from typing import List, Dict, Any

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
