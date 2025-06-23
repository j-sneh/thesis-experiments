import json
from typing import List, Dict, Any

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    result = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            result.append(json.loads(line))
    return result

def save_results(file_path: str, results: List[Dict[str, Any]]):
    """Save results to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
