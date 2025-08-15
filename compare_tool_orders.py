#!/usr/bin/env python3
"""
Script to compare tool orders between two experiment directories.

This script:
1. Reads JSONL files from two directories
2. Filters by attempt field == 0  
3. Matches entries by cluster-<n>-q<n> format
4. Extracts tool names from tools_provided[]["function"]["name"]
5. Compares if tools are in the same order
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl_data(file_path: str) -> List[dict]:
    """Load data from a JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line in {file_path}: {e}")
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    return data


def extract_tool_names(entry: dict) -> List[str]:
    """Extract tool names from tools_provided field."""
    tool_names = []
    tools_provided = entry.get('tools_provided', [])
    
    for tool in tools_provided:
        try:
            name = tool['function']['name']
            tool_names.append(name)
        except (KeyError, TypeError):
            print(f"Warning: Could not extract tool name from entry {entry.get('id', 'unknown')}")
    
    return tool_names


def filter_attempt_zero(data: List[dict]) -> List[dict]:
    """Filter entries where attempt == 0."""
    return [entry for entry in data if entry.get('attempt') == 0]


def group_by_cluster_q(data: List[dict]) -> Dict[str, dict]:
    """Group entries by cluster-<n>-q<n> pattern."""
    grouped = {}
    
    for entry in data:
        entry_id = entry.get('id', '')
        
        # Extract cluster-n-qn pattern
        if entry_id.startswith('cluster-') and '-q' in entry_id:
            # Extract the pattern: cluster-3-q4 -> cluster-3-q4
            key = entry_id
            grouped[key] = entry
        
    return grouped


def compare_tool_orders(dir1: str, dir2: str) -> None:
    """Compare tool orders between two directories."""
    
    # Construct file paths
    dir1_path = Path(dir1)
    dir2_path = Path(dir2)
    
    # Find the JSONL files (not improvement history)
    jsonl_files1 = [f for f in dir1_path.glob("*.jsonl") if "improvement_history" not in f.name]
    jsonl_files2 = [f for f in dir2_path.glob("*.jsonl") if "improvement_history" not in f.name]
    
    if not jsonl_files1:
        print(f"Error: No JSONL files found in {dir1}")
        return
    if not jsonl_files2:
        print(f"Error: No JSONL files found in {dir2}")
        return
        
    # Use the first JSONL file found in each directory
    file1 = jsonl_files1[0]
    file2 = jsonl_files2[0]
    
    print(f"Comparing:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    print()
    
    # Load data
    data1 = load_jsonl_data(str(file1))
    data2 = load_jsonl_data(str(file2))
    
    if not data1 or not data2:
        print("Error: Could not load data from one or both files")
        return
    
    # Filter by attempt == 0
    data1_filtered = filter_attempt_zero(data1)
    data2_filtered = filter_attempt_zero(data2)
    
    print(f"Loaded {len(data1_filtered)} attempt-0 entries from file 1")
    print(f"Loaded {len(data2_filtered)} attempt-0 entries from file 2")
    print()
    
    # Group by cluster-q pattern
    grouped1 = group_by_cluster_q(data1_filtered)
    grouped2 = group_by_cluster_q(data2_filtered)
    
    # Find common keys
    common_keys = set(grouped1.keys()) & set(grouped2.keys())
    
    if not common_keys:
        print("No matching cluster-q entries found between the two files")
        return
    
    print(f"Found {len(common_keys)} matching entries to compare")
    print()
    
    # Compare tool orders
    matches = 0
    mismatches = 0
    
    for key in sorted(common_keys):
        entry1 = grouped1[key]
        entry2 = grouped2[key]
        
        tools1 = extract_tool_names(entry1)
        tools2 = extract_tool_names(entry2)

        # Pretty print the tool names
        print(f"  {key}:")
        print(f"    File 1 tools: {tools1}")
        print(f"    File 2 tools: {tools2}")
        print()
        
        if tools1 == tools2:
            print(f"✓ {key}: Tool orders MATCH ({len(tools1)} tools)")
            matches += 1
        else:
            print(f"✗ {key}: Tool orders DIFFER")
            mismatches += 1
        print()
    
    # Summary
    print("=" * 50)
    print("SUMMARY:")
    print(f"  Matches: {matches}")
    print(f"  Mismatches: {mismatches}")
    print(f"  Total compared: {len(common_keys)}")
    
    if mismatches == 0:
        print("🎉 All tool orders match!")
    else:
        print(f"⚠️  {mismatches} entries have different tool orders")


def main():
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python compare_tool_orders.py <dir1> <dir2>")
        print()
        print("Example:")
        print("  python compare_tool_orders.py ollama/llama3.2_3b/1755270224 ollama/llama3.2_3b/1755270289")
        sys.exit(1)
    
    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    
    # Check if directories exist
    if not os.path.isdir(dir1):
        print(f"Error: Directory does not exist: {dir1}")
        sys.exit(1)
    if not os.path.isdir(dir2):
        print(f"Error: Directory does not exist: {dir2}")
        sys.exit(1)
    
    compare_tool_orders(dir1, dir2)


if __name__ == "__main__":
    main()
