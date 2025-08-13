#!/bin/bash
set -euo pipefail

python run_multiple_experiments.py --model qwen3:8b --cluster-id 1 10 --defense-mechanism objective --server-port 11434

python run_multiple_experiments.py --model llama3.2:3b --cluster-id 1 10 --defense-mechanism objective --server-port 11435

python run_multiple_experiments.py --model gpt-oss:20b --cluster-id 1 10 --defense-mechanism objective --server-port 11436

python run_multiple_experiments.py --model phi4-mini-tool-prompt:3.8b-fp16 --attacker-llm-model phi4-mini:3.8b-fp16 --defender-llm-model phi4-mini:3.8b-fp16 --cluster-id 1 10 --defense-mechanism objective --server-port 11437