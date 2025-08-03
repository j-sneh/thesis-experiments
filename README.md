# Gaming Tool Descriptions

Research for thesis.

Runs experiments to determine if certain modifications to tool descriptions can cause bias in tool selection.

Also runs defenses and attack scenarios.

# Re-run Experiments

Ensure you have `uv` installed

Download dependencies

```
uv sync
```

Run
```
uv run main.py
```

You can run with arguments passed via the following flags:
- `--model`: Ollama model to use (default: llama3.2)
- `--data-path`: Path to the data file (default: data/BFCL_v3_simple.jsonl)
- `--output-path`: Path to save the results (default: results)
- `--modification`: Modification to apply to the tool description (default: assertive_cue)
  - Choices: "assertive_cue", "active_maintenance", "none", "noop"
- `--defense-mechanism`: Defense mechanism to apply to the tool description (default: objective)
  - Choices: "objective", "reword", "none", "noop"
- `--attack-mode`: Attack mode to use (default: no-attack)
  - Choices: "attack", "suffix-attack", "no-attack"
- `--attacker-llm-model`:  Model to use for attacker mode (defaults to main model if not specified)
- `--defender-llm-model`: Model to use for defender mode (defaults to main model if not specified)
- `--max-attempts`: Maximum number of attack attempts (default: 5)
- `--dataset-size`: Number of items to use from the dataset (default: use all items)
- `--engine`: Inference engine to use (default: ollama)
  - Choices: "vllm", "ollama"

## Example Usage

Basic experiment with default settings:
```bash
uv run main.py
```

Experiment with custom model and modification:
```bash
uv run main.py --model qwen2.5:7b --modification active_maintenance
```

Attack scenario with different models for attacker and defender:
```bash
uv run main.py --attack-mode attack --attacker-llm-model llama3.2 --defender-llm-model qwen2.5 --defense-mechanism reword
```

Suffix attack with limited dataset size:
```bash
uv run main.py --attack-mode suffix-attack --dataset-size 50 --max-attempts 3
```

Using VLLM engine instead of Ollama:
```bash
uv run main.py --engine vllm --model qwen2.5
```


Running this command:
```
CUDA_VISIBLE_DEVICES=6,7 python main.py --attack-mode cluster-attack --model qwen3:8b --data-path data/clusters/bias_dataset_bfcl_format.jsonl --question-start 0 --question-end 100 --attack-modification both --cluster-id 1 --target-tool-index 1 --defense-mechanism none --max-attempts 10 --output-path one-index-qwen-ollama --server-type ollama
```