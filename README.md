# Gaming Tool Descriptions

Research for thesis.

Runs experiments to determine if certain modifications to tool descriptions can cause bias in tool selection.

Also runs defenses

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
- `--output-path`: Path to save the results (default: results.jsonl)
- `--modification`: Modification to apply to the tool description (default: assertive_cue)
- `--defense-mechanism`: Defense mechanism to apply to the tool description (default: objective)


