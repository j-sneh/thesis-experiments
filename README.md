# Gaming Tool Descriptions

Research for thesis.

Runs experiments to determine if certain modifications to tool descriptions can cause bias in tool selection.

Also runs defenses and attack scenarios.


There are a lot of files in the package, and most of them can be ignored. There are a lot of data files and hacky things to get stuff to work.

The commands to get the experiments to run are 

```bash
./setup.sh
```

This script will also pull the models from ollama. Sometimes there are issues connecting to the remote server to pull the model.
If there are any problems, then you can manually pull `ollama pull <model name>` from the script.

Have your shell set to the conda environment

```bash
conda activate attack
```

I have updated so that no looping in bash is needed to run model experiments, it can all be done by the script `run_multiple_experiments.py`. I made updates (to temperature + random order of tooling), so the experiments should be run again. 

# Evaluations for AML - 

# AML

Run the attack loop, but with --defense-mechanism objective will have the attacker train on the defender
```bash
python run_multiple_experiments.py --model qwen2.5:7b --cluster-id 1 10 --server-type ollama --defense-mechanism objective
```
```bash
python run_multiple_experiments.py --model gpt-oss:20b --cluster-id 1 10 --server-type ollama --defense-mechanism objective
```
```bash
python run_multiple_experiments.py --model llama3.1:8b --cluster-id 1 10 --server-type ollama --defense-mechanism objective
```



# New Evaluations for AML - Transferability Experiments

## Evaluating with llama3.1 (4 commands - against all other models)
```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model llama3.1:8b \
    --eval-mode \
    --eval-dir "data/replay/deepseek-chat/attack" \
    --cluster-id 1 10
```

```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model llama3.1:8b \
    --eval-mode \
    --eval-dir "data/replay/gemini2.5-flash-lite/attack" \
    --cluster-id 1 10
```

```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model llama3.1:8b \
    --eval-mode \
    --eval-dir "data/replay/gpt-oss/attack" \
    --cluster-id 1 10
```

```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model llama3.1:8b \
    --eval-mode \
    --eval-dir "data/replay/qwen2.5/attack" \
    --cluster-id 1 10
```

## Evaluating with qwen2.5 (4 commands - against all other models, except gpt-4o)
```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model qwen2.5:7b \
    --eval-mode \
    --eval-dir "data/replay/deepseek-chat/attack" \
    --cluster-id 1 10
```

```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model qwen2.5:7b \
    --eval-mode \
    --eval-dir "data/replay/gemini2.5-flash-lite/attack" \
    --cluster-id 1 10
```



```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model qwen2.5:7b \
    --eval-mode \
    --eval-dir "data/replay/gpt-oss/attack" \
    --cluster-id 1 10
```

```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model qwen2.5:7b \
    --eval-mode \
    --eval-dir "data/replay/llama3.1/attack" \
    --cluster-id 1 10
```

## Evaluating with gpt-oss (4 commands - against all other models except 4o)
```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model gpt-oss:20b \
    --eval-mode \
    --eval-dir "data/replay/deepseek-chat/attack" \
    --cluster-id 1 10
```

```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model gpt-oss:20b \
    --eval-mode \
    --eval-dir "data/replay/gemini2.5-flash-lite/attack" \
    --cluster-id 1 10
```


```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model gpt-oss:20b \
    --eval-mode \
    --eval-dir "data/replay/llama3.1/attack" \
    --cluster-id 1 10
```

```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model gpt-oss:20b \
    --eval-mode \
    --eval-dir "data/replay/qwen2.5/attack" \
    --cluster-id 1 10
```

# Defense Evaluations
# Run --eval-mode experiments with --defense-mechanism objective on same model data

## Defense evaluation with llama3.1 (against its own attack data)
```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model llama3.1:8b \
    --eval-mode \
    --eval-dir "data/replay/llama3.1/attack" \
    --cluster-id 1 10 \
    --defense-mechanism objective
```

## Defense evaluation with qwen2.5 (against its own attack data)
```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model qwen2.5:7b \
    --eval-mode \
    --eval-dir "data/replay/qwen2.5/attack" \
    --cluster-id 1 10 \
    --defense-mechanism objective
```

## Defense evaluation with gpt-oss (against its own attack data)
```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model gpt-oss:20b \
    --eval-mode \
    --eval-dir "data/replay/gpt-oss/attack" \
    --cluster-id 1 10 \
    --defense-mechanism objective
```
# Azure ChatGPT

Ensure these environment variables are set:
```bash
OX_AZURE_API_VERSION=<versionxyz>
OX_AZURE_ENDPOINT=<url>
```

If it works, run the full base attack. I've been running with `max-workers = 1`, since I was getting rate limited by other APIs
```bash
python run_multiple_experiments.py --server-type azure --model gpt-4o --cluster-id 1 10 --max-workers 1
```
and run baseline mode, as well
```bash
python run_multiple_experiments.py --baseline-mode --modification combination  --server-type azure --model gpt-4o --cluster-id 1 10 --max-workers 1 && \
python run_multiple_experiments.py --baseline-mode --modification combination  --server-type azure --model gpt-4o --cluster-id 1 10 --defense-mechanism objective --max-workers 1
```


# FOR LATER: ONCE GPT-4o Works
## Evaluating with gpt-4o (requires complete gpt-4o attack data first)
Note: These commands should only be run once the gpt-4o attack data is fully generated.

### These 
```bash
python run_multiple_experiments.py \
    --server-type azure \
    --model gpt-4o \
    --eval-mode \
    --eval-dir "data/replay/deepseek-chat/attack" \
    --cluster-id 1 10 \
    --max-workers 1
```

```bash
python run_multiple_experiments.py \
    --server-type azure \
    --model gpt-4o \
    --eval-mode \
    --eval-dir "data/replay/gemini2.5-flash-lite/attack" \
    --cluster-id 1 10 \
    --max-workers 1
```

```bash
python run_multiple_experiments.py \
    --server-type azure \
    --model gpt-4o \
    --eval-mode \
    --eval-dir "data/replay/gpt-oss/attack" \
    --cluster-id 1 10 \
    --max-workers 1
```

```bash
python run_multiple_experiments.py \
    --server-type azure \
    --model gpt-4o \
    --eval-mode \
    --eval-dir "data/replay/llama3.1/attack" \
    --cluster-id 1 10 \
    --max-workers 1
```

```bash
python run_multiple_experiments.py \
    --server-type azure \
    --model gpt-4o \
    --eval-mode \
    --eval-dir "data/replay/qwen2.5/attack" \
    --cluster-id 1 10 \
    --max-workers 1
```

## Defense evaluation with gpt-4o (against its own attack data, once available)
```bash
python run_multiple_experiments.py \
    --server-type azure \
    --model gpt-4o \
    --eval-mode \
    --eval-dir "data/replay/gpt-4o/attack" \
    --cluster-id 1 10 \
    --defense-mechanism objective \
    --max-workers 1
```

## Commands to evaluate against gpt-4o data (once available)
Note: These should be run after gpt-4o attack data is generated.** These are run on AML**

```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model llama3.1:8b \
    --eval-mode \
    --eval-dir "data/replay/gpt-4o/attack" \
    --cluster-id 1 10
```

```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model qwen2.5:7b \
    --eval-mode \
    --eval-dir "data/replay/gpt-4o/attack" \
    --cluster-id 1 10
```

```bash
python run_multiple_experiments.py \
    --server-type ollama \
    --model gpt-oss:20b \
    --eval-mode \
    --eval-dir "data/replay/gpt-4o/attack" \
    --cluster-id 1 10
```
















# AML

We should run on all of these models:
- qwen2.5:7b
- llama3.1:8b
- gpt-oss:20b

Each of the following commands should be run as different jobs.

Run the default attacks (parallelization is on by default):
```bash
python run_multiple_experiments.py --model qwen2.5:7b --cluster-id 1 10 --server-port 11434 --server-type ollama
```
```bash
python run_multiple_experiments.py --model gpt-oss:20b --cluster-id 1 10 --server-port 11436 --server-type ollama
```
```bash
python run_multiple_experiments.py --model llama3.1:8b --cluster-id 1 10 --server-port 11435 --server-type ollama
```

Run the baseline attack with the manual combination modification AND then again, with a defense added. These are relatively quick and can be chained together on the same GPU
```bash
python run_multiple_experiments.py --baseline-mode --modification combination  --model qwen2.5:7b --cluster-id 1 10 --server-type ollama --server-port 11434 && \
python run_multiple_experiments.py --baseline-mode --modification combination  --model qwen2.5:7b --cluster-id 1 10 --server-type ollama --server-port 11434 --defense-mechanism objective
```
```bash
python run_multiple_experiments.py --baseline-mode --modification combination  --model gpt-oss:20b --cluster-id 1 10 --server-type ollama --server-port 11435 && \
python run_multiple_experiments.py --baseline-mode --modification combination  --model gpt-oss:20b --cluster-id 1 10 --server-type ollama --server-port 11435 --defense-mechanism objective
```
```bash
python run_multiple_experiments.py --baseline-mode --modification combination  --model llama3.1:8b --cluster-id 1 10 --server-type ollama --server-port 11436 && \
python run_multiple_experiments.py --baseline-mode --modification combination  --model llama3.1:8b --cluster-id 1 10 --server-type ollama --server-port 11436 --defense-mechanism objective
```

# For Future Reference
### Don't use these now

Replay Attack
```bash
python run_multiple_experiments.py \
    --model "deepseek-chat" \
    --cluster-id 1 \
    --tool-index 0 \
    --eval-mode \
    --eval-dir "final-results/gemini/attack" \
    --server-type gemini \
    --api-key $GOOGLE_API_KEY \
    --debug
```


DEEPSEEK example:
```bash
python run_multiple_experiments.py --server-type external --model "deepseek-chat" --baseline-mode --modification combination --max-workers 1 --api-key $DEEPSEEK_API_KEY --model-url $DEEPSEEK_URL --cluster-id 1 5
```

GEMINI:
```bash
python run_multiple_experiments.py --api-key $GOOGLE_API_KEY --server-type gemini --model gemini-2.5-flash --cluster-id 5 --tool-index 0 --debug
```

REPLAY EXAMPLE:
```
python run_multiple_experiments.py \
    --api-key $GOOGLE_API_KEY \
    --server-type gemini \
    --model gemini-2.5-flash \
    --eval-mode \
    --eval-dir "data/replay/deepseek-chat/attack" \
    --cluster-id 1 \
    --max-workers 1
```

ABLATIONS:
```bash
# Using run_multiple_experiments.py
python run_multiple_experiments.py --model "llama3.2:3b" --cluster-id 1 --num-feedback-tools 3 --num-feedback-queries 5

# Using main.py  
python main.py --attack-mode cluster-attack --cluster-id 1 --target-tool-index 0 --question-start 0 --question-end 100 --num-feedback-tools 0 --num-feedback-queries 15
```

python run_multiple_experiments.py \
    --server-type external \
    --model "deepseek-chat" \
    --num-feedback-tools 0 \
    --num-feedback-queries 0 \
    --api-key $DEEPSEEK_API_KEY --model-url $DEEPSEEK_URL \
    --cluster-id 1 10 --max-workers 10



python run_multiple_experiments.py --server-type external --model "deepseek-chat" --api-key $DEEPSEEK_API_KEY --model-url $DEEPSEEK_URL --cluster-id 2 3


## Defense evaluation with deepseek-chat (against its own attack data)
```bash
python run_multiple_experiments.py \
    --server-type external \
    --model "deepseek-chat" \
    --eval-mode \
    --eval-dir "data/replay/deepseek-chat/attack" \
    --cluster-id 1 10 \
    --defense-mechanism objective \
    --api-key $DEEPSEEK_API_KEY \
    --model-url $DEEPSEEK_URL \
    --max-workers 1
```

## Defense evaluation with gemini2.5-flash-lite (against its own attack data)
```bash
python run_multiple_experiments.py \
    --server-type gemini \
    --model gemini-2.5-flash-lite \
    --eval-mode \
    --eval-dir "data/replay/gemini2.5-flash-lite/attack" \
    --cluster-id 1 10 \
    --defense-mechanism objective \
    --api-key $GOOGLE_API_KEY \
    --max-workers 1
```
