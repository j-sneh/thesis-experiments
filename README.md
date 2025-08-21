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

I have also implemented defense in the pipeline, which we should run with:

```bash
python run_multiple_experiments.py --model qwen2.5:7b --cluster-id 1 10 --defense-mechanism objective --server-port 11434 --server-type ollama
```
```bash
python run_multiple_experiments.py --model llama3.1:8b --cluster-id 1 10 --defense-mechanism objective --server-port 11435 --server-type ollama
```
```bash
python run_multiple_experiments.py --model gpt-oss:20b --cluster-id 1 10 --defense-mechanism objective --server-port 11436 --server-type ollama
```
Run the baseline attack with the manual combination modification
```bash
python run_multiple_experiments.py --baseline-mode --modification combination  --model qwen2.5:7b --cluster-id 1 10 --server-type ollama --server-port 11434
```
```bash
python run_multiple_experiments.py --baseline-mode --modification combination  --model gpt-oss:20b --cluster-id 1 10 --server-type ollama --server-port 11435
```
```bash
python run_multiple_experiments.py --baseline-mode --modification combination  --model llama3.1:8b --cluster-id 1 10 --server-type ollama --server-port 11436
```


DEEPSEEK:
```bash
python run_multiple_experiments.py --server-type external --model "deepseek-reasoner" --baseline-mode --modification combination --max-workers 1 --api-key $DEEPSEEK_API_KEY --model-url $DEEPSEEK_URL --cluster-id 1 5
```

GEMINI:
```bash
python run_multiple_experiments.py --api-key $GOOGLE_API_KEY --server-type gemini --model gemini-2.5-flash --cluster-id 5 --tool-index 0 --debug
```