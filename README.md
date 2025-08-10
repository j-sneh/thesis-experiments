# Gaming Tool Descriptions

Research for thesis.

Runs experiments to determine if certain modifications to tool descriptions can cause bias in tool selection.

Also runs defenses and attack scenarios.


There are a lot of files in the package, and most of them can be ignored. There are a lot of data files and hacky things to get stuff to work.

The commands to get the experiments to run are 

```bash
./setup.sh
```

Have your shell set to the conda environment

```bash
conda activate attack
```

Run an experiment (Qwen attacking Qwen), on cluster 1, all tools, 10 "attempts"

Should work with any tool-calling model on ollama, like:
- qwen3:8b
- llama3.2:3b
- gpt-oss

Some special templating stuff needs to be done for phi-4, I can include that in a script down the line. Run these commands :D

```bash
python run_multiple_experiments.py --model qwen3:8b  --cluster-id 1
```

If that works, run on all 9 remaining clusters:

```bash
for i in {2..10}; do
  python run_multiple_experiments.py --model qwen3:8b --cluster-id $i
done
```