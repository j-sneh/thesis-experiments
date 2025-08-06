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

Run an experiment (Qwen attacking Qwen), on cluster 1, all tools, 10 iterations

Currently accepted models:
Qwen/Qwen3-8B
```bash
CUDA_VISIBLE_DEVICES=0,7 python run_multiple_experiments.py --model Qwen/Qwen3-8B --cluster-id 5
```
