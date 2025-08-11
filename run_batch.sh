for i in {2..10}; do
  python run_multiple_experiments.py --model qwen3:8b --cluster-id $i
done