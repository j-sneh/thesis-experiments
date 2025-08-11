for i in {2..10}; do
  python run_multiple_experiments.py --model llama3.2:3b --cluster-id $i
done