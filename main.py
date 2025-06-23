import argparse
from core.experiment import run_experiment

def main():
    parser = argparse.ArgumentParser(description="Run LLM tool selection experiment.")
    parser.add_argument("--model", type=str, default="llama3.2", help="Ollama model to use.")
    parser.add_argument("--data-path", type=str, default="data/BFCL_v3_simple.jsonl", help="Path to the data file.")
    parser.add_argument("--output-path", type=str, default="results.json", help="Path to save the results.")
    parser.add_argument("--modification", type=str, default="assertive_cue", help="Modification to apply to the tool description.")
    
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        data_path=args.data_path,
        output_path=args.output_path,
        modification=args.modification
    )

if __name__ == "__main__":
    main()

