import argparse
from core.experiment import Experiment, run_experiment

def main():
    parser = argparse.ArgumentParser(description="Run LLM tool selection experiment.")
    parser.add_argument("--model", type=str, default="llama3.2", help="Ollama model to use.")
    parser.add_argument("--data-path", type=str, default="data/BFCL_v3_simple.jsonl", help="Path to the data file.")
    parser.add_argument("--output-path", type=str, default="results.jsonl", help="Path to save the results.")
    parser.add_argument("--modification", type=str, default="assertive_cue", help="Modification to apply to the tool description.")
    parser.add_argument("--defense-mechanism", type=str, default="objective", help="Defense mechanism to apply to the tool description.")
    parser.add_argument("--attacker-mode", type=bool, default=False, help="Whether to run in attacker mode.")
    parser.add_argument("--attacker-llm-model", type=str, help="Ollama model to use for attacker mode.")
    parser.add_argument("--defender-llm-model", type=str, help="Ollama model to use for defender mode.")
    args = parser.parse_args()

    if args.attacker_mode and args.attacker_llm_model is None:
        # If attacker mode is enabled but no attacker LLM model is provided, use the same model as the main LLM
        args.attacker_llm_model = args.model

    if args.attacker_mode and args.defender_llm_model is None:
        # If attacker mode is enabled but no defender LLM model is provided, use the same model as the main LLM
        args.defender_llm_model = args.model

    run_experiment(
        model_name=args.model,
        data_path=args.data_path,
        output_path=args.output_path,
        modification=args.modification,
        defense_mechanism=args.defense_mechanism,
        attacker_mode=args.attacker_mode,
        attacker_llm_model=args.attacker_llm_model,
        defender_llm_model=args.defender_llm_model
    )

if __name__ == "__main__":
    main()

