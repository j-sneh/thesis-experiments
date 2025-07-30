import argparse
from core.experiment import Experiment, run_experiment

def main():
    parser = argparse.ArgumentParser(description="Run LLM tool selection experiment.")
    parser.add_argument("--model", type=str, default="llama3.2", help="Ollama model to use.")
    parser.add_argument("--data-path", type=str, default="data/BFCL_v3_simple.jsonl", help="Path to the data file.")
    parser.add_argument("--output-path", type=str, default="results", help="Path to save the results.")
    parser.add_argument("--modification", type=str, default="assertive_cue", help="Modification to apply to the tool description.", choices=["assertive_cue", "active_maintenance", "none", "noop"])
    parser.add_argument("--defense-mechanism", choices=["objective", "reword", "none", "noop"], type=str, default="objective", help="Defense mechanism to apply to the tool description.")
    parser.add_argument("--attack-mode", type=str, choices=["attack", "suffix-attack", "no-attack"], default="no-attack", help="Attack mode to use: 'attack', 'suffix-attack', or 'no-attack'.")
    parser.add_argument("--attacker-llm-model", type=str, help="Ollama model to use for attacker mode.")
    parser.add_argument("--defender-llm-model", type=str, help="Ollama model to use for defender mode.")
    parser.add_argument("--max-attempts", type=int, default=5, help="Maximum number of attack attempts.")
    parser.add_argument("--dataset-size", type=int, help="Number of items to use from the dataset (default: use all items).")
    parser.add_argument("--client", type=str, choices=["vllm", "ollama", "openai"], default="ollama", help="Inference client to use: 'vllm' or 'ollama'.")
    args = parser.parse_args()

    if args.attack_mode != "no-attack" and args.attacker_llm_model is None:
        # If attack mode is enabled but no attacker LLM model is provided, use the same model as the main LLM
        args.attacker_llm_model = args.model

    if args.attack_mode != "no-attack" and args.defender_llm_model is None:
        # If attack mode is enabled but no defender LLM model is provided, use the same model as the main LLM
        args.defender_llm_model = args.model

    run_experiment(
        model_name=args.model,
        data_path=args.data_path,
        output_path=args.output_path,
        modification=args.modification,
        defense_mechanism=args.defense_mechanism,
        attacker_mode=args.attack_mode,  # Pass the string value
        attacker_llm_model=args.attacker_llm_model,
        defender_llm_model=args.defender_llm_model,
        max_attempts=args.max_attempts,
        dataset_size=args.dataset_size,
        client=args.client
    )

if __name__ == "__main__":
    main()

