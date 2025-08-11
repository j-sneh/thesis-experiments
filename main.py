import argparse
from core.experiment import HeadToHeadExperiment, run_head_to_head_experiment


def main():
    parser = argparse.ArgumentParser(description="Run LLM tool selection experiment.")
    parser.add_argument("--model", type=str, default="llama3.2", help="Model to use.")
    parser.add_argument("--data-path", type=str, default="data/BFCL_v3_simple.jsonl", help="Path to the data file.")
    parser.add_argument("--output-path", type=str, default="results", help="Path to save the results.")
    parser.add_argument("--modification", type=str, default="assertive_cue", help="Modification to apply to the tool description.", choices=["assertive_cue", "active_maintenance", "none", "noop"])
    parser.add_argument("--defense-mechanism", choices=["objective", "reword", "none", "noop"], type=str, default="objective", help="Defense mechanism to apply to the tool description.")
    parser.add_argument("--attack-mode", type=str, choices=["attack", "suffix-attack", "cluster-attack", "no-attack"], default="no-attack", help="Attack mode to use: 'attack', 'suffix-attack', 'cluster-attack', or 'no-attack'.")
    parser.add_argument("--attacker-llm-model", type=str, help="Model to use for attacker mode.")
    parser.add_argument("--defender-llm-model", type=str, help="Model to use for defender mode.")
    parser.add_argument("--max-attempts", type=int, default=5, help="Maximum number of attack attempts.")
    parser.add_argument("--dataset-size", type=int, help="Number of items to use from the dataset (default: use all items).")
    parser.add_argument("--client", type=str, choices=["vllm", "ollama", "openai", "hflocal"], default="hflocal", help="Inference client to use: 'vllm', 'ollama', or 'openai'.")
    
    
    # Cluster attack specific arguments
    parser.add_argument("--cluster-id", type=int, help="Cluster ID for cluster-attack mode (1-10).")
    parser.add_argument("--target-tool-index", type=int, help="Target tool index for cluster-attack mode (0-4).")
    parser.add_argument("--question-start", type=int, help="Start index for questions in cluster-attack mode.")
    parser.add_argument("--question-end", type=int, help="End index for questions in cluster-attack mode. (exclusive)")
    parser.add_argument("--attack-modification-type", type=str, choices=["description", "name", "both"], default="both", help="Type of modification for cluster-attack mode: 'description', 'name', or 'both'.")

    # Server args
    parser.add_argument("--server-port", type=int, default=8000, help="Port to use for the server.")
    parser.add_argument("--server-type", type=str, choices=["ollama", "vllm", "hflocal"], default="ollama", help="Type of server to use: 'ollama' or 'vllm'.")
    
    args = parser.parse_args()

    # Validate cluster attack parameters
    if args.attack_mode == "cluster-attack":
        if args.cluster_id is None:
            parser.error("--cluster-id is required for cluster-attack mode")
        if args.target_tool_index is None:
            parser.error("--target-tool-index is required for cluster-attack mode")
        if args.question_start is None:
            parser.error("--question-start is required for cluster-attack mode")
        if args.question_end is None:
            parser.error("--question-end is required for cluster-attack mode")
        
        # Validate ranges
        if args.question_start < 0:
            parser.error("--question-start must be non-negative")
        if args.question_end <= args.question_start:
            parser.error("--question-end must be greater than --question-start")

    if args.attack_mode != "no-attack" and args.attacker_llm_model is None:
        # If attack mode is enabled but no attacker LLM model is provided, use the same model as the main LLM
        args.attacker_llm_model = args.model

    if args.attack_mode not in ["none", "noop"] and args.defender_llm_model is None:
        # If attack mode is enabled but no defender LLM model is provided, use the same model as the main LLM
        args.defender_llm_model = args.model

    breakpoint()
    run_head_to_head_experiment(
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
        client=args.client,
        cluster_id=args.cluster_id,
        target_tool_index=args.target_tool_index,
        question_start=args.question_start,
        question_end=args.question_end,
        attack_modification_type=args.attack_modification_type,
        server_port=args.server_port,
        server_type=args.server_type
    )

if __name__ == "__main__":
    main()

