from typing import List, Dict, Any
from typing import Optional, Tuple
from core.utils import load_data
from core.tool_modifier import duplicate_and_rename_tool, modify_tool_description, format_tool_for_openai_api, get_defended_description
from core.llm_clients import LLMClient, OllamaClient, VLLMClient
from tqdm import tqdm
import json
import copy
import os

class Experiment:
    def __init__(self, llm_client: LLMClient, attacker_llm_client: LLMClient, defender_llm_client:LLMClient, data_path: str, output_path: str, modification: str, defense_mechanism: str, attack_mode: str = "no-attack", max_attempts: int = 5, dataset_size: Optional[int] = None):
        self.llm_client = llm_client
        self.data = load_data(data_path)
        # Limit dataset size if specified
        if dataset_size is not None:
            self.data = self.data[:dataset_size]
        self.output_path = output_path
        self.modification = modification
        self.defense_mechanism = defense_mechanism
        self.results: List[Dict[str, Any]] = []
        self.attack_mode = attack_mode
        self.attacker_llm_client = attacker_llm_client
        self.defender_llm_client = defender_llm_client
        self.max_attempts = max_attempts
        self.attacker_prompt_path = None

        if attack_mode == "attack":
            self.attacker_prompt_path = "data/prompts/attacker/robust-to-reword"
        elif attack_mode == "suffix-attack":
            self.attacker_prompt_path = "data/prompts/attacker/suffix-robust-to-reword"
        elif attack_mode == "no-attack":
            self.attacker_prompt_path = None
        else:
            raise ValueError(f"Invalid attack mode: {attack_mode}")

    def attack_get_improvement(self, tool: Dict[str, Any], improvement_history: list, attack_mode: str) -> Tuple[str, str]:
        if self.attacker_llm_client is None:
            raise RuntimeError("attacker_llm_client is None in attacker_mode. This should not happen.")
        with open(self.attacker_prompt_path, 'r', encoding="utf-8") as file:
            prompt = file.read()
        
    

        attack_mode_string = 'description' if attack_mode == 'attack' else 'suffix' if attack_mode == 'suffix-attack' else None

        if attack_mode_string is None:
            raise ValueError(f"Invalid attack mode: {attack_mode}")

        messages = [
            {"role": "system", "content": prompt},
        ]
        if attack_mode == 'attack':
            messages.append({"role": "user", "content": tool['description']})
        else:
            messages.append({"role": "user", "content": "Please generate a suffix that will be appended to tool descriptions, using the subsequent feedback provided, if available."})
        # Add improvement_history of previous attempts

        for prev in improvement_history:
            messages.append({"role": "assistant", "content": json.dumps({'improvement': prev['improvement'], attack_mode_string: prev[attack_mode_string]})})
            messages.append({"role": "user", "content": f"FEEDBACK: Your last {attack_mode_string} was selected {prev['percent']}% of the time."})

        response = self.attacker_llm_client.invoke(messages, None)
        improvement = ''
        return_val = ''

        try:
            attacker_json = json.loads(response['message']['content'])
            return_val = attacker_json[attack_mode_string]
            improvement = attacker_json.get('improvement', '')
        except KeyError as e:
            raise TypeError(f"[Attacker LLM] Invalid JSON format: {e}\nRaw output: {response}")
        except Exception as e:
            print(f"[Attacker LLM] Failed to parse attacker output: {e}\nRaw output: {response['message']['content']}")
            return_val = tool['description'] if attack_mode == 'attack' else ''
            improvement = ''

        return improvement, return_val
    
    def apply_defense_to_tools(self, tools, defense, llm_client):
        defended_tools = [copy.deepcopy(tool) for tool in tools]
        for i, tool in enumerate(defended_tools):
             defended_tools[i]['function']['description'] = get_defended_description(tool, defense, llm_client)
        return defended_tools

    def run_trial_pair(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], 
                      defense: str, target_tool_name: str, output_file, 
                      item: Dict[str, Any], modification: str = None, attempt: int = 0) -> Tuple[int, int]:
        """
        Run a pair of trials with the same tools in different orders (original-first and biased-first).
        
        Args:
            messages: The messages to send to the LLM
            tools: The tools to provide (should be formatted for OpenAI API)
            defense: The defense mechanism to apply
            target_tool_name: The name of the tool to count selections for
            output_file: File to write results to
            item: The original data item
            modification: The modification applied (for result logging)
            
        Returns:
            Tuple of (total_count, total_calls) for the target tool
        """
        defended_tools = self.apply_defense_to_tools(tools, defense, self.defender_llm_client)
        
        total_count = 0
        total_calls = 0
        
        for trial_type in ["original-first", "biased-first"]:
            trial_tools = defended_tools if trial_type == "original-first" else defended_tools[::-1]
            
            response = self.llm_client.invoke(messages, trial_tools)
            
            called_tool_names = [r['function']['name'] for r in response['message']['tool_calls']] if response and 'message' in response and response['message']['tool_calls'] else []
            
            count = called_tool_names.count(target_tool_name)
            total_count += count
            total_calls += len(called_tool_names) if called_tool_names else 1
            
            # Create result record
            result = {
                "id": item["id"],
                "question": item["question"],
                "original_tool": item["function"][0],
                "tools_provided": trial_tools,
                "called_tool_names": called_tool_names,
                "trial_type": trial_type,
                "defense_used": defense,
                "attack_mode": self.attack_mode,
                "attempt": attempt,
            }
            
            if modification is not None:
                result['modification'] = modification
                
            output_file.write(json.dumps(result) + "\n")
            
        return total_count, total_calls

    def run_suffix_attack(self):
        suffix = ""
        improvement_history = []
        
        with open(f"{self.output_path}.jsonl", "w", encoding="utf-8") as output_file, open(f"{self.output_path}-improvement_history.jsonl", "w", encoding="utf-8") as improvement_history_file:
            for attempt in range(self.max_attempts):
                # Use the attacker prompt and feedback to generate a new suffix
                improvement, suffix = self.attack_get_improvement(None, improvement_history, self.attack_mode)
                total_count = 0
                total_calls = 0
                item_stats = []
                
                for item in tqdm(self.data, desc=f"Suffix-Attack Attempt {attempt+1}"):
                    messages = item['question'][0]
                    original_tool = item['function'][0]
                    tools = duplicate_and_rename_tool(original_tool)
                    # Apply current suffix to tool2's description
                    tools[1]['description'] += suffix
                    function_name_2 = tools[1]['name']

                    formatted_tools = [format_tool_for_openai_api(t) for t in tools]
                    
                    # Use shared trial logic
                    item_total_count, item_total_calls = self.run_trial_pair(
                        messages, formatted_tools, self.defense_mechanism, 
                        function_name_2, output_file, item, suffix, attempt
                    )

                    percent = 100.0 * item_total_count / item_total_calls if item_total_calls > 0 else 0.0
                    item_stats.append({'id': item['id'], 'percent': percent})
                    total_count += item_total_count
                    total_calls += item_total_calls

                # Calculate aggregate statistics
                overall_percent = 100.0 * total_count / total_calls if total_calls > 0 else 0.0

                # Save improvement_history for this attempt
                improvement_history.append({
                    'attempt': attempt,
                    'suffix': suffix,
                    'improvement': improvement,
                    'percent': overall_percent,
                    'item_stats': item_stats
                })
                improvement_history_file.write(json.dumps(improvement_history[-1]) + "\n")
                improvement_history_file.flush()

    def run_attack(self):
        """Run the attack mode experiment with iterative improvement."""
        with open(f"{self.output_path}.jsonl", "w", encoding="utf-8") as output_file, open(f"{self.output_path}-improvement_history.jsonl", "w", encoding="utf-8") as improvement_history_file:
            for item in tqdm(self.data, desc="Processing items"):
                messages = item['question'][0]
                original_tool = item['function'][0]
                tools = duplicate_and_rename_tool(original_tool)
                
                improvement_history = []
                percent = 0.0
                description = tools[1]['description']
                function_name_2 = tools[1]['name']


                for attempt in range(self.max_attempts):
                    # Set tool2's description to the latest
                    improvement, description = self.attack_get_improvement(tools[1], improvement_history, self.attack_mode)
                    tools[1]['description'] = description
                    formatted_tools = [format_tool_for_openai_api(t) for t in tools]
                    
                    # Use shared trial logic for attack attempts
                    total_count, total_calls = self.run_trial_pair(
                        messages, formatted_tools, self.defense_mechanism, 
                        function_name_2, output_file, item, description, attempt
                    )
                    
                    # Calculate overall percentage
                    percent = 100.0 * total_count / total_calls if total_calls > 0 else 0.0

                    improvement_history.append({'id': item['id'], 'attempt': attempt, 'improvement': improvement, 'description': description, 'percent': percent})
                    improvement_history_file.write(json.dumps(improvement_history[-1]) + "\n")
                    improvement_history_file.flush()

    def run_no_attack(self):
        """Run the no-attack mode experiment with simple modification."""
        with open(f"{self.output_path}.jsonl", "w", encoding="utf-8") as output_file:
            for item in tqdm(self.data, desc="Processing items"):
                messages = item['question'][0]
                original_tool = item['function'][0]
                tools = duplicate_and_rename_tool(original_tool)
                
                # No-attack mode: simple modification
                tools[1] = modify_tool_description(tools[1], self.modification)
                formatted_tools = [format_tool_for_openai_api(tool) for tool in tools]
                
                # Run trials with both defense mechanisms
                self.run_trial_pair(
                    messages, formatted_tools, self.defense_mechanism, 
                    tools[1]['name'], output_file, item, self.modification, 0
                )

    def run(self):
        if self.attack_mode == "suffix-attack":
            self.run_suffix_attack()
        elif self.attack_mode == "attack":
            self.run_attack()
        elif self.attack_mode == "no-attack":
            self.run_no_attack()
        else:
            raise ValueError(f"Invalid attack mode: {self.attack_mode}")
        
        print(f"Experiment finished. Results saved to {self.output_path}.jsonl and {self.output_path}-improvement_history.jsonl.")

def run_experiment(model_name, data_path, output_path, modification, defense_mechanism, attacker_mode="no-attack", attacker_llm_model=None, defender_llm_model=None, max_attempts=5, dataset_size=None, engine="ollama"):
    """
    Run an LLM tool selection experiment.
    
    Args:
        model_name: The name of the main LLM model to use
        data_path: Path to the data file containing test cases
        output_path: Path where results will be saved
        modification: Type of modification to apply to tool descriptions
        defense_mechanism: Defense mechanism to apply
        attacker_mode: Attack mode string: 'attack', 'suffix-attack', or 'no-attack'
        attacker_llm_model: Model to use for attacker (if different from main model)
        defender_llm_model: Model to use for defender (if different from main model)
        max_attempts: Maximum number of attack attempts
        dataset_size: Number of items to use from the dataset (None for all items)
        engine: Inference engine to use ('ollama' or 'vllm')
    """
    
        

    client = VLLMClient if engine == "vllm" else OllamaClient
    llm_client = client(model_name)
    attacker_llm_client = client(attacker_llm_model) if attacker_llm_model else llm_client     
    defender_llm_client = client(defender_llm_model) if defender_llm_model else llm_client

    experiment = Experiment(
        llm_client=llm_client,
        attacker_llm_client=attacker_llm_client,
        defender_llm_client=defender_llm_client,
        data_path=data_path,
        output_path=output_path,
        modification=modification,
        defense_mechanism=defense_mechanism,
        attack_mode=attacker_mode,
        max_attempts=max_attempts,
        dataset_size=dataset_size
    )
    experiment.run()

