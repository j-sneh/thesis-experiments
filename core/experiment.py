from typing import List, Dict, Any
from typing import Optional, Tuple
from core.utils import load_data, load_cluster_data, spawn_server, OLLAMA_PATH, parse_json_inside_string
from core.tool_modifier import duplicate_and_rename_tool, modify_tool_description, format_tool_for_openai_api, get_defended_description, modify_tool_for_cluster_attack
from core.llm_clients import ChatGPTAzureClient, LLMClient, OpenAIClient, HFLocalClient, GeminiClient
from tqdm import tqdm
import json
import copy
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import random
import dotenv
import re

class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
    
    @property
    def value(self):
        return self._value

class HeadToHeadExperiment:
    def __init__(self, llm_client: LLMClient, attacker_llm_client: LLMClient, defender_llm_client:LLMClient, data_path: str, output_path: str, modification: str, defense_mechanism: str, attack_mode: str = "no-attack", max_attempts: int = 5, dataset_size: Optional[int] = None, 
                 cluster_id: Optional[int] = None, target_tool_index: Optional[int] = None, question_start: Optional[int] = None, question_end: Optional[int] = None, 
                 attack_modification_type: str = "both", seed: int = 42, eval_mode: bool = False, eval_name: Optional[str] = None, eval_description: Optional[str] = None, eval_config: Optional[str] = None, eval_attempt: Optional[int] = None, num_feedback_tools: int = 5, num_feedback_queries: int = 10):
        self.llm_client = llm_client
        self.output_path = output_path
        self.modification = modification
        self.defense_mechanism = defense_mechanism
        self.results: List[Dict[str, Any]] = []
        self.attack_mode = attack_mode
        self.attacker_llm_client = attacker_llm_client
        self.defender_llm_client = defender_llm_client
        self.max_attempts = max_attempts
        self.attacker_prompt_path = None
        self.attack_modification_type = attack_modification_type  # "description", "name", or "both"
        self.seed = seed
        
        # Evaluation mode parameters
        self.eval_mode = eval_mode
        self.eval_name = eval_name
        self.eval_description = eval_description
        self.eval_config = eval_config
        self.eval_attempt = eval_attempt
        self.num_feedback_tools = num_feedback_tools
        self.num_feedback_queries = num_feedback_queries

        self.question_start = question_start
        self.question_end = question_end
        
        # Parse evaluation configuration if provided
        if self.eval_mode and self.eval_config:
            self.eval_name, self.eval_description = self._parse_eval_config()
        
        # Load data based on attack mode
        if attack_mode == "cluster-attack":
            if cluster_id is None or target_tool_index is None or question_start is None or question_end is None:
                raise ValueError("cluster-attack mode requires cluster_id, target_tool_index, question_start, and question_end parameters")
            self.data = load_cluster_data(data_path, cluster_id, target_tool_index, question_start, question_end)
            self.attacker_prompt_path = "data/prompts/attacker/cluster-robust-to-reword"
        else:
            self.data = load_data(data_path)
            # Limit dataset size if specified
            if dataset_size is not None:
                self.data = self.data[:dataset_size]

            if attack_mode == "attack":
                self.attacker_prompt_path = "data/prompts/attacker/robust-to-reword"
            elif attack_mode == "suffix-attack":
                self.attacker_prompt_path = "data/prompts/attacker/suffix-robust-to-reword"
            elif attack_mode == "no-attack":
                self.attacker_prompt_path = None
            else:
                raise ValueError(f"Invalid attack mode: {attack_mode}")

    def _parse_eval_config(self) -> Tuple[str, str]:
        """Parse evaluation configuration from JSON/JSONL file."""
        
        if not os.path.exists(self.eval_config):
            raise FileNotFoundError(f"Evaluation config file not found: {self.eval_config}")
        
        configs = []
        try:
            with open(self.eval_config, 'r', encoding='utf-8') as f:
                if self.eval_config.endswith('.jsonl'):
                    # JSONL format - one JSON object per line
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                config = json.loads(line)
                                configs.append(config)
                            except json.JSONDecodeError as e:
                                raise ValueError(f"Invalid JSON on line {line_num} in {self.eval_config}: {e}")
                else:
                    # Single JSON format
                    try:
                        config = json.load(f)
                        if isinstance(config, list):
                            configs = config
                        else:
                            configs = [config]
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in {self.eval_config}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading evaluation config file {self.eval_config}: {e}")
        
        if not configs:
            raise ValueError(f"No configurations found in {self.eval_config}")
        
        # Select configuration based on criteria
        selected_config = None
        if self.eval_attempt is not None:
            # Select by specific attempt number
            for config in configs:
                if config.get('attempt') == self.eval_attempt:
                    selected_config = config
                    break
            if selected_config is None:
                raise ValueError(f"No configuration found with attempt number {self.eval_attempt} in {self.eval_config}")
        else:
            # Select by highest percentage
            best_config = None
            best_percent = -1
            for config in configs:
                percent = config.get('percent', 0)
                if percent > best_percent:
                    best_percent = percent
                    best_config = config
            selected_config = best_config
        
        if selected_config is None:
            raise ValueError(f"No valid configuration found in {self.eval_config}")
        
        # Extract name and description
        name = selected_config.get('name')
        description = selected_config.get('description')
        
        if name is None or description is None:
            raise ValueError(f"Selected configuration missing 'name' or 'description' fields: {selected_config}")
        
        return name, description

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

        response = self.attacker_llm_client.invoke(messages, None, seed=self.seed)
        if response is None:
            raise ValueError(f"[Attacker LLM] No response from attacker LLM")
        improvement = ''
        return_val = ''

        try:
            attacker_json = json.loads(response['message']['content'])
            return_val = attacker_json[attack_mode_string]
            improvement = attacker_json['improvement']
        except KeyError as e:
            raise TypeError(f"[Attacker LLM] Invalid JSON format: {e}\nRaw output: {response}")
        except TypeError as e:
            print(f"[Attacker LLM] Failed to parse attacker output: {e}\nRaw output: {response['message']['content']}")
            return_val = tool['description'] if attack_mode == 'attack' else ''
            improvement = 'Failed to parse the output. Reusing the previous output. Next time, I should only output JSON text and nothing else'
        except Exception as e:
            print(f"[Attacker LLM] Failed to parse attacker output: {e}\nRaw output: {response['message']['content']}")
            return_val = tool['description'] if attack_mode == 'attack' else ''
            improvement = 'Failed to parse the output. Reusing the previous output. Next time, I should only output JSON text and nothing else'

        return improvement, return_val

    

    
    
    def cluster_attack_get_improvement_with_feedback(self, tool: Dict[str, Any], improvement_history: list, current_description: str, current_name: str, question_results: list, overall_percent: float, tools_provided: list) -> Tuple[str, str, str]:
        """
        Get improvement suggestions for cluster attack mode with detailed feedback.
        
        Args:
            tool: The target tool
            improvement_history: History of previous attempts
            current_description: Current tool description
            current_name: Current tool name
            question_results: Results from each question in this attempt
            overall_percent: Overall selection percentage
            tools_provided: Tools that were provided to the LLM
            
        Returns:
            Tuple of (improvement, new_description, new_name)
        """
        if self.attacker_llm_client is None:
            raise RuntimeError("attacker_llm_client is None in cluster_attack_mode. This should not happen.")
        
        with open(self.attacker_prompt_path, 'r', encoding="utf-8") as file:
            prompt = file.read()
        
        messages = [
            {"role": "system", "content": prompt},
        ]
        
        # Add current tool information
        tool_info = f"Current tool name: {current_name}\nCurrent tool description: {current_description}"
        messages.append({"role": "user", "content": tool_info})
        
        # Add improvement_history of previous attempts
        for prev in improvement_history:
            prev_content = {
                'improvement': prev['improvement'],
                'description': prev['description'],
                'name': prev['name']
            }
            messages.append({"role": "assistant", "content": json.dumps(prev_content)})
            messages.append({"role": "user", "content": f"FEEDBACK: This modification was selected {prev['percent']}% of the time."})
        
        # Add detailed feedback from current attempt
        feedback_info = f"OVERALL FEEDBACK: Your current modification was selected {overall_percent:.2f}% of the time.\n\n"
        
        num_tools_included = self.num_feedback_tools
        
        # Only add tools feedback if num_feedback_tools > 0
        tools_included = tools_provided
        if num_tools_included > 0:
            feedback_info += f"Tools provided to the LLM:\n"
            # if we are not including all tools, we need to select a subset of tools to include
            # we always include the target tool
            if num_tools_included < len(tools_provided):
                tools_included = []
                tools_to_choose_from = []
                for tool in tools_provided:
                    # Ensure the target tool is included
                    if tool['function']['name'] == current_name:
                        tools_included.append(tool)
                    else:
                        tools_to_choose_from.append(tool)
                tools_included += tools_to_choose_from[:num_tools_included - len(tools_included)]

            # Add the tools to the feedback
            for _, tool in enumerate(tools_included):
                tool_name = tool['function']['name']
                tool_desc = tool['function'].get('description', 'No description')
                feedback_info += f"{tool_name} - {tool_desc} - {tool['function']['parameters']}\n"
        
        # Only add query feedback if num_feedback_queries > 0
        tool_names_included = [tool['function']['name'] for tool in tools_included]
        num_queries = self.num_feedback_queries
        if num_queries > 0:
            feedback_info += f"Here are a sample of the questions asked and the tools that were subsequently called:\n"
            # Select up to half successful and half unsuccessful trials, or num_queries in total
            # filter out results where not all tools were included
            filtered_results = [result for result in question_results if all(tool_name in tool_names_included for tool_name in result['called_tool_names'])]
            selected_results = []
            if len(filtered_results) < num_queries:
                selected_results = filtered_results
            else:
                successful = []
                unsuccessful = []
                for result in filtered_results:
                    if result['target_tool_selected']:
                        successful.append(result)
                    else:
                        unsuccessful.append(result)
                # Randomly shuffle the lists
                random.shuffle(successful)
                random.shuffle(unsuccessful)
                # Take up to num_queries//2 from each, but if not enough, fill from the other
                selected_results = successful[:num_queries//2] + unsuccessful[:num_queries//2]
                if len(selected_results) < num_queries:
                    # Fill up to num_queries with remaining from either list
                    remaining = (successful[num_queries//2:] + unsuccessful[num_queries//2:])[:num_queries - len(selected_results)]
                    selected_results += remaining

            for result in selected_results:
                question_text = result['question'][0]['content']
                selected = True if result['target_tool_selected'] else False
                called_tools = ", ".join(result['called_tool_names']) if result['called_tool_names'] else "None"
                feedback_info += f"Question: {question_text}\nTarget Tool Selected First: {selected} | Called Tools: {called_tools}\n"
        
        messages.append({"role": "user", "content": feedback_info})
        
        # DEBUG: Save the full attacker prompt to debug file
        debug_file_path = f"{self.output_path}-attacker_prompt_debug.txt"
        with open(debug_file_path, "a", encoding="utf-8") as debug_file:
            debug_prompt = "\n" + "="*80 + "\n"
            debug_prompt += f"ATTACKER PROMPT DEBUG - Attempt {len(improvement_history)}\n"
            debug_prompt += "="*80 + "\n"
            for i, msg in enumerate(messages):
                debug_prompt += f"Message {i+1} ({msg['role']}):\n"
                debug_prompt += f"{msg['content']}\n"
                debug_prompt += "-"*40 + "\n"
            debug_prompt += "="*80 + "\n"
            debug_file.write(debug_prompt)
            debug_file.flush()
        
        response = self.attacker_llm_client.invoke(messages, None, temperature=0.5, seed=self.seed) # higher temperature was causing no json output
        improvement = ''
        new_description = None
        new_name = None
        
        try:
            content = response['message']['content']
            # remove content between <think> and </think> if it exists
            # todo: customize parsing for different models
            if '<think>' in content and '</think>' in content:
                import re
                match = re.search(r'<think>.*?</think>(.*)', content, flags=re.DOTALL)

                if match:
                    content = match.group(1).strip()

            attacker_json = parse_json_inside_string(content)
            improvement = attacker_json.get('improvement', '')
            
            # Handle modification type restrictions
            if self.attack_modification_type in ["description", "both"]:
                new_description = attacker_json.get('description')
            if self.attack_modification_type in ["name", "both"]:
                new_name = attacker_json.get('name')
                
        except KeyError as e:
            raise TypeError(f"[Attacker LLM] Invalid JSON format: {e}\nRaw output: {response}")
        except TypeError as e:
            print(f"[Attacker LLM] Failed to parse attacker output: {e}\nRaw output: {response}")
            improvement = ''
            new_description = None
            new_name = None
        except Exception as e:
            print(f"[Attacker LLM] Failed to parse attacker output: {e}\nRaw output: {response['message']['content']}")
            improvement = ''
            new_description = None
            new_name = None
        
        return improvement, new_description, new_name
    
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
            
            response = self.llm_client.invoke(messages, trial_tools, seed=self.seed)
            
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
            output_file.flush()
            
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
            # for item in tqdm(self.data, desc="Processing items"):
            #     messages = item['question'][0]
            #     original_tool = item['function'][0]
            #     tools = duplicate_and_rename_tool(original_tool)
                
            #     # No-attack mode: simple modification
            #     tools[1] = modify_tool_description(tools[1], self.modification)
            #     formatted_tools = [format_tool_for_openai_api(tool) for tool in tools]
                
            #     # Run trials with both defense mechanisms
            #     self.run_trial_pair(
            #         messages, formatted_tools, self.defense_mechanism, 
            #         tools[1]['name'], output_file, item, self.modification, 0
            #     )

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for item in tqdm(self.data, desc="Queueing items"):
                    messages = item['question'][0]
                    original_tool = item['function'][0]
                    tools = duplicate_and_rename_tool(original_tool)
                    
                    # No-attack mode: simple modification
                    tools[1] = modify_tool_description(tools[1], self.modification)
                    formatted_tools = [format_tool_for_openai_api(tool) for tool in tools]
                    futures.append(executor.submit(self.run_trial_pair, messages, formatted_tools, self.defense_mechanism, tools[1]['name'], output_file, item, self.modification, 0))

                # Wait for all futures to complete
                for future in tqdm(futures, desc="Completing trials"):
                    future.result() 
        
        """
        For the cluster experiment, we need to pick a cluster in the data (1-5)
        Then, we need to pick a tool from the cluster, which is the tool we will be attacking.
        We will also need to pick a range of questions (prompts) associated with the cluster.
        
        We will run for n attempts, where each attempt is a revised tool description. The information the attacker will have is the following:
        - The current tool description
        - Which tool was selected in the previous attempt
        - Which tools were provided to the LLM in the previous attempt

        The revision will be either a new tool description, or a new name, or both.
        """



                
    
                


    def run(self):
        if self.attack_mode == "suffix-attack":
            self.run_suffix_attack()
        elif self.attack_mode == "attack":
            self.run_attack()
        elif self.attack_mode == "no-attack":
            self.run_no_attack()
        elif self.attack_mode == "cluster-attack":
            self.run_cluster_attack()
        else:
            raise ValueError(f"Invalid attack mode: {self.attack_mode}")
        
        print(f"Experiment finished. Results saved to {self.output_path}.jsonl and {self.output_path}-improvement_history.jsonl.")

    def run_cluster_attack(self):
        """Run the cluster-attack mode experiment with iterative improvement."""
        with open(f"{self.output_path}.jsonl", "w", encoding="utf-8") as output_file, open(f"{self.output_path}-improvement_history.jsonl", "w", encoding="utf-8") as improvement_history_file:
            # Get the target tool and all tools from the cluster
            all_tools = self.data['all_tools']
            
            target_tool_index = self.data['target_tool_index']
            improvement_history = []
            improvement = 'This is the original tool'


            current_description = all_tools[target_tool_index].get('description', '')
            current_name = all_tools[target_tool_index]['name']

            all_other_tool_names = set([tool['name'] for tool in all_tools if tool['name'] != current_name])
            
            # Repl
            if self.eval_mode:
                # Use predetermined values from evaluation mode
                current_description = self.eval_description if self.eval_description is not None else current_description
                current_name = self.eval_name if self.eval_name is not None else current_name
                
                # Apply modification to the predetermined description if specified
                if self.modification and self.modification not in ['none', 'noop']:
                    temp_tool = {'description': current_description}
                    modified_tool = modify_tool_description(temp_tool, self.modification)
                    current_description = modified_tool['description']
                
                improvement = 'Evaluation mode: using predetermined tool configuration'

            
            for attempt in range(self.max_attempts):
                # Create modified tools list with the updated target tool
                all_tools[target_tool_index] = modify_tool_for_cluster_attack(all_tools[target_tool_index], current_description, current_name)
                
                # Format tools for OpenAI API
                formatted_tools = [format_tool_for_openai_api(t) for t in all_tools]
                
                # Apply defense if needed
                defended_tools = self.apply_defense_to_tools(formatted_tools, self.defense_mechanism, self.defender_llm_client)
                
                # Initialize thread-safe counters and shared data structures
                selection_counter = ThreadSafeCounter()
                calls_counter = ThreadSafeCounter()
                question_results = []
                results_lock = threading.Lock()
                file_lock = threading.Lock()

                def run_trial(question, idx):
                    # random seed for each trial
                    random.seed(self.seed + attempt + idx)
                    messages = question
                    
                    # shuffle tools
                    shuffled_tools = defended_tools.copy()
                    random.shuffle(shuffled_tools)  

                    # Run the trial with all tools
                    response = self.llm_client.invoke(messages, shuffled_tools, seed=self.seed, temperature=0.0)
                    
                    # Extract called tool names
                    tool_calls = []
                    called_tool_names = []
                    if response and 'message' in response and response['message'].get('tool_calls'):
                        tool_calls = response['message']['tool_calls']
                        called_tool_names = [r['function']['name'] for r in tool_calls]
                    
                    # Count selections of our target tool
                    # We only look at the first tool call, as we assume the LLM will only call one tool
                    # TODO: Handle the case where the LLM calls multiple tools
                    target_selected = False if tool_calls is None or len(tool_calls) == 0 else called_tool_names[0] == current_name
                    
                    # Update counters
                    if target_selected:
                        selection_counter.increment()
                    calls_counter.increment()
                    
                    # Store question result (thread-safe)
                    question_result = {
                        'question': messages,
                        # TODO: deal with defense, too -- may need to give limited information or all information
                        'tools': formatted_tools, 
                        'called_tool_names': called_tool_names,
                        'target_tool_selected': target_selected,
                        'target_tool_name': current_name
                    }
                    with results_lock:
                        question_results.append(question_result)
                    
                    # Create result record for output file
                    result = {
                        "id": f"cluster-{self.data['cluster_id']}-q{self.question_start + idx}",
                        "question": messages,
                        "called_tool_names": called_tool_names,
                        "target_tool_name": current_name,
                        "target_tool_selected": target_selected,
                        "defense_used": self.defense_mechanism,
                        "attack_mode": self.attack_mode,
                        "attempt": attempt,
                        "cluster_id": self.data['cluster_id'],
                        "target_tool_index": target_tool_index,
                        "tools_provided": shuffled_tools,
                        "original_tools": all_tools
                    }
                    
                    # Thread-safe file writing
                    with file_lock:
                        output_file.write(json.dumps(result) + "\n")
                        output_file.flush()
                    
                # Parallelize the question processing
                with ThreadPoolExecutor(max_workers=10) as executor:
                    list(tqdm(executor.map(lambda args: run_trial(args[1], args[0]), enumerate(self.data['questions'])), 
                             total=len(self.data['questions']), 
                             desc=f"Cluster Attack Attempt {attempt+1}"))
                
                # Get final counts after all threads complete
                total_count = selection_counter.value
                total_calls = calls_counter.value
                # Calculate overall percentage
                percent = 100.0 * total_count / total_calls if total_calls > 0 else 0.0
                
                
                # Save improvement history
                improvement_record = {
                    'attempt': attempt,
                    'improvement': improvement,
                    'description': current_description,
                    'name': current_name,
                    'percent': percent,
                    'cluster_id': self.data['cluster_id'],
                    'target_tool_index': target_tool_index,
                    'total_questions': len(self.data['questions']),
                    'total_selections': total_count,
                    'total_calls': total_calls
                }
                improvement_history.append(improvement_record)
                improvement_history_file.write(json.dumps(improvement_record) + "\n")
                improvement_history_file.flush()

                print(f"Attempt {attempt+1}: {percent:.2f}% selection rate ({total_count}/{total_calls} selections)")

                if attempt == self.max_attempts - 1:
                    # no improvement if last attempt
                    break
                
                # Get improvement from attacker with detailed feedback
                improvement = ''
                new_description = None
                new_name = None

                
                max_iters = 10
                iteration = 0
                while (new_description is None or new_name is None):
                    print(f"Generating improvement for attempt {attempt}. Iteration: {iteration}/{max_iters}")
                    if iteration >= max_iters:
                        raise RuntimeError(f"Model failed to generate a valid JSON with a new description or name after {max_iters} iterations.")
                    improvement, new_description, new_name = self.cluster_attack_get_improvement_with_feedback(
                        all_tools[target_tool_index], improvement_history, current_description, current_name, 
                        question_results, percent, formatted_tools
                    )
                    iteration += 1

                    if new_name is not None:
                        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$", new_name):
                            # replace all non-alphanumeric characters with an underscore
                            new_name = re.sub(r"[^a-zA-Z0-9_]", "_", new_name)
                            new_name = re.sub(r"_+", "_", new_name)
                        if len(new_name) > 64:
                            new_name = new_name[:63]
                        if new_name in all_other_tool_names:
                            # was sometimes copying names from other tools, which is not what we want
                            print(f"New name {new_name} is already in use. Using current name {current_name} instead.")
                            new_name = current_name
                
                
                # Update the target tool with new description/name
                if new_description is not None:
                    current_description = new_description
                if new_name is not None:
                    current_name = new_name
                

        
def run_head_to_head_experiment(model_name, data_path, output_path, modification, defense_mechanism, attacker_mode="no-attack", attacker_llm_model=None, defender_llm_model=None, max_attempts=5, dataset_size=None, client="openai",
                                cluster_id=None, target_tool_index=None, question_start=None, question_end=None, attack_modification_type="both", server_port=8000, server_type="hflocal",
                                model_url=None, attacker_url=None, defender_url=None, seed=42, eval_mode=False, eval_name=None, eval_description=None, eval_config=None, eval_attempt=None,
                                api_key=None, num_feedback_tools=5, num_feedback_queries=10):
    """
    Run an LLM tool selection experiment.
    
    Args:
        model_name: The name of the main LLM model to use
        data_path: Path to the data file containing test cases
        output_path: Path where results will be saved
        modification: Type of modification to apply to tool descriptions
        defense_mechanism: Defense mechanism to apply
        attacker_mode: Attack mode string: 'attack', 'suffix-attack', 'cluster-attack', or 'no-attack'
        attacker_llm_model: Model to use for attacker (if different from main model)
        defender_llm_model: Model to use for defender (if different from main model)
        max_attempts: Maximum number of attack attempts
        dataset_size: Number of items to use from the dataset (None for all items)
        client: Inference client to use ('ollama', 'vllm', or 'openai')
        cluster_id: Cluster ID for cluster-attack mode (1-10)
        target_tool_index: Target tool index for cluster-attack mode (0-4)
        question_start: Start index for questions in cluster-attack mode
        question_end: End index for questions in cluster-attack mode
        attack_modification_type: Type of modification for cluster-attack mode ('description', 'name', or 'both')
        server_port: Port to use for spawning servers (if not using existing URLs)
        server_type: Type of server to spawn ('ollama' or 'vllm')
        model_url: URL for the main model server (if already running, skips spawning)
        attacker_url: URL for the attacker model server (if already running, skips spawning)
        defender_url: URL for the defender model server (if already running, skips spawning)
        seed: Random seed for reproducible results
        api_key: API key for external OpenAI-compatible server (required when server_type is 'external')
    """

    client_class = None
    # TODO: deprecate client argument entirely
    if server_type == "hflocal":
        client_class = HFLocalClient
    elif server_type == "gemini":
        client_class = GeminiClient
    else:
        client_class = OpenAIClient

    
    # Start the servers for the LLMs or use provided URLs
    models = [model_name, attacker_llm_model, defender_llm_model]
    provided_urls = [model_url, attacker_url, defender_url]
    
    model_processes = {}
    models_needing_servers = []
    
    # Handle external server type - use model_url for all models
    if server_type == "external":
        for model in models:
            if model:
                model_processes[model] = (model_url, None, None)
                print(f"Using external server for {model} at {model_url}")
    elif server_type == "gemini":
        # Handle gemini server type - use default URL or provided model_url
        gemini_url = model_url if model_url else "https://generativelanguage.googleapis.com/v1beta/openai/"
        for model in models:
            if model:
                model_processes[model] = (gemini_url, None, None)
                print(f"Using Gemini server for {model} at {gemini_url}")
    else:
        # First, handle models with provided URLs
        for model, url in zip(models, provided_urls):
            if url and url != '':
                # Use provided URL, no server spawning needed
                model_processes[model] = (url, None, None)
                print(f"Using existing server for {model} at {url}")
            elif model:
                # Model exists but no URL provided, needs server
                models_needing_servers.append(model)
    
    # Remove duplicates while preserving order
    models_needing_servers = set(models_needing_servers)
    
    # Spawn servers for models that need them
    if len(models_needing_servers) > 0:
        if server_type == "ollama":
            # Need to download the models if they are not already present
            # Ollama can handle multiple models on one server
            url, process, log_handle = spawn_server(None, server_port, server_type, output_path)
            for model in models_needing_servers:
                model_processes[model] = (url, process, log_handle)
                print(f"Spawned Ollama server for {model} at {url}")
        elif server_type == "vllm":  # vllm
            # vllm needs separate server for each model
            for model in models_needing_servers:
                print(f"Spawning vLLM server for {model} at port {server_port}")
                url, process, log_handle = spawn_server(model, server_port, server_type, output_path)
                model_processes[model] = (url, process, log_handle)
                server_port += 1
                print(f"Spawned vLLM server for {model} at {url}")

    try:
        # breakpoint()
        if server_type == "hflocal":
            llm_client = client_class(model_name, None)
            attacker_llm_client = client_class(attacker_llm_model, base_url=None) if attacker_llm_model else llm_client     
            defender_llm_client = client_class(defender_llm_model, base_url=None) if defender_llm_model else llm_client
        elif server_type in ["external", "gemini"]:
            # For external and gemini servers, pass the API key
            llm_client = client_class(model_name, base_url=model_processes[model_name][0], api_key=api_key)
            attacker_llm_client = client_class(attacker_llm_model, base_url=model_processes[attacker_llm_model][0], api_key=api_key) if attacker_llm_model else llm_client     
            defender_llm_client = client_class(defender_llm_model, base_url=model_processes[defender_llm_model][0], api_key=api_key) if defender_llm_model else llm_client
        elif server_type == "azure":
            # model_name is the "azure_deployment" name

            # We assume OX_AZURE_API_VERSION and OX_AZURE_ENDPOINT are set in the environment variables, this is to keep it separate from the rest
            # of the configuration, since it should not be outputted to 
            # the args file
            dotenv.load_dotenv()
            api_version = os.getenv("OX_AZURE_API_VERSION")
            azure_endpoint = os.getenv("OX_AZURE_ENDPOINT")
            if api_version is None or azure_endpoint is None:
                raise ValueError("OX_AZURE_API_VERSION and OX_AZURE_ENDPOINT must be set in the environment variables for azure runs")

            llm_client = ChatGPTAzureClient(model_name, api_version, azure_endpoint)
            attacker_llm_client = ChatGPTAzureClient(attacker_llm_model, api_version, azure_endpoint) if attacker_llm_model else llm_client
            defender_llm_client = ChatGPTAzureClient(defender_llm_model, api_version, azure_endpoint) if defender_llm_model else llm_client
        else:
            # For ollama and vllm servers, use default API key
            llm_client = client_class(model_name, base_url=model_processes[model_name][0])
            attacker_llm_client = client_class(attacker_llm_model, base_url=model_processes[attacker_llm_model][0]) if attacker_llm_model else llm_client     
            defender_llm_client = client_class(defender_llm_model, base_url=model_processes[defender_llm_model][0]) if defender_llm_model else llm_client

        if server_type not in ["hflocal", "external", "gemini"]:
            llm_client.wait_for_server_to_start()
            attacker_llm_client.wait_for_server_to_start()
            defender_llm_client.wait_for_server_to_start()

        if server_type == "ollama":
            # Ollama needs to pull the models before the client connects
           # for model in models:
           #     subprocess.run([OLLAMA_PATH, "pull", model], env=os.environ, check=True)
           pass # for now, assume models have been pulled - this has been a source of error for me
    # 

        experiment = HeadToHeadExperiment(
            llm_client=llm_client,
            attacker_llm_client=attacker_llm_client,
            defender_llm_client=defender_llm_client,
            data_path=data_path,
            output_path=output_path,
            modification=modification,
            defense_mechanism=defense_mechanism,
            attack_mode=attacker_mode,
            max_attempts=max_attempts,
            dataset_size=dataset_size,
            cluster_id=cluster_id,
            target_tool_index=target_tool_index,
            question_start=question_start,
            question_end=question_end,
            attack_modification_type=attack_modification_type,
            seed=seed,
            eval_mode=eval_mode,
            eval_name=eval_name,
            eval_description=eval_description,
            eval_config=eval_config,
            eval_attempt=eval_attempt,
            num_feedback_tools=num_feedback_tools,
            num_feedback_queries=num_feedback_queries
        )
        experiment.run()
    finally:
        # Only terminate processes that were actually spawned (not using provided URLs)
        for model, (url, process, log_handle) in model_processes.items():
            if process is not None and log_handle is not None:
                # Kill the server before exiting
                print(f"Terminating spawned server for {model}")
                process.terminate()
                log_handle.close()
