from typing import List, Dict, Any
from core.utils import load_data
from core.tool_modifier import duplicate_and_rename_tool, modify_tool_description, format_tool_for_openai_api, get_defended_description
from core.llm_clients import LLMClient, OllamaClient
from tqdm import tqdm
import json
import copy

class Experiment:
    def __init__(self, llm_client: LLMClient, data_path: str, output_path: str, modification: str, defense_mechanism: str):
        self.llm_client = llm_client
        self.data = load_data(data_path)
        self.output_path = output_path
        self.modification = modification
        self.defense_mechanism = defense_mechanism
        self.results: List[Dict[str, Any]] = []

    def run(self):
        with open(self.output_path, "w", encoding="utf-8") as f:
            for item in tqdm(self.data, desc="Processing items"):
                # print(item)
                messages = item['question'][0]
                original_tool = item['function'][0]

                # Duplicate and rename
                tools = duplicate_and_rename_tool(original_tool)

                # For now, tool 2 is always the modified tool
                
                # Modify one of the tools
                tools[1] = modify_tool_description(tools[1], self.modification)

                # Format tools for LLM
                tools = [format_tool_for_openai_api(tool) for tool in tools]
                

                # Run experiment with each tool order   
                for defense_mechanism in ["noop", self.defense_mechanism]:
                    defended_tools = [copy.deepcopy(tool) for tool in tools]
                    # Apply defense mechanism
                    for i, tool in enumerate(defended_tools):
                        defended_tools[i]['function']['description'] = get_defended_description(tool, defense_mechanism, self.llm_client)

                    for trial_type in ["original-first", "biased-first"]:
                        # Invoke LLM
                        response = self.llm_client.invoke(messages, defended_tools if trial_type == "original-first" else defended_tools[::-1])

                        # Store original descriptions and post-defense descriptions
                        for i, tool in enumerate(tools):
                            tool['original_description'] = tool['function']['description']
                            tool['post_defense_description'] = defended_tools[i]['function']['description']
                        # Record result (store tools with 'modified' field for analysis)
                        result = self._record_result(item, response, tools, trial_type, defense_mechanism)
                        f.write(json.dumps(result) + "\n")

    def _record_result(self, item: Dict[str, Any], response: Dict[str, Any], tools: List[Dict[str, Any]], trial_type: str, defense_used: str):
        called_tool_names = None
        if response and 'message' in response and response['message']['tool_calls']:
            # TODO: Handle multiple tool calls
            called_tool_names = [r['function']['name'] for r in response['message']['tool_calls']]

        result = {
            "id": item["id"],
            "question": item["question"],
            "original_tool": item["function"][0],
            "tools_provided": tools,
            "called_tool_names": called_tool_names,  
            "trial_type": trial_type,
            "modification": self.modification,
            "defense_used": defense_used
        }
        self.results.append(result)
        return result

def run_experiment(model_name: str, data_path: str, output_path: str, modification: str, defense_mechanism: str):
    llm_client = OllamaClient(model=model_name)
    experiment = Experiment(llm_client, data_path, output_path, modification, defense_mechanism)
    experiment.run()
    print(f"Experiment finished. Results saved to {output_path}")
