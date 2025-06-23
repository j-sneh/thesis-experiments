from typing import List, Dict, Any
from core.utils import load_data, save_results
from core.tool_modifier import duplicate_and_rename_tool, modify_tool_description, format_tool_for_openai_api
from core.llm_clients import LLMClient, OllamaClient
from tqdm import tqdm
import json

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
                tool1, tool2 = duplicate_and_rename_tool(original_tool)
                
                # Modify one of the tools
                modified_tool2 = modify_tool_description(tool2, self.modification)
                modified_tool2['modified'] = True  # Mark as modified

                # Prepare tools for LLM (remove 'modified' field if present)
                tool1_for_llm = format_tool_for_openai_api(tool1)
                tool2_for_llm = format_tool_for_openai_api({k: v for k, v in modified_tool2.items() if k != 'modified'})
                tools_for_llm = [tool1_for_llm, tool2_for_llm]

                # Invoke LLM
                response = self.llm_client.invoke(messages, tools_for_llm)

                # print(f"Trial {trial_type}: {response}")

                # Record result (store tools with 'modified' field for analysis)
                result = self._record_result(item, response, [tool1, modified_tool2], messages)
                f.write(json.dumps(result) + "\n")
                # break #TODO: remove this

    def _record_result(self, item: Dict[str, Any], response: Dict[str, Any], tools: List[Dict[str, Any]], messages: List[Dict[str, Any]]):
        called_tool_name = None
        if response and 'message' in response and response['message']['tool_calls']:
            called_tool_name = response['message']['tool_calls'][0]['function']['name']
        result = {
            "id": item["id"],
            "question": item["question"],
            # "messages": messages,
            "original_tool": item["function"][0],
            "tools_provided": tools,
            # "llm_response": response,
            "called_tool_name": called_tool_name,
            "modification": self.modification
        }
        self.results.append(result)
        return result

def run_experiment(model_name: str, data_path: str, output_path: str, modification: str, defense_mechanism: str):
    llm_client = OllamaClient(model=model_name)
    experiment = Experiment(llm_client, data_path, output_path, modification, defense_mechanism)
    experiment.run()
    print(f"Experiment finished. Results saved to {output_path}")
