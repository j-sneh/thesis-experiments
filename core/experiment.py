from typing import List, Dict, Any
from core.utils import load_data, save_results
from core.tool_modifier import duplicate_and_rename_tool, modify_tool_description, format_tool_for_openai_api
from core.llm_clients import LLMClient, OllamaClient

class Experiment:
    def __init__(self, llm_client: LLMClient, data_path: str, output_path: str, modification: str):
        self.llm_client = llm_client
        self.data = load_data(data_path)
        self.output_path = output_path
        self.modification = modification
        self.results: List[Dict[str, Any]] = []

    def run(self):
        for item in self.data:
            # print(item)
            messages = item['question'][0]
            original_tool = item['function'][0]

            # Duplicate and rename
            tool1, tool2 = duplicate_and_rename_tool(original_tool)
            
            # Modify one of the tools
            modified_tool2 = modify_tool_description(tool2, self.modification)

            tools = [format_tool_for_openai_api(tool1), format_tool_for_openai_api(modified_tool2)]

            # Invoke LLM
            response = self.llm_client.invoke(messages, tools)

            print(response)

            # Record result
            self._record_result(item, response, tools, messages)
            break #TODO: remove this

        save_results(self.output_path, self.results)

    def _record_result(self, item: Dict[str, Any], response: Dict[str, Any], tools: List[Dict[str, Any]], messages: List[Dict[str, Any]]):
        called_tool_name = None
        if response and 'message' in response and response['message']['tool_calls']:
            called_tool_name = response['message']['tool_calls'][0]['function']['name']
        
        self.results.append({
            "id": item["id"],
            "question": item["question"],
            # "messages": messages,
            "original_tool": item["function"][0],
            "tools_provided": tools,
            # "llm_response": response,
            "called_tool_name": called_tool_name,
            "modification": self.modification
        })

def run_experiment(model_name: str, data_path: str, output_path: str, modification: str):
    llm_client = OllamaClient(model=model_name)
    experiment = Experiment(llm_client, data_path, output_path, modification)
    experiment.run()
    print(f"Experiment finished. Results saved to {output_path}")
