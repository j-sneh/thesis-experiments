from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time
import ollama
import vllm
import openai
from vllm.entrypoints.openai.tool_parsers import Hermes2ProToolParser, Llama3JsonToolParser
from vllm.reasoning import Qwen3ReasoningParser

# TODO: standardise tool call response format so it matches OpenAI's format
class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]|None) -> Dict[str, Any]:
        """
        Invoke the LLM with a list of messages and a list of tools.

        :param messages: The chat history/messages to send to the LLM.
        :param tools: A list of tools available to the LLM.
        :return: The LLM's response.
        """
        pass
    
    def wait_for_server_to_start(self):
        """
        Wait for the server to start.
        """
        pass

class OllamaClient(LLMClient):
    """LLM client for Ollama."""

    def __init__(self, model: str):
        self.model = model
        self.client = ollama.Client()

    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]|None) -> Dict[str, Any]:
        """
        Invoke the Ollama model with a list of messages and a list of tools.
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=tools,
            )
            return response.model_dump()
        except Exception as e:
            print(f"An error occurred: {e}")
            return {}

class VLLMClient(LLMClient):
    """LLM client for VLLM."""

    def __init__(self, model: str):
        self.model = model
        # float16 for quantization purposes
        # TODO: look at https://github.com/vllm-project/llm-compressor for quantization

        # max_model_len was 37440 because this is the size of the KV cache
        # 32768 is used because it is the max len (for pos embeddings) in qwen2.5
        self.llm = vllm.LLM(model=model, dtype='half', max_model_len=32768)
        self.tokenizer = self.llm.get_tokenizer()



        if model.lower().startswith("qwen"):
            self.tool_parser = Hermes2ProToolParser(self.tokenizer)
            self.reasoning_parser = Qwen3ReasoningParser(self.tokenizer)
        elif model.lower().startswith("meta-llama"):
            self.tool_parser = Llama3JsonToolParser(self.tokenizer)
            self.reasoning_parser = None
        else:
            self.tool_parser = None
            self.reasoning_parser = None

    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]|None) -> Dict[str, Any]:
        """
        Invoke the VLLM model with a list of messages and a list of tools.
        """

        # Sampling parameters are set by default, but need max_tokens to be set to generate long descriptions
        sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.05, max_tokens=1024)
        try:    
            response =self.llm.chat(
                messages=messages,
                tools=tools,
                sampling_params=sampling_params
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            return {}

        content = response[0].outputs[0].text.strip()

        tool_calls = None
        reasoning = None
        if self.reasoning_parser:
            # request must be passed in, but it is not in use currently, so we pass in None -- this may break in the future
            try:
                reasoning, content = self.reasoning_parser.extract_reasoning_content(content, None)
            except Exception as e:
                print(f"An error occurred: {e}")
                reasoning = None

        if self.tool_parser:
            # request must be passed in, but it is not in use currently, so we pass in None -- this may break in the future
            try:
                extracted_tools = self.tool_parser.extract_tool_calls(content, None)
                content = extracted_tools.content
                tool_calls = [{'function': {'name': tool.function.name, 'arguments': tool.function.arguments}} for tool in extracted_tools.tool_calls]
            except Exception as e:
                print(f"An error occurred: {e}")
                tool_calls = []
        

        response = {
            'model': self.model,
            'message': {
                'role': 'assistant',
                'content': content,
                'thinking': reasoning,
                'tool_calls': tool_calls
            }
        }

        return response

class OpenAIClient(LLMClient):
    """LLM client for OpenAI Proxy."""

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1",):
        self.model = model
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key="no key needed thus far"
        )

    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]|None) -> Dict[str, Any]:
        """
        Invoke the OpenAI Proxy model with a list of messages and a list of tools.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
            ).model_dump()['choices'][0]
            return response
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def wait_for_server_to_start(self):
        """
        Wait for the server to start.
        """
        while True:
            try:
                # some random API call to check if the server is up
                self.client.models.list() 
                break
            except Exception as e:
                time.sleep(0.5)
                print(f"Waiting for server to start... {e}")