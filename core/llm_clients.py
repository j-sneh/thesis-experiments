from abc import ABC, abstractmethod
from typing import List, Dict, Any
import ollama
import vllm

from vllm.entrypoints.openai.tool_parsers import Hermes2ProToolParser, Llama3JsonToolParser

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
            self.tool_parser = None
        elif model.lower().startswith("meta-llama"):
            self.tool_parser = Llama3JsonToolParser(self.tokenizer)
        else:
            self.tool_parser = None

    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]|None) -> Dict[str, Any]:
        """
        Invoke the VLLM model with a list of messages and a list of tools.
        """

        # Sampling parameters are set by default
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

        output = response[0].outputs[0].text.strip()

        print(response[0])

        tools = None
        if self.tool_parser:
            tools = self.tool_parser.extract_tool_calls(output, None)

        raise Exception(f"Tools: {tools}, Output: {output}")

        return response.model_dump()