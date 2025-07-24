from abc import ABC, abstractmethod
from typing import List, Dict, Any
import ollama
import vllm

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
        self.llm = vllm.LLM(model=model, dtype="float16")

    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]|None) -> Dict[str, Any]:
        """
        Invoke the VLLM model with a list of messages and a list of tools.
        """

        # Sampling parameters are set by default
        try:    
            response =self.llm.chat(
                messages=messages,
                tools=tools,
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            return {}

        return response.model_dump()