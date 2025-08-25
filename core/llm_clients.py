from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import ollama
import vllm
import openai
import os
from vllm.entrypoints.openai.tool_parsers import Hermes2ProToolParser, Llama3JsonToolParser
from vllm.reasoning import Qwen3ReasoningParser
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import base64
from azure.identity import (
    AuthenticationRecord, get_bearer_token_provider, DeviceCodeCredential, TokenCachePersistenceOptions, DefaultAzureCredential)
from openai import AzureOpenAI
import backoff

# TODO: standardise tool call response format so it matches OpenAI's format
class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]|None, temperature: float = 0.0, seed: int = None) -> Dict[str, Any]:
        """
        Invoke the LLM with a list of messages and a list of tools.

        :param messages: The chat history/messages to send to the LLM.
        :param tools: A list of tools available to the LLM.
        :param temperature: Temperature for sampling.
        :param seed: Random seed for reproducibility.
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

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)

    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]|None, temperature: float = 0.0, seed: int = None) -> Dict[str, Any]:
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

    def __init__(self, model: str, base_url: str = None):
        self.model = model
        self.base_url = base_url
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

    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]|None, temperature: float = 0.0, seed: int = None) -> Dict[str, Any]:
        """
        Invoke the VLLM model with a list of messages and a list of tools.
        """

        # Sampling parameters are set by default, but need max_tokens to be set to generate long descriptions
        sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.05, max_tokens=1024, seed=seed)
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
                extracted_tools = self.tool_parser.extract_tool_calls(content, None)
                content = extracted_tools.content
                tool_calls = [{'function': {'name': tool.function.name, 'arguments': tool.function.arguments}} for tool in extracted_tools.tool_calls]
        

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

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", api_key: str = "NO KEY NEEDED LOL"):
        self.model = model
        self.base_url = base_url
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=api_key
        )

    @backoff.on_exception(backoff.expo, openai.APIStatusError, max_tries=5)
    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]|None, temperature: float = 0.0, seed: int = None) -> Dict[str, Any]:
        """
        Invoke the OpenAI Proxy model with a list of messages and a list of tools.
        """
        kwargs = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
        }
        if seed is not None:
            kwargs["seed"] = seed
            
        response = self.client.chat.completions.create(**kwargs).model_dump()
        return response['choices'][0]
    
    def wait_for_server_to_start(self, timeout: int = 600):
        """
        Wait for the server to start.
        """
        start_time = time.time()
        print(f"Waiting for server to start... {self.base_url}")
        while True:
            try:
                # some random API call to check if the server is up
                self.client.models.list() 
                break
            except Exception as e:
                if "Connection error." not in str(e):
                    raise
                time.sleep(0.5)
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Server did not start within {timeout} seconds")

class GeminiClient(LLMClient):
    """LLM client for Google Gemini API."""

    def __init__(self, model: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/", api_key: str = None):
        if api_key is None:
            raise ValueError("API key is required for Gemini client")
        self.model = model
        self.base_url = base_url
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=api_key
        )

    @backoff.on_exception(backoff.expo, openai.APIStatusError, max_tries=5)
    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]|None, temperature: float = 0.0, seed: int = None) -> Dict[str, Any]:
        """
        Invoke the Gemini model with a list of messages and a list of tools.
        Note: Gemini does not support the seed parameter, so it's ignored.
        """
        kwargs = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
        }
            # Note: Gemini does not support the seed parameter, so it's ignored.
            
        response = self.client.chat.completions.create(**kwargs).model_dump()
        return response['choices'][0]
    
    def wait_for_server_to_start(self, timeout: int = 600):
        """
        Wait for the server to start.
        For Gemini, we just do a quick health check.
        """
        start_time = time.time()
        print(f"Checking Gemini API availability... {self.base_url}")
        while True:
            try:
                # Test API call to check if the server is up
                self.client.models.list() 
                break
            except Exception as e:
                if "Connection error." not in str(e):
                    raise
                time.sleep(0.5)
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Gemini API did not respond within {timeout} seconds")




def get_credential() -> DeviceCodeCredential:
    lib_name = "xyz"

    auth_record_root_path = os.environ.get("localxyz", os.path.expanduser("~"))

    auth_record_path = os.path.join(auth_record_root_path, lib_name, "auth_record.json")

    cache_options = TokenCachePersistenceOptions(name=f"{lib_name}.cache", allow_unencrypted_storage=True)

    if os.path.exists(auth_record_path):
        with open(auth_record_path, "r") as f:
            record_json = f.read()

        deserialized_record = AuthenticationRecord.deserialize(record_json)
        credential = DeviceCodeCredential(cache_persistence_options=cache_options)
    else:
        os.makedirs(os.path.dirname(auth_record_path), exist_ok=True)
        credential = DeviceCodeCredential(cache_persistence_options=cache_options)
        record_json = credential.authenticate().serialize()
        with open(auth_record_path, "w") as f:
            f.write(record_json)
    print("Authentication successful.")
    return credential


class ChatGPTAzureClient(LLMClient):
    def __init__(self, model, api_version, azure_endpoint) -> None:
        # Extracts temperature and max_tokens from the provided gen_config
        self.model = model  # Model name, e.g., "gpt-4"

        # Assuming get_bearer_token_provider, get_credential, and AzureOpenAI are defined elsewhere
        token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

        # Initialize the Azure OpenAI client with provided configurations
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_deployment=self.model,
            azure_ad_token_provider=token_provider
        )
    
    @backoff.on_exception(backoff.expo, (openai.APIStatusError, openai.RateLimitError),max_time=90)
    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]|None, temperature: float = 0.0, seed: int = None) -> Dict[str, Any]:
        """
        Invoke the Azure OpenAI model with a list of messages and a list of tools.
        """
            # merge self.gen_config with kwargs
        kwargs = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
        }
            
        if seed is not None:
            kwargs["seed"] = seed
            
        response = self.client.chat.completions.create(**kwargs).model_dump()
        return response['choices'][0]
        



def ollama_to_openai(messages):
    return messages


def make(api_version,deployment,azure_endpoint):
    config = {
        'provider': 'azure',
        'model': deployment,
        'gen_config': {
            'max_tokens': 1000,
        },
        'azure': {
            'api_version': api_version,
            'azure_deployment': deployment,
            'azure_endpoint': azure_endpoint,
        },
    }

    provider = config['provider']
    model = config['model']
    gen_config = config['gen_config']
    if provider == 'azure':

        return ChatGPTAzure(
            model=model,
            gen_config=gen_config,
            config=config['azure']
        )

class HFLocalClient(LLMClient):
    """
    Minimal local HF client that mirrors your server client's .invoke() signature:
    returns {'message': {'content': <str>}}.
    Loads model fully on GPU (no CPU offload).
    """
    def __init__(
        self,
        model: str,
        base_url: str = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        max_model_len: Optional[int] = None,
        attn_impl: Optional[str] = "flash_attention_2",  # if available for your GPU
        trust_remote_code: bool = False,
        compile_model: bool = False,   # optional: torch.compile for extra speed
        gen_cfg: Optional[GenerationConfig] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        # ensure pad token exists for batching/truncation
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # breakpoint()
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "cuda" if device == "cuda" else device,  # keeps all weights on GPU
            "trust_remote_code": trust_remote_code,
        }
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs).eval()

        # Optional compile (PyTorch 2.3+), helps some GPUs/models
        if compile_model:
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

        # effective max length
        self.max_model_len = max_model_len or getattr(self.model.config, "max_position_embeddings", 8192)

        # default generation config; can be overridden per-call
        self.gen_cfg = gen_cfg or GenerationConfig(
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # minor speedups
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        # If the tokenizer provides a chat template, use it; otherwise, do a simple join.
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback basic formatting
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
            else:
                parts.append(f"<|user|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def invoke(
        self,
        messages: List[Dict[str, str]],
        _unused=None,
        temperature: float = 0.0,
        seed: int = None,
        *,
        stop: Optional[List[str]] = None,
        generation_overrides: Optional[dict] = None,
    ) -> Dict[str, Any]:
        prompt = self._messages_to_prompt(messages)

        # Tokenize to CUDA with truncation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_model_len,
        ).to("cuda")

        gen_cfg = self.gen_cfg
        if generation_overrides:
            gen_cfg = GenerationConfig(**{**gen_cfg.to_dict(), **generation_overrides})

        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=self.model.dtype):
            out = self.model.generate(
                **inputs,
                max_new_tokens=gen_cfg.max_new_tokens,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                do_sample=gen_cfg.do_sample,
                repetition_penalty=gen_cfg.repetition_penalty,
                eos_token_id=gen_cfg.eos_token_id,
                pad_token_id=gen_cfg.pad_token_id,
            )

        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Apply optional stop strings client-side
        if stop:
            cut = len(text)
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    cut = min(cut, idx)
            text = text[:cut]

        return {"message": {"content": text}}