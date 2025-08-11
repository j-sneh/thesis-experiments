from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time
import ollama
import vllm
import openai
from vllm.entrypoints.openai.tool_parsers import Hermes2ProToolParser, Llama3JsonToolParser
from vllm.reasoning import Qwen3ReasoningParser
import requests
import os

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

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)

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

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1",):
        self.model = model
        self.base_url = base_url
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key="no key needed thus far"
        )

    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]|None, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Invoke the OpenAI Proxy model with a list of messages and a list of tools.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=temperature,
            ).model_dump()
            return response['choices'][0]
        except Exception as e:
            print(f"An error occurred: {e}")
    
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

# pip install transformers accelerate torch --extra-index-url https://download.pytorch.org/whl/cu124
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

class HFLocalClient:
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
