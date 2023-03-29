from langchain.llms.base import LLM
from llama_index import PromptHelper
from transformers import pipeline as Pipeline
from typing import Optional, List, Mapping, Any
import torch


class CustomLLM(LLM):

    model: str
    pipeline: Any
    out_tokens: int

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = self.pipeline(prompt, max_new_tokens=self.out_tokens)[0]["generated_text"]

        # only return newly generated tokens
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model}

    @property
    def _llm_type(self) -> str:
        return "custom"

