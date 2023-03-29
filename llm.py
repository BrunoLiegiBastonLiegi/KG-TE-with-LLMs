from langchain.llms.base import LLM
from llama_index import PromptHelper
from transformers import pipeline as Pipeline
from typing import Optional, List, Mapping, Any
import torch


class CustomLLM(LLM):

    def __init__(self, model, pipeline, device, out_tokens=256):
        super().__init__()
        #self.model = model
        self.pipeline = Pipeline(
            pipeline,
            model=model,
            device=device,
            model_kwargs={"torch_dtype":torch.bfloat16}
        )
        self.out_tk

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = self.pipeline(prompt, max_new_tokens=self.out_tk)[0]["generated_text"]

        # only return newly generated tokens
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model}

    @property
    def _llm_type(self) -> str:
        return "custom"

