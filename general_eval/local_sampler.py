from typing import Any
import json

from classes import MessageList, SamplerBase
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class LocalSampler(SamplerBase):

    def __init__(
        self,
        model: str,
        system_message: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        ngpu: int = 1,
        seed: int | None = None,
        top_p: float = 0.95,
        top_k: int = 20,
        context_length: int = 20480,
        gpu_memory_utilization: float = 0.90,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        repetition_penalty: float = 1.2,
    ):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ngpu = ngpu
        self.seed = seed
        self.top_p = top_p
        self.top_k = top_k
        self.context_length = context_length
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.repetition_penalty = repetition_penalty

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=self.trust_remote_code,
        )
        self.model_engine = LLM(
            model=self.model,
            tensor_parallel_size=self.ngpu,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            max_model_len=self.context_length,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def _apply_template(self, message_list: MessageList) -> str:
        if getattr(self.tokenizer, "chat_template", None) is None:
            raise NotImplementedError("Tokenizer chat template is None")
        return self.tokenizer.apply_chat_template(
            message_list,
            tokenize=False,
            add_generation_prompt=True,
        )

    def __call__(self, message_lists: MessageList, file_path: str, seed: int | None = None):
        if self.system_message:
            for i in range(len(message_lists)):
                message_lists[i] = [self._pack_message("system", self.system_message)] + message_lists[i]

        prompts = [self._apply_template(message_list) for message_list in message_lists]

        current_seed = seed if seed is not None else self.seed
        sampling_params_kwargs = dict(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else None,
            repetition_penalty=self.repetition_penalty,
            top_p=self.top_p,
        )
        # top_k <= 0 means disable for vLLM.
        if self.top_k is not None and self.top_k > 0:
            sampling_params_kwargs["top_k"] = self.top_k
        if current_seed is not None:
            # Supported by recent vLLM versions; harmless if unavailable would raise early.
            sampling_params_kwargs["seed"] = current_seed
        sampling_params = SamplingParams(**sampling_params_kwargs)
        print("[LocalSampler/vLLM] sampling_params=", sampling_params)

        outputs = self.model_engine.generate(prompts, sampling_params)

        res_data = []
        gen_results = []
        for message_list, prompt, output in zip(message_lists, prompts, outputs):
            text = output.outputs[0].text.strip() if output.outputs else ""
            data = {
                "instruction": message_list[0]["content"],
                "prompt": prompt,
                "response": text,
            }
            res_data.append(data)
            gen_results.append(text)

        meta_data = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_message": self.system_message,
            "seed": current_seed,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "context_length": self.context_length,
            "backend": "vllm",
            "model_path": self.model,
        }

        saved_data = {"meta_info": meta_data, "data": res_data}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(saved_data, f, indent=4, ensure_ascii=False)
        print("Gen Results saved to", file_path)

        return gen_results
