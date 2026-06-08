import time
from typing import Any

import openai
from openai import OpenAI

from classes import MessageList, SamplerBase

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI."
)

BASE_URL = ""
API_KEY = ""


class ChatCompletionSampler(SamplerBase):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        base_url: str = BASE_URL,
        api_key: str = API_KEY,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list

        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content or ""

            except openai.BadRequestError as e:
                print("Bad Request Error:", e)
                return ""

            except Exception as e:
                exception_backoff = min(2**trial, 60)
                print(
                    f"Request failed, retry {trial} after {exception_backoff} sec:",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1