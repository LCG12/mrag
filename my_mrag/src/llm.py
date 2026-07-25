from openai import OpenAI

from .config import Settings


class DeepSeekClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )

    def answer(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个简洁、准确的中文助手。",
                },
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content or ""

