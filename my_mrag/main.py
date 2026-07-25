import argparse

from src.config import get_settings
from src.llm import DeepSeekClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal DeepSeek chat example")
    parser.add_argument("--question", required=True, help="要发送给模型的问题")
    args = parser.parse_args()

    settings = get_settings()
    client = DeepSeekClient(settings)
    print(client.answer(args.question))


if __name__ == "__main__":
    main()
