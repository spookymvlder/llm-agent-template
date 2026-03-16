from __future__ import annotations

import os
from llama_index.llms.openai import OpenAI


def build_openai_llm(model: str, api_key: str | None = None) -> OpenAI:
    """Builds a chatgpt model for use with llamaindex rag. Requires API key for OpenAI to be set in .env file.

    Args:
        model (str): The OpenAI model/version to use.
        api_key (str | None, optional): Your private API key for use with OpenAI. Costs dollars to use. Defaults to None.

    Raises:
        ValueError: If no API key is set.

    Returns:
        OpenAI: An LLM that operates relying on OpenAI's APIs.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI provider.  Set in your .env file or pass as an argument.")
    try:
        return OpenAI(model=model, api_key=api_key)
    except Exception as e:
        raise ValueError(f"Failed to initialize OpenAI LLM: {e}") from e
