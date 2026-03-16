from __future__ import annotations

import os
from llama_index.llms.google_genai import GoogleGenAI


def build_gemini_llm(model: str, api_key: str | None = None) -> GoogleGenAI:
    """Builds a gemini model for use with llamaindex rag. Requires API key for google to be set in .env file.

    Args:
        model (str): The model/version of gemini to use.
        api_key (str | None, optional): Your private API key for Google gemini. Costs dollars. Defaults to None.

    Raises:
        ValueError: If no API key provided.

    Returns:
        GoogleGenAI: An LLM that operates relying on gemini's APIs.
    """
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for Gemini provider. Set in your .env file or pass as an argument.")
    try:
        return GoogleGenAI(model=model, api_key=api_key)
    except Exception as e:
        raise ValueError(f"Failed to initialize Gemini LLM: {e}") from e
