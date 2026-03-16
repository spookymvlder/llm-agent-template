from __future__ import annotations

from llama_index.llms.ollama import Ollama


def build_ollama_llm(
    *,
    model: str,
    base_url: str,
    request_timeout_s: float,
    context_window: int,
    temperature: float,
) -> Ollama:
    """Builds a local model for use with llamaindex rag agent. Is local so no API keys are needed, but won't be as strong as 
    an API-keyed models. More args are possible, see Ollama documentation.

    Args:
        model (str): The model/version to use.
        base_url (str): The url the model is running on.
        request_timeout_s (float): Timeout limit for connecting to Ollama API.
        context_window (int): Number of tokens for model to use.
        temperature (float): Sampling temperature.

    Returns:
        Ollama: An Ollama llm for rag use.
    """
    try:
        return Ollama(
            model=model,
            base_url=base_url,
            request_timeout=request_timeout_s,
            context_window=context_window,
            temperature=temperature,
        )
    except Exception as e:
        raise ValueError(f"Failed to initialize Ollama LLM: {e}") from e
