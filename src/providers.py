from enum import StrEnum

class LLMProvider(StrEnum):
    OPENAI     = "openai"
    ANTHROPIC  = "anthropic"
    GEMINI     = "gemini"
    OLLAMA     = "ollama"

class EmbeddingProvider(StrEnum):
    HUGGINGFACE = "huggingface"
    OLLAMA      = "ollama"