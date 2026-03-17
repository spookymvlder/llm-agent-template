from enum import Enum, auto, StrEnum

class LLMProvider(StrEnum):
    OPENAI     = auto()
    ANTHROPIC  = auto()
    GEMINI     = auto()
    OLLAMA     = auto()

class EmbeddingProvider(StrEnum):
    HUGGINGFACE = auto()
    OLLAMA      = auto()