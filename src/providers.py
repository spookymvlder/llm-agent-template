from enum import auto, StrEnum

class LLMProvider(StrEnum):
    OPENAI     = auto()
    ANTHROPIC  = auto()
    GEMINI     = auto()
    OLLAMA     = auto()

class EmbeddingProvider(StrEnum):
    HUGGINGFACE = auto()
    OLLAMA      = auto()

# Env var holding each hosted provider's API key. Providers not listed (Ollama) need no key.
API_KEY_ENV_VARS: dict[LLMProvider, str] = {
    LLMProvider.OPENAI:    "OPENAI_API_KEY",
    LLMProvider.GEMINI:    "GOOGLE_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
}
