import os
from openai import OpenAI as OpenAIClient
from llama_index.llms.openai import OpenAI


def loadGPT(version):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Create OpenAI client (if needed for direct API calls)
    client = OpenAIClient(api_key=api_key)
    
    # Create LlamaIndex LLM wrapper
    llm = OpenAI(model=version, api_key=api_key)
    
    return client, llm


