import os
from openai import OpenAI as keyLoader
from llama_index.llms.openai import OpenAI


def loadGPT(version):
    client = keyLoader(api_key="OPENAI_API_KEY")

    llm = OpenAI(model=version)
    return client, llm


