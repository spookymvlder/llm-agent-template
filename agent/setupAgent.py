import os
import logging
from dotenv import load_dotenv


from initializeIndex import LoadIndex


# Configure logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


from llama_index.core.agent import ReActAgent


agent = "" #"gpt or gemini"
version = "" # "gpt-4" or "gemini-2.5-flash"

match agent:
    case "gpt":
        llm = gpt.loadGpt(version)
    case "gemini":
        llm = gemini.loadGemini(version)
    case _:
        print("No agent provided.")
        quit


import phoenix as px
session = px.launch_app()

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

tracer_provider = register()
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)



persistPath = "./storage/outputDirectory"
sourcePath = "./storage/sourceFiles/example.pdf"
docName = "example.pdf"
index = LoadIndex(persistPath, sourcePath, docName)


q_engine = index.as_query_engine(similarity_top_k=3, llm=llm)

import agentTools

agent = ReActAgent.from_tools(
    agentTools.getQueryTools(),
    llm=llm,
    verbose=True,
    max_turns=10,
)

