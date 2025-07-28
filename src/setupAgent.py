import os
import logging
from dotenv import load_dotenv
from initializeIndex import LoadIndex


# Configure logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


from llama_index.core.agent import ReActAgent
from llama_index.core import Settings


from models import gemini, gpt

AGENT_TYPE = os.getenv("AGENT_TYPE", "gemini")  # Can be "gpt" or "gemini"
MODEL_VERSION = os.getenv("MODEL_VERSION", "gemini-2.0-flash-exp")  # Updated model


def initialize_llm():
    """Initialize LLM based on configuration"""
    match AGENT_TYPE:
        case "gpt":
            logger.info(f"Loading GPT model: {MODEL_VERSION}")
            _, llm = gpt.loadGPT(MODEL_VERSION)  
            return llm
        case "gemini":
            logger.info(f"Loading Gemini model: {MODEL_VERSION}")
            llm = gemini.loadGemini(MODEL_VERSION) 
            return llm
        case _:
            logger.error("No valid agent type provided. Use 'gpt' or 'gemini'")
            raise ValueError(f"Unsupported agent type: {AGENT_TYPE}")

# Initialize LLM
llm = initialize_llm()
Settings.llm = llm


# Phoenix observability - optional
USE_OBSERVABILITY = os.getenv("USE_OBSERVABILITY", "true").lower() == "true"

if USE_OBSERVABILITY:
    try:
        import phoenix as px
        session = px.launch_app()
        
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        from phoenix.otel import register
        
        tracer_provider = register()
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info("Phoenix observability enabled")
    except ImportError:
        logger.warning("Phoenix not available, skipping observability")
else:
    logger.info("Observability disabled")



# Index configuration - make paths configurable
PERSIST_PATH = os.getenv("PERSIST_PATH", "./storage/outputDirectory")
SOURCE_PATH = os.getenv("SOURCE_PATH", "./storage/sourceFiles/example.pdf")
DOC_NAME = os.getenv("DOC_NAME", "example.pdf")

# Ensure directories exist
os.makedirs(os.path.dirname(PERSIST_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SOURCE_PATH), exist_ok=True)

# Load or create index
try:
    index = LoadIndex(PERSIST_PATH, SOURCE_PATH, DOC_NAME)
    logger.info("Index loaded successfully")
except Exception as e:
    logger.error(f"Failed to load index: {e}")
    # Create a dummy index for hackathon demos if files don't exist
    from llama_index.core import VectorStoreIndex, Document
    dummy_doc = Document(text="This is a sample document for demonstration purposes.")
    index = VectorStoreIndex.from_documents([dummy_doc])
    logger.info("Created dummy index for demo purposes")


q_engine = index.as_query_engine(similarity_top_k=3, llm=llm)

import agentTools

agent = ReActAgent.from_tools(
    agentTools.getQueryTools(),
    llm=llm,
    verbose=True,
    max_turns=10,
)

logger.info(f"Agent initialized with {AGENT_TYPE} ({MODEL_VERSION})")

# Convenience function for quick testing
def quick_test():
    """Quick test function for hackathon debugging"""
    try:
        response = agent.chat("Hello, can you tell me what tools you have available?")
        print(f"Agent response: {response}")
        return True
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing agent setup...")
    if quick_test():
        print("Agent setup successful!")
    else:
        print("Agent setup failed!")