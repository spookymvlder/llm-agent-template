# llm-agent-template

A template for setting up a quick LLM agent and Flask server, intended for use during agent hackathons.

AgentTools contains example tools that can be used by the agent. 
InitializeIndex supports loading documents in to train LLM on dedicated materials. Remove q_engine and references to it if not using.
Can be modified to load document list if necessary.

# 1. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Test agent setup
python setupAgent.py

# 4. Run web interface
python flaskApp.py

# 5. Or test via CLI
python callAgent.py



## .env example ##
# LLM Configuration
AGENT_TYPE=gemini                    # or "gpt"
MODEL_VERSION=gemini-2.0-flash-exp   # or "gpt-4o-mini"

# API Keys (choose at least one)
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here

# Storage Paths
PERSIST_PATH=./storage/outputDirectory
SOURCE_PATH=./storage/sourceFiles/example.pdf
DOC_NAME=example.pdf

# Optional Features
USE_OBSERVABILITY=true  # Set to false to disable Phoenix