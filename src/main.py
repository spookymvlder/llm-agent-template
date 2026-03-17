import asyncio
import argparse
import uvicorn
import logging

import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config import CONFIG as cfg
from startup import bootstrap, ingest_only




log = logging.getLogger(__name__)


async def chat() -> None:
    """Interactive CLI chat against the agent."""
    response = bootstrap()
    agent = response.agent
    
    print("Chat started. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        response = await agent.run(user_msg=user_input, max_iterations=3)
        print(f"Agent: {response}\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Agent")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve",  help="Start the FastAPI server")
    sub.add_parser("chat",   help="Interactive CLI chat")
    sub.add_parser("ingest", help="Ingest new documents without starting the server")
    args = parser.parse_args()


    if args.command == "serve" or args.command is None:
        uvicorn.run("src.fastapi_app:app", host="0.0.0.0", port=8000, reload=cfg.debug)
    elif args.command == "chat":
        asyncio.run(chat())
    elif args.command == "ingest":
        ingest_only()

if __name__ == "__main__":
    main()