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
from src.chat import answer
from src.app_context import Profile
from src.startup import bootstrap


log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


async def chat(question: str | None = None) -> None:
    """
    Interactive CLI chat against the agent. With a question, answers it once and exits.

    Requires main to be run with command line argument 'chat'.
    """
    ctx = bootstrap(Profile.CHAT)
    _, memory = ctx.conversations.get("cli")

    async def ask(message: str) -> str:
        decision, result = await answer(ctx, message, memory, cfg.max_iterations)
        cited = sorted({s.metadata.get("source_path") or s.metadata.get("stream_id") or s.doc_id or "?"
                        for s in result.sources})
        route = f"[route: {decision.route}{' (fallback)' if decision.fallback else ''}] " if decision else ""
        return route + result.response + (f"\n  [sources: {', '.join(cited)}]" if cited else "")

    if question:
        print(await ask(question))
        return

    print("Chat started. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input:
            print(f"Agent: {await ask(user_input)}\n")


def ingest(collections: list[str] | None, reindex: bool, include_manual: bool) -> None:
    """Ingest new files into the vector store (all collections unless some are named) and exit."""
    ctx = bootstrap(Profile.INGEST, collections=collections, reindex=reindex, include_manual=include_manual)
    for handle in ctx.collections.values():
        print(f"{handle.name}: {handle.count()} chunk(s) indexed from {handle.settings.raw_dir}")


def main() -> None:
    """
    Main access function. Can alternatively be launched through fastapi_app.py if just interested in launching server.
    Both entry points will process raw data as necessary.

    Subcommands:
    serve:   starts the FastAPI server (default). The server reloads on code changes when the
                environment mode is set to 'dev'.
    chat:    allows the user to chat with the RAG agent directly without utilizing FastAPI endpoints.
    ingest:  builds the vector database, which persists locally, and quits without launching the RAG agent.
    """
    parser = argparse.ArgumentParser(description="LLM Agent")
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Start the FastAPI server (default)")
    serve_p.add_argument("--host", default=DEFAULT_HOST)
    serve_p.add_argument("--port", type=int, default=DEFAULT_PORT)

    chat_p = sub.add_parser("chat", help="Interactive CLI chat")
    chat_p.add_argument("-q", "--question", help="Ask one question, print the answer and exit")

    ingest_p = sub.add_parser("ingest", help="Ingest new documents without starting the server")
    ingest_p.add_argument(
        "-c", "--collection", action="append", dest="collections", metavar="NAME",
        help="Only ingest this collection (repeatable). Default: all collections.",
    )
    ingest_p.add_argument(
        "--reindex", action="store_true",
        help="Delete the collection(s) and re-embed every file and stream. Needed after changing the "
             "embedding model, chunk settings or distance metric.",
    )
    ingest_p.add_argument(
        "--manual", action="store_true", dest="include_manual",
        help="Also embed new/changed files in folders marked {\"_mode\": \"manual\"}.",
    )
    args = parser.parse_args()

    if args.command in ("serve", None):
        uvicorn.run(
            "src.fastapi_app:app",
            host=getattr(args, "host", DEFAULT_HOST),
            port=getattr(args, "port", DEFAULT_PORT),
            reload=cfg.debug,
        )
    elif args.command == "chat":
        asyncio.run(chat(args.question))
    elif args.command == "ingest":
        ingest(args.collections, args.reindex, args.include_manual)

if __name__ == "__main__":
    main()
