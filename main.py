import logging
import os
import re

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from core.config import settings

# Pinta o nome do nó (CLASSIFY/DIRECT/REASONING/EVALUATE/STREAM_FINAL) em laranja
# pra dar pra acompanhar no terminal o caminho que o grafo realmente percorreu
# e confirmar que o ciclo reasoning↔evaluate está iterando.
if os.name == "nt":
    os.system("")  # destrava ANSI no console do Windows

_ORANGE = "\033[38;5;208m"
_RESET  = "\033[0m"
_NODE_PATTERN = re.compile(r"\] (CLASSIFY|DIRECT|REASONING|VERIFY|EVALUATE|STREAM_FINAL) \|")


class _NodeColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        return _NODE_PATTERN.sub(
            lambda m: f"] {_ORANGE}{m.group(1)}{_RESET} |", s,
        )


_handler = logging.StreamHandler()
_handler.setFormatter(_NodeColorFormatter(
    fmt="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
))
logging.basicConfig(
    level=logging.INFO,
    handlers=[_handler],
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

app = FastAPI(
    title="Zenno — Smart LLM Proxy",
    description="OpenAI-compatible API with intelligent LangGraph routing for local Ollama models.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount under /v1 (OpenAI-canonical) AND root, so agents that omit the prefix
# (Qwen Code, some old SDK builds, hand-rolled clients) still hit the routes.
app.include_router(router, prefix="/v1")
app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    from pathlib import Path
    base = Path(__file__).parent
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        # Only watch Zenno's own packages — ignores files the agent creates in the project root
        reload_dirs=[str(base / "api"), str(base / "core"), str(base / "graph")],
        reload_includes=["*.py"],
    )
