# Zenno — Documentacao Tecnica

## Visao Geral

**Zenno** e um proxy inteligente para LLMs que expoe uma API compativel com OpenAI, utilizando LangGraph para roteamento dinamico de requisicoes entre modelos locais via Ollama.

- **Versao da aplicacao:** 1.0.0
- **Linguagem:** Python 3.12+
- **Arquitetura:** FastAPI + LangGraph (State Graph)
- **Backend LLM:** Ollama (local)

---

## Dependencias

### Declaradas (`requirements.txt`)

| Pacote | Versao Minima |
|---|---|
| `fastapi` | (sem versao fixa) |
| `uvicorn[standard]` | (sem versao fixa) |
| `langgraph` | >= 0.2.0 |
| `langchain` | >= 0.3.0 |
| `langchain-openai` | >= 0.2.0 |
| `openai` | >= 1.0.0 |
| `python-dotenv` | (sem versao fixa) |
| `pydantic-settings` | (sem versao fixa) |

### Instaladas (ambiente `.venv`)

#### Dependencias Principais

| Pacote | Versao Instalada | Uso no Projeto |
|---|---|---|
| `fastapi` | 0.136.1 | Framework web, rotas HTTP, StreamingResponse |
| `uvicorn` | 0.46.0 | ASGI server (hot-reload em dev) |
| `starlette` | 1.0.0 | Base do FastAPI (rotas, middleware) |
| `pydantic` | 2.13.4 | Validacao de dados, model_dump() |
| `pydantic-settings` | 2.14.1 | Carregamento de configuracao via .env |
| `pydantic_core` | 2.46.4 | Core do Pydantic (dependencia) |
| `langgraph` | 1.2.0 | StateGraph, compilacao do workflow |
| `langgraph-checkpoint` | 4.1.0 | Checkpointing de estado do graph |
| `langgraph-prebuilt` | 1.1.0 | Componentes pre-construidos do LangGraph |
| `langgraph-sdk` | 0.3.14 | SDK para interacao com LangGraph |
| `langchain` | 1.3.0 | Framework LLM (dependencia transitiva) |
| `langchain-core` | 1.4.0 | Core abstractions do LangChain |
| `langchain-openai` | 1.2.1 | Integracao OpenAI do LangChain |
| `langchain-protocol` | 0.0.15 | Protocolos do LangChain |
| `langsmith` | 0.8.4 | Observabilidade/tracing (transitivo) |
| `openai` | 2.36.0 | Cliente async para Ollama (API compativel) |
| `python-dotenv` | 1.2.2 | Carregamento de variaveis de ambiente |

#### Dependencias de HTTP / Rede

| Pacote | Versao Instalada | Uso |
|---|---|---|
| `httpx` | 0.28.1 | HTTP client async (OpenAI SDK) |
| `httpcore` | 1.0.9 | Transport layer do httpx |
| `h11` | 0.16.0 | HTTP/1.1 protocol (httpcore) |
| `certifi` | 2026.4.22 | CA certificates bundle |
| `charset-normalizer` | 3.4.7 | Deteccao de charset HTTP |
| `idna` | 3.15 | Internationalized domain names |
| `urllib3` | 2.7.0 | HTTP client (requests) |
| `requests` | 2.34.1 | HTTP client sync (transitivo) |
| `requests-toolbelt` | 1.0.0 | Extras para requests (transitivo) |
| `sniffio` | 1.3.1 | Deteccao de async library |
| `anyio` | 4.13.0 | Async I/O abstraction |

#### Dependencias de Servidor / Streaming

| Pacote | Versao Instalada | Uso |
|---|---|---|
| `httptools` | 0.7.1 | HTTP parser (uvicorn standard) |
| `websockets` | 16.0 | WebSocket support (uvicorn standard) |
| `watchfiles` | 1.1.1 | File watcher (uvicorn reload) |
| `click` | 8.3.3 | CLI framework (uvicorn) |
| `colorama` | 0.4.6 | Color output no Windows |

#### Dependencias de Serializacao / Utilidade

| Pacote | Versao Instalada | Uso |
|---|---|---|
| `orjson` | 3.11.9 | JSON serializer rapido (LangGraph) |
| `ormsgpack` | 1.12.2 | MessagePack serializer (LangGraph) |
| `PyYAML` | 6.0.3 | YAML parser (LangChain) |
| `jsonpatch` | 1.33 | JSON Patch (LangChain) |
| `jsonpointer` | 3.1.1 | JSON Pointer (jsonpatch dep) |
| `jiter` | 0.14.0 | JSON parser em Rust (OpenAI SDK) |
| `uuid_utils` | 0.15.0 | UUID utilities (LangGraph) |
| `xxhash` | 3.7.0 | Hashing rapido (LangGraph) |
| `zstandard` | 0.25.0 | Compressao (LangGraph checkpoint) |

#### Dependencias de LLM / Tokens

| Pacote | Versao Instalada | Uso |
|---|---|---|
| `tiktoken` | 0.12.0 | Tokenizer OpenAI (OpenAI SDK) |
| `regex` | 2026.5.9 | Regex avancado (tiktoken) |

#### Dependencias de Resiliencia / Utilidade

| Pacote | Versao Instalada | Uso |
|---|---|---|
| `tenacity` | 9.1.4 | Retry library (LangChain) |
| `tqdm` | 4.67.3 | Progress bars (transitivo) |
| `distro` | 1.9.0 | Detect OS distro (OpenAI SDK) |
| `packaging` | 26.2 | Version parsing |
| `typing_extensions` | 4.15.0 | Type hints backport |
| `typing-inspection` | 0.4.2 | Type introspection (Pydantic) |
| `annotated-types` | 0.7.0 | Annotated types (Pydantic) |
| `annotated-doc` | 0.0.4 | Doc annotations (Pydantic) |

#### Build / Package Manager

| Pacote | Versao Instalada |
|---|---|
| `pip` | 26.0.1 |

---

## Estrutura do Projeto

```
zenno/
├── main.py                  # Entry point: FastAPI app + uvicorn
├── requirements.txt         # Declaracao de dependencias
├── .env.example             # Template de variaveis de ambiente
├── api/
│   ├── __init__.py
│   └── routes.py            # Rotas HTTP (/chat/completions, /models)
├── core/
│   ├── __init__.py
│   ├── config.py            # Configuracao via pydantic-settings + .env
│   ├── classifier.py        # Heuristic rule-based domain/complexity classifier
│   ├── domains.py           # Per-domain system prompt + temperature + samples + verify
│   └── sandbox.py           # Python sandbox executor + final-answer extractor
├── graph/
│   ├── __init__.py
│   ├── state.py             # GraphState (TypedDict)
│   ├── nodes.py             # Nodes do LangGraph (classify, direct, reasoning, verify, evaluate, stream_final)
│   └── workflow.py          # StateGraph builder + compilacao
└── .venv/                   # Virtual environment
```

---

## Arquitetura

### Fluxo de Requisicao

```
Cliente (OpenAI SDK)
        │
        ▼
┌─────────────────────┐
│   FastAPI Router    │  POST /v1/chat/completions
│   (api/routes.py)   │  GET  /v1/models
│                     │  GET  /health
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   LangGraph         │
│   (workflow.py)     │  StateGraph com 5 nodes:
└─────────┬───────────┘  ┌──────────────┐
          │              │  classify    │ ← Classifica simple vs complex
          ▼              ├──────────────┤
    ┌───────────┐        │   direct     │ ← Resposta direta (Ollama)
    │  tools?   │───────▶├──────────────┤
    └─────┬─────┘        │  reasoning   │ ← Loop de raciocinio iterativo
          │              ├──────────────┤
          ▼              │  evaluate    │ ← Avaliacao de qualidade
    ┌───────────┐        ├──────────────┤
    │ simple?   │        │ stream_final │ ← Entrega ao cliente
    └─────┬─────┘        └──────────────┘
    Yes /  \ No
     ▼       ▼
  direct  reasoning → evaluate → (loop ou stream_final)
```

### Nodes do LangGraph

| Node | Funcao | Arquivo |
|---|---|---|
| `classify` | Heuristica rule-based primeiro (`core/classifier.py`) + fallback LLM. Decide *complexity* e *domain* (math/code/logic/plan/factual/conversation/complex). | `graph/nodes.py` |
| `direct` | Resposta direta do Ollama, com system prompt do dominio aplicado (factual/conversation). Suporta tools e streaming. | `graph/nodes.py` |
| `reasoning` | Iteracao 0: amostra N candidatos em paralelo (self-consistency) com prompt e temperatura do dominio (`core/domains.py`), sintetiza via majority voting (math) / presenca de bloco python (code) / longest (default). Iteracao 1+: refinamento single-shot com critique + saida do sandbox. | `graph/nodes.py` |
| `verify`   | Quando o dominio pede (`math`, `code`), extrai bloco Python da resposta e executa em subprocess isolado (`core/sandbox.py`). Resultado vira ground-truth para o evaluator. | `graph/nodes.py` |
| `evaluate` | Se o sandbox falhou, força NEEDS_WORK com stderr como critique. Caso contrario, avaliacao LLM critica recebendo o stdout do sandbox como referencia. | `graph/nodes.py` |
| `stream_final` | Entrega resposta final ao cliente (SSE ou JSON) | `graph/nodes.py` |

### Fluxo do Reasoning

```
classify ──complex──▶ reasoning ──▶ verify ──▶ evaluate ──good──▶ stream_final
                          ▲                         │
                          └──── needs_improvement ──┘
                          (refina com critique + saida do sandbox)
```

### Modulos auxiliares

| Modulo | Responsabilidade |
|---|---|
| `core/classifier.py` | Classificador heuristico rule-based (regex por dominio). Retorna `(domain, strategy, confidence)`. Confidence baixa → fallback LLM. |
| `core/domains.py`    | Config por dominio: system prompt especializado, exploration/refinement temperatures, numero de samples, flag de verify. |
| `core/sandbox.py`    | Extracao de bloco Python, denylist estatico (os/socket/subprocess/file write), execucao em subprocess isolado (`-I`), timeout duro. Tambem `extract_final_answer` para majority voting. |

### Rotas da API

| Metodo | Endpoint | Descricao |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completions (OpenAI-compatible), suporta stream e tool calling |
| `GET` | `/v1/models` | Lista modelos disponiveis |
| `GET` | `/health` | Health check |

### Configuracao (`core/config.py`)

| Variavel | Tipo | Padrao | Descricao |
|---|---|---|---|
| `ollama_base_url` | str | `http://localhost:1234/v1` | URL base do Ollama |
| `ollama_api_key` | str | `ollama` | API key do Ollama |
| `default_model` | str | `llama3.2` | Modelo padrao |
| `max_reasoning_iterations` | int | `3` | Max de iteracoes no loop de raciocinio |
| `api_host` | str | `0.0.0.0` | Host do servidor |
| `api_port` | int | `8000` | Porta do servidor |
| `max_api_retries` | int | `3` | Max de retries em chamadas API |
| `retry_base_delay` | float | `1.0` | Delay base para retry (segundos) |
| `retry_max_delay` | float | `10.0` | Delay maximo para retry (segundos) |
| `api_request_timeout` | float | `120.0` | Timeout de requisicao API (segundos) |
| `classify_heuristic_threshold` | float | `0.7` | Confianca minima do classificador heuristico (<= cai pro LLM) |
| `self_consistency_enabled` | bool | `true` | Liga self-consistency multi-sample |
| `self_consistency_max_samples` | int | `5` | Cap global de samples por iteracao |
| `sandbox_enabled` | bool | `true` | Liga execucao de codigo em sandbox para verificacao |
| `sandbox_timeout` | float | `5.0` | Timeout duro por execucao Python (segundos) |

---

## Matriz de Compatibilidade

| Componente | Versao Minima Declarada | Versao Instalada | Status |
|---|---|---|---|
| langgraph | >= 0.2.0 | 1.2.0 | ✅ Atualizado |
| langchain | >= 0.3.0 | 1.3.0 | ✅ Atualizado |
| langchain-openai | >= 0.2.0 | 1.2.1 | ✅ Atualizado |
| openai | >= 1.0.0 | 2.36.0 | ✅ Atualizado |
| fastapi | (any) | 0.136.1 | ✅ Estavel |
| uvicorn | (any) | 0.46.0 | ✅ Estavel |
| pydantic-settings | (any) | 2.14.1 | ✅ Estavel |
| python-dotenv | (any) | 1.2.2 | ✅ Estavel |

---

## Observacoes

- **Nenhuma dependencia do `requirements.txt` possui versao maxima**, o que permite atualizacoes automaticas mas pode introduzir breaking changes em futuras instalacoes.
- **LangGraph** saltou da versao minima declarada (0.2.0) para 1.2.0 — API de StateGraph permanece compativel.
- **OpenAI SDK** na versao 2.x e usado como cliente compativel com Ollama (necessita `base_url` customizado).
- **Pydantic v2** (2.13.4) e usado via `pydantic-settings` para carregamento de configuracao.
- O projeto nao possui arquivo `pyproject.toml` ou `setup.py` — gerencia dependencias apenas via `requirements.txt`.
