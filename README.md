# AIOnSite — Sovereign On-Premise Agentic AI Workbench

A privacy-preserving, locally deployable agentic AI backend. Submit a
complex task and the system plans it, builds an execution DAG, routes each
step to a model provider, runs agents and deterministic tools, verifies the
result, and produces a final answer with a complete audit trail.

The core design rule: **agent logic is never coupled to a model provider.**
Configuration selects the provider; the same pipeline runs on OpenAI,
Ollama, vLLM, or any OpenAI-compatible local server — or fully offline with
the deterministic `echo` provider.

---

## Architecture

```
User / API
   │
   ▼
Orchestrator ── policy check (sovereign / confidential → local-only)
   │
   ▼
Planner  ──────────────►  Plan (validated Pydantic; never trusts raw LLM JSON)
   │
   ▼
Pipeline Builder ───────►  Pipeline DAG (nodes, edges, deps, retry, timeout)
   │
   ▼
Pipeline Executor  ── ModelRouter picks provider/model per node
   │                  ├─ layer-by-layer, parallel within a layer
   │                  ├─ bounded retries + timeouts
   │                  └─ Agents (LLMAgent) run a bounded tool-use loop
   │                        └─ Tool Registry (typed, permissioned, logged)
   ▼
Verifier (terminal node) ─►  VerificationResult (rule-based + optional LLM)
   │
   ▼
Final Answer  +  Audit Trail (append-only, secret-redacted)
```

Provider abstraction:

```
Settings ─► create_llm() ─► BaseLLM ─► {OpenAI-compatible | Ollama | vLLM | local | echo}
```

## Directory structure

```
src/
  core/            config.py (Pydantic Settings), logging.py (structlog), exceptions.py
  llm/             base.py (BaseLLM), factory.py, schemas.py,
                   openai_provider.py, ollama_provider.py, vllm_provider.py,
                   local_provider.py, echo_provider.py
  agents/          base_agent.py (Agent / LLMAgent), roles.py, planner.py,
                   router.py (ModelRouter), inspection_agent.py
  pipeline/        models.py, graph.py (DAG), builder.py, executor.py, state.py
  tools/           base_tool.py, registry.py, builtin/ (calculator, deviation,
                   text_stats, json_parse, read_file)
  verification/    schemas.py, verifier.py
  memory/          base.py (MemoryStore), in_memory.py, memory_manager.py
  audit/           events.py, trail.py
  api/             app.py (FastAPI), schemas.py
  orchestrator.py  wires the whole flow
  main.py          CLI entry point
configs/           settings.yaml, models.yaml, tools.yaml  (non-secret policy)
scripts/           run_demo.py
tests/             mirrors src/  (84 tests, no network / keys required)
```

## Installation

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# POSIX:    source .venv/bin/activate
pip install -r requirements.txt
```

## Environment setup

```bash
cp .env.example .env      # then edit
```

Key variables (all validated at startup):

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `openai` \| `local` \| `ollama` \| `vllm` \| `echo` |
| `LLM_MODEL` | model name (e.g. `qwen2.5:7b-instruct`) |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | generation params |
| `OLLAMA_BASE_URL` | default `http://localhost:11434` |
| `VLLM_BASE_URL` / `LOCAL_LLM_BASE_URL` | OpenAI-compatible endpoints |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | cloud (refused under sovereign mode) |
| `SOVEREIGN_MODE` | `true` ⇒ cloud providers rejected, all inference stays local |
| `MAX_RETRIES` / `MAX_REPLANS` / `NODE_TIMEOUT_SECONDS` | orchestration limits |

Secrets live only in `.env` (git-ignored). `configs/*.yaml` holds
non-secret behaviour/policy only.

## Running a local model with Ollama

```bash
# install Ollama (https://ollama.com), then:
ollama serve                       # start the local server
ollama pull qwen2.5:7b-instruct    # or any model you prefer
```

```bash
# .env
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b-instruct
OLLAMA_BASE_URL=http://localhost:11434
SOVEREIGN_MODE=true
```

If Ollama is not running or the model is missing, the provider returns a
clear, actionable error (`ollama serve` / `ollama pull ...`) instead of a
stack trace.

vLLM / other OpenAI-compatible servers: set `LLM_PROVIDER=vllm` (or
`local`) and point `VLLM_BASE_URL` / `LOCAL_LLM_BASE_URL` at the endpoint.

## Commands

```bash
python -m src.main                 # print resolved config + provider health
python -m src.main --task "Analyze V-101 ..."      # run one task, print JSON
python -m src.main --task "..." --confidential     # force local-only routing
python -m src.main --serve --port 8000             # run the HTTP API

python scripts/run_demo.py                          # full end-to-end demo (offline)
LLM_PROVIDER=ollama python scripts/run_demo.py      # demo against a real local model

pytest -q                                           # run the test suite
ruff check src/                                     # lint
```

## HTTP API

```
GET  /health
POST /tasks                  {"task": "...", "confidential": false}
GET  /tasks/{execution_id}
GET  /tasks/{execution_id}/pipeline     # DAG: nodes, edges, status, provider, tools
GET  /tasks/{execution_id}/audit        # ordered audit events
GET  /tasks                             # execution ids
```

Example:

```bash
python -m src.main --serve &
curl -s localhost:8000/health
curl -s -X POST localhost:8000/tasks \
  -H 'content-type: application/json' \
  -d '{"task":"Compare observed pressure 120 bar against approved limit 100 bar and recommend."}'
```

The `/pipeline` response is machine-readable and complete enough for a
frontend pipeline visualiser (node types, dependencies, live status,
provider/model, tools used, timings, intermediate results, verification,
errors, retries) — no log scraping required.

## Security & privacy model

- No hard-coded secrets; `.env` is git-ignored; secrets are redacted from
  logs and audit metadata.
- **Sovereign mode** (`SOVEREIGN_MODE=true`): cloud providers are rejected
  at config load *and* at routing time; every execution records a
  `policy_enforced` audit event. A per-task `confidential=true` forces
  local-only routing even when sovereign mode is off.
- Tools are typed and permissioned; the registry refuses tools needing
  permissions a deployment has not granted. The filesystem tool is
  sandboxed to `data/` with traversal protection. No unrestricted shell
  execution. The calculator uses an AST walk, not `eval`.
- Structured LLM output (plans, verdicts) is validated with Pydantic and
  graph checks; malformed output triggers a bounded retry then a
  deterministic fallback.
- Retries, replans and node timeouts are all bounded — no infinite loops.

## Extending

- **New provider**: add a `BaseLLM` subclass in `src/llm/`, wire one branch
  in `factory.py`, add an enum value in `core/config.py`. No agent changes.
- **New tool**: subclass `BaseTool` with Pydantic `InputModel`/`OutputModel`
  and `permissions`, add it to `src/tools/builtin/__init__.py`.
- **New agent role**: add a prompt in `agents/roles.py`.
- **Persistent memory / vector store**: implement `MemoryStore`.
- **Durable audit sink**: pass `on_event` to `AuditTrail`.
