# Third-party components

Nothing here is bundled into the repository. Python packages install from PyPI via
`tools/setup_python_env.sh` (pinned in `requirements.txt`); model files download via
`tools/fetch_models.sh`; recipe datasets download via each recipe's `fetch.sh`. This file
records what each is and under what license, so the noncommercial/commercial split in
[`README.md`](./README.md) can be relied on.

## Runtime Python dependencies

| Package | Role | License |
|---|---|---|
| `httpx` (+ `anyio`, `certifi`, `h11`, `httpcore`, `idna`) | HTTP client for the local inference server | BSD-3-Clause / MIT / MPL-2.0 (certifi: MPL-2.0) |
| `typing_extensions` | typing back-ports | PSF |

Optional feature dependencies, installed only when their feature is used and each imported
lazily inside a single driver module (added by the milestone that introduces the feature):

| Package | Feature | License |
|---|---|---|
| `numpy` | dense-vector math | BSD-3-Clause |
| `lancedb` (+ `pyarrow`) | default vector database | Apache-2.0 |
| `sqlite-vec` | alternative in-SQLite vector driver | Apache-2.0 OR MIT |
| `qdrant-client` | alternative vector driver | Apache-2.0 |
| `psycopg` / `pgvector` | Postgres record + vector driver (production tier) | LGPL-3.0 / PostgreSQL |
| `bm25s` | alternative lexical index | MIT |
| `SQLAlchemy` | optional schema introspector | MIT |

## Development dependencies

`pytest`, `coverage`, `hypothesis`, `ruff`, `mypy` and their transitive dependencies —
all MIT / BSD / Apache-2.0 / PSF. Test-only; nothing under `src/` imports them.

## Default models (downloaded, not bundled)

Chosen commercially clean so the dual license is not undermined by a gated or
non-commercial model. Each is a separate work under its own license.

| Model | Role | License |
|---|---|---|
| `Qwen3.5-2B-Instruct` | default producer persona | Apache-2.0 |
| `Qwen3.5-0.8B-Instruct` | default reviewer personas | Apache-2.0 |
| `Qwen3-0.6B` | CI / smoke (CPU-runnable) | Apache-2.0 |
| `Qwen3-Embedding-0.6B` | default embeddings | Apache-2.0 |
| `bge-m3` | alternative multilingual embeddings | MIT |
| `bge-reranker-v2-m3` | default reranker | Apache-2.0 |

Deliberately **not** defaults: EmbeddingGemma / Gemma 1–3 (custom license, prohibited-use
policy, manually gated) and the Jina rerankers (CC-BY-NC-4.0, non-commercial).

## Recipe datasets (fetched on demand, never committed)

Each dataset's license governs your use of *that data*, independent of this software.

| Recipe | Dataset | License |
|---|---|---|
| translation | Tatoeba bilingual pairs; OPUS (optional) | CC-BY 2.0 FR |
| nl_to_sql | Spider; BIRD (optional) | CC BY-SA 4.0 |
| form_autofill | Chinook; MovieLens 25M (optional) | Chinook: permissive (MIT-style); MovieLens: research-use |
