#!/usr/bin/env bash
# Start a local llama.cpp server in ROUTER MODE: one process, one port, that routes each request
# to the model named in its `model` field, loading models on demand and keeping up to
# --models-max resident. This is the serving shape the model pool targets (chat + embeddings +
# rerank from one endpoint).
#
# Place your GGUF model files (or a llama.cpp models cache) in the --models-dir. The names the
# personas use in models.toml must match what the router exposes for those files.
#
# Usage: tools/serve_models.sh [--models-dir DIR] [--port N] [--host H] [--models-max N] [-- args]
#   --models-dir DIR   directory of GGUF models the router may load (default: ./models)
#   --port N           port to listen on (default: 8080)
#   --host H           address to bind (default: 127.0.0.1)
#   --models-max N     max models kept resident before LRU eviction (default: 4)
#   -- args            passed verbatim to llama-server (GPU offload, context size, etc.)

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

models_dir="${REPO_ROOT}/models"
port=8080
host="127.0.0.1"
models_max=4
passthrough=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models-dir) models_dir="${2:?--models-dir needs a value}"; shift 2 ;;
        --port) port="${2:?--port needs a value}"; shift 2 ;;
        --host) host="${2:?--host needs a value}"; shift 2 ;;
        --models-max) models_max="${2:?--models-max needs a value}"; shift 2 ;;
        --) shift; passthrough=("$@"); break ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

[[ "$port" =~ ^[1-9][0-9]*$ ]] || die "--port must be a positive integer, got '${port}'"
[[ "$models_max" =~ ^[1-9][0-9]*$ ]] || die "--models-max must be a positive integer, got '${models_max}'"
require_command llama-server "Build llama.cpp (the llama-server binary) and put it on PATH. See https://github.com/ggml-org/llama.cpp"
[[ -d "$models_dir" ]] || die "models directory not found: ${models_dir}. Create it and put GGUF files there, or pass --models-dir."

note "router mode on http://${host}:${port}/v1  (models from ${models_dir}, up to ${models_max} resident)"
note "point the config at:  [endpoint.local] base_url = \"http://${host}:${port}/v1\""
# Launched with NO model = router mode. Reranking needs a reranker model plus, per-model,
# --embedding --pooling rank; supply those via a --models-preset INI after -- if you serve one.
exec llama-server --host "$host" --port "$port" --models-dir "$models_dir" \
    --models-max "$models_max" --jinja "${passthrough[@]}"
