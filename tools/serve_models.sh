#!/usr/bin/env bash
# Start a local llama.cpp server in ROUTER MODE: one process, one port, that routes each request
# to the model named in its `model` field, loading models on demand and keeping up to
# --models-max resident. This is the serving shape the model pool targets (chat + embeddings +
# rerank from one endpoint).
#
# Place your GGUF model files (or a llama.cpp models cache) in the --models-dir. The names the
# personas use in models.toml must match what the router exposes for those files.
#
# Usage: tools/serve_models.sh [--config FILE --endpoint NAME] [--models-dir DIR]
#                              [--port N] [--host H] [--models-max N] [-- args]
#   --config FILE      read host/port/models-max/server_args from this models.toml (one source of
#                      truth); requires --endpoint. Manual flags below override what it provides.
#   --endpoint NAME    the [endpoint.<name>] in --config to serve (default: local)
#   --models-dir DIR   directory of GGUF models the router may load (default: ./models)
#   --port N           port to listen on (default: 8080, or the config's base_url port)
#   --host H           address to bind (default: 127.0.0.1, or the config's base_url host)
#   --models-max N     max models kept resident before LRU eviction (default: 4, or the config's)
#   -- args            passed verbatim to llama-server, after any server_args from the config

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

models_dir="${REPO_ROOT}/models"
config=""
endpoint="local"
port=""
host=""
models_max=""
passthrough=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) config="${2:?--config needs a value}"; shift 2 ;;
        --endpoint) endpoint="${2:?--endpoint needs a value}"; shift 2 ;;
        --models-dir) models_dir="${2:?--models-dir needs a value}"; shift 2 ;;
        --port) port="${2:?--port needs a value}"; shift 2 ;;
        --host) host="${2:?--host needs a value}"; shift 2 ;;
        --models-max) models_max="${2:?--models-max needs a value}"; shift 2 ;;
        --) shift; passthrough=("$@"); break ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

[[ -d "$models_dir" ]] || die "models directory not found: ${models_dir}. Create it and put GGUF files there, or pass --models-dir."

# When a config is given, derive the launch flags from models.toml (the same EndpointSpec the pool
# uses), then let any manual flag override. Nothing here contacts a server, so it fails fast on a
# bad config rather than at launch.
config_flags=()
if [[ -n "$config" ]]; then
    [[ -f "$config" ]] || die "config not found: ${config}"
    python="$(project_python)"
    assert_python_new_enough "$python"
    mapfile -t config_flags < <(
        PYTHONPATH="${REPO_ROOT}/src" "$python" -m ragkit.llm.serveargs \
            --config "$config" --endpoint "$endpoint" --models-dir "$models_dir"
    ) || die "could not derive launch flags from ${config} (endpoint '${endpoint}'); see above"
fi

# Manual overrides win over the config-derived flags. With no config and no flag, fall back to the
# documented defaults.
[[ -n "$port" ]] && { [[ "$port" =~ ^[1-9][0-9]*$ ]] || die "--port must be a positive integer, got '${port}'"; }
[[ -n "$models_max" ]] && { [[ "$models_max" =~ ^[1-9][0-9]*$ ]] || die "--models-max must be a positive integer, got '${models_max}'"; }
require_command llama-server "Build llama.cpp (the llama-server binary) and put it on PATH. See https://github.com/ggml-org/llama.cpp"

override_flags=()
[[ -n "$host" ]] && override_flags+=(--host "$host")
[[ -n "$port" ]] && override_flags+=(--port "$port")
[[ -n "$models_max" ]] && override_flags+=(--models-max "$models_max")

if [[ ${#config_flags[@]} -eq 0 ]]; then
    # No config: assemble the defaults, applying any manual overrides.
    config_flags=(--host "${host:-127.0.0.1}" --port "${port:-8080}" \
        --models-dir "$models_dir" --models-max "${models_max:-4}" --jinja)
    override_flags=()
fi

note "router mode  (models from ${models_dir}; endpoint '${endpoint}' from ${config:-defaults})"
# Launched with NO model = router mode. Reranking needs a reranker model plus, per-model,
# --embedding --pooling rank; put those in the endpoint's server_args or after -- .
exec llama-server "${config_flags[@]}" "${override_flags[@]}" "${passthrough[@]}"
