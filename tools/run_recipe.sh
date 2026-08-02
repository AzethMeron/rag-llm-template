#!/usr/bin/env bash
# Run one recipe end to end against a locally-served model: start the server, import the held-out
# questions, produce, export, and score.
#
# This is the sequence every recipe README spells out by hand (serve, import, run, export, eval),
# which is four-plus commands and so belongs in a script (CLAUDE.md's script threshold). Doing it
# by hand is also where the two real hazards live: leaving a server resident so the next recipe's
# models will not fit in VRAM, and evaluating a journal from an earlier run without noticing.
# This owns the server's lifetime (stopped on exit, success or failure, including Ctrl-C) and
# refuses to reuse a stale run store unless told to resume.
#
# It never touches a recipe's reference corpus. The [pairings]/[vector] stores are read as they
# are; if a recipe has none, its own fetch.sh builds one — that is a separate, one-time step.
#
# Usage: tools/run_recipe.sh --recipe NAME [--limit N] [--concurrency N] [--resume] [--no-eval]
#   --recipe NAME      a directory under recipes/ (e.g. med_evidence)
#   --limit N          stop after N records (a smoke test; the default runs the full set)
#   --concurrency N    records produced at once (default: 2)
#   --resume           keep an existing run store and continue it, instead of starting clean
#   --no-eval          produce and export, but skip the recipe's eval.py
#   --endpoint NAME    the models.toml endpoint to serve (default: local)
#   --set K=V          a persona-instruction substitution, repeatable (translation needs
#                      --set source_language=... --set target_language=...)

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

recipe=""
limit=""
concurrency="2"
resume=0
run_eval=1
endpoint="local"
substitutions=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --recipe) recipe="${2:?--recipe needs a value}"; shift 2 ;;
        --limit) limit="${2:?--limit needs a value}"; shift 2 ;;
        --concurrency) concurrency="${2:?--concurrency needs a value}"; shift 2 ;;
        --endpoint) endpoint="${2:?--endpoint needs a value}"; shift 2 ;;
        --set) substitutions+=(--set "${2:?--set needs a KEY=VALUE}"); shift 2 ;;
        --resume) resume=1; shift ;;
        --no-eval) run_eval=0; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

[[ -n "$recipe" ]] || die "--recipe NAME is required (a directory under recipes/)"
recipe_dir="${REPO_ROOT}/recipes/${recipe}"
[[ -d "$recipe_dir" ]] || die "no recipe '${recipe}' -- expected ${recipe_dir}"
config_dir="${recipe_dir}/config"
[[ -f "${config_dir}/models.toml" ]] || die "no models.toml in ${config_dir}"
heldout="${recipe_dir}/data/heldout.jsonl"
gold="${recipe_dir}/data/gold.jsonl"
[[ -f "$heldout" ]] || die "no held-out questions at ${heldout} -- run recipes/${recipe}/fetch.sh first"
for name in limit concurrency; do
    value="${!name}"
    [[ -z "$value" || "$value" =~ ^[1-9][0-9]*$ ]] || die "--${name} must be a positive integer, got '${value}'"
done

python="$(project_python)"
assert_python_new_enough "$python"
has_module "$python" ragkit || export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
require_command llama-server "Build llama.cpp and put llama-server on PATH (tools/build_llama_cpp.sh)."

work="${REPO_ROOT}/work"
mkdir -p "$work" || die "could not create ${work}"
run_db="${work}/${recipe}.db"
journal="${work}/${recipe}.jsonl"
server_log="${work}/${recipe}.server.log"

if [[ "$resume" == 0 && -e "$run_db" ]]; then
    note "removing the previous run store ${run_db} (pass --resume to continue it instead)"
    rm -f "$run_db" "${run_db}-wal" "${run_db}-shm" || die "could not remove ${run_db}"
fi

# The server is this script's resource: started here, stopped here, on every exit path. A server
# left resident is not a cosmetic leak -- it holds VRAM the next recipe's models need.
server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        note "stopping the model server (pid ${server_pid})"
        kill "$server_pid" 2>/dev/null || true
        for _ in $(seq 1 50); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 0.2
        done
        kill -9 "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

base_url="$(PYTHONPATH="${REPO_ROOT}/src" "$python" - "$config_dir/models.toml" "$endpoint" <<'PY' || die "could not read the endpoint's base_url"
import sys
from pathlib import Path
from ragkit.llm.pool import load_models
print(load_models(Path(sys.argv[1])).endpoint(sys.argv[2]).base_url)
PY
)"
health="${base_url%/v1}/health"

note "serving ${recipe} models (endpoint '${endpoint}') -- log: ${server_log}"
"${REPO_ROOT}/tools/serve_models.sh" --config "${config_dir}/models.toml" --endpoint "$endpoint" \
    --models-dir "${REPO_ROOT}/models" >"$server_log" 2>&1 &
server_pid=$!

for attempt in $(seq 1 300); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        die "the model server exited before becoming ready; see ${server_log}"
    fi
    curl -sf -m 2 "$health" >/dev/null 2>&1 && break
    [[ "$attempt" == 300 ]] && die "the model server never became ready at ${health}; see ${server_log}"
    sleep 1
done
note "server ready at ${base_url}"

"$python" -m ragkit.cli import --catalog "$heldout" --run-db "$run_db" --no-log-file \
    || die "import failed"

run_args=(run --config "$config_dir" --run-db "$run_db" --concurrency "$concurrency" --no-log-file)
[[ -n "$limit" ]] && run_args+=(--limit "$limit")
[[ ${#substitutions[@]} -gt 0 ]] && run_args+=("${substitutions[@]}")
"$python" -m ragkit.cli "${run_args[@]}" || die "run failed"

"$python" -m ragkit.cli export --run-db "$run_db" -j "$journal" --no-log-file || die "export failed"

if [[ "$run_eval" == 1 ]]; then
    [[ -f "${recipe_dir}/eval.py" ]] || die "no eval.py in ${recipe_dir} (pass --no-eval to skip)"
    [[ -f "$gold" ]] || die "no gold at ${gold} (pass --no-eval to skip scoring)"
    note "scoring ${recipe}"
    # legal_procurement scores the retriever, not the journal, so it takes different arguments.
    if [[ "$recipe" == "legal_procurement" ]]; then
        "$python" -m "recipes.${recipe}.eval" --config "$config_dir" \
            --heldout "$heldout" --gold "$gold" --journal "$journal" || die "eval failed"
    else
        "$python" -m "recipes.${recipe}.eval" --journal "$journal" --gold "$gold" \
            $( [[ "$recipe" == "nl_to_sql" ]] && echo "--config ${config_dir}" ) \
            || die "eval failed"
    fi
fi

note "done: ${recipe} -- run store ${run_db}, journal ${journal}"
