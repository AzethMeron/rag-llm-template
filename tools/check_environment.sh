#!/usr/bin/env bash
# Report, read-only, whether the environment can run the code and tests. Changes nothing.
#
# [ok]   a requirement that is satisfied
# [FAIL] a requirement that is not -- the suite or the code cannot run until it is fixed
# [note] an optional capability that only some features need (a real inference run, an
#        alternative storage driver); its absence never blocks the test suite
#
# Usage: tools/check_environment.sh

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

case "${1:-}" in
    "") ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument '${1}' (see --help)" ;;
esac

status=0
report() {
    # report ok|FAIL|note <message>
    local kind="$1"; shift
    case "$kind" in
        ok)   echo "  [ok] $*" ;;
        note) echo "  [note] $*" ;;
        FAIL) echo "  [FAIL] $*"; status=1 ;;
    esac
}

python="$(project_python)"

if assert_python_new_enough "$python" 2>/dev/null; then
    report ok "python interpreter is 3.${MIN_PYTHON_MINOR}+ ($("$python" -V 2>&1))"
else
    report FAIL "python is older than 3.${MIN_PYTHON_MINOR} (tomllib and modern typing are required)"
fi

if [[ -x "${VENV_DIR}/bin/python" ]]; then
    report ok "virtual environment present at ${VENV_DIR}"
else
    report FAIL "no .venv/ -- run tools/setup_python_env.sh"
fi

# SQLite capabilities are the default storage backend's foundation, and neither is a language
# guarantee: on Linux, FTS5 and loadable extensions are properties of the distro's libsqlite3,
# not of CPython. Probe both here rather than discovering the gap at query time.
if "$python" - <<'PY' 2>/dev/null
import sqlite3
sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(a)")
PY
then
    report ok "SQLite FTS5 is available (default lexical/BM25 index)"
else
    report FAIL "SQLite FTS5 is not compiled into this Python's libsqlite3; the default lexical index needs it"
fi

if "$python" -c "import sqlite3; sqlite3.connect(':memory:').enable_load_extension(True)" 2>/dev/null; then
    report ok "SQLite loadable extensions are enabled (needed only by the optional sqlite-vec driver)"
else
    report note "SQLite loadable extensions are disabled in this build; the optional sqlite-vec vector driver will not load (the default LanceDB driver does not need it)"
fi

for module in pytest coverage hypothesis; do
    if has_module "$python" "$module"; then
        report ok "$module is installed"
    else
        report FAIL "$module is not installed -- run tools/setup_python_env.sh"
    fi
done

for module in ruff mypy; do
    if has_module "$python" "$module"; then
        report ok "$module is installed"
    else
        report FAIL "$module is not installed -- run tools/setup_python_env.sh"
    fi
done

if has_module "$python" ragkit; then
    report ok "ragkit imports"
else
    report note "ragkit is not importable yet without PYTHONPATH=src (expected before install; the tools set it themselves)"
fi

# Optional runtime pieces: absent on this machine and never needed by the test suite.
has_module "$python" numpy \
    && report ok "numpy is installed (dense-vector math)" \
    || report note "numpy absent -- needed only for the embedding/dense retrieval features"
has_module "$python" lancedb \
    && report ok "lancedb is installed (default vector database)" \
    || report note "lancedb absent -- needed only to run real retrieval against the default vector store"
command -v llama-server >/dev/null 2>&1 \
    && report ok "llama-server on PATH (local inference)" \
    || report note "no llama-server on PATH -- needed only for a real inference run (tools/serve_models.sh); the tests never contact a model"

if [[ "$status" == 0 ]]; then
    echo "environment is ready. Run the tests with tools/run_tests.sh" >&2
else
    echo "environment is NOT ready; fix the [FAIL] items above." >&2
fi
exit "$status"
