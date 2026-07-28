#!/usr/bin/env bash
# Create (or open) the stores described by a storage.toml, so a run has its databases ready.
#
# Building each store from config creates its on-disk artefacts: the SQLite schema, the LanceDB
# table. Idempotent -- running it again opens what exists. Needs the pinned environment.
#
# Usage: tools/init_storage.sh --config DIR
#   --config DIR   a config directory containing storage.toml

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

config=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) config="${2:?--config needs a value}"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

[[ -n "$config" ]] || die "--config DIR is required (a directory with storage.toml)"
[[ -f "${config}/storage.toml" ]] || die "no storage.toml in ${config}"

python="$(project_python)"
assert_python_new_enough "$python"
has_module "$python" ragkit || export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"$python" - "$config" <<'PY' || die "storage initialisation failed (see the error above)"
import sys
from pathlib import Path
from ragkit.store import load_storage
storage = load_storage(Path(sys.argv[1]) / "storage.toml")
built = [name for name, value in (("sql", storage.sql), ("vector", storage.vector),
                                  ("lexical", storage.lexical),
                                  ("introspector", storage.introspector)) if value is not None]
print(">> storage ready:", ", ".join(built) or "(nothing configured)")
PY
