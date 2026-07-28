#!/usr/bin/env bash
# Static analysis: ruff (ruleset pinned in ruff.toml) + mypy.
#
# Both run through the project interpreter (python -m ...) so they resolve from .venv/
# without the caller having to activate it -- the same contract the other tools honour.
#
# Usage: tools/lint.sh [--fix]
#   --fix   apply ruff's safe autofixes before reporting

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

fix=0
for arg in "$@"; do
    case "$arg" in
        --fix) fix=1 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$arg' (see --help)" ;;
    esac
done

python="$(project_python)"
assert_python_new_enough "$python"
has_module "$python" ruff || die "ruff is not installed; run tools/setup_python_env.sh"
has_module "$python" mypy || die "mypy is not installed; run tools/setup_python_env.sh"

cd "$REPO_ROOT"
status=0

if [[ "$fix" == 1 ]]; then
    "$python" -m ruff check --fix src tests recipes || status=1
else
    "$python" -m ruff check src tests recipes || status=1
fi

"$python" -m mypy src || status=1
# Recipe plugins import ragkit by the same dotted path a run uses; MYPYPATH puts src on the
# search path so ragkit (with its py.typed marker) resolves as a typed package.
MYPYPATH="${REPO_ROOT}/src" "$python" -m mypy --explicit-package-bases recipes || status=1

if [[ "$status" == 0 ]]; then
    note "clean."
else
    note "lint found issues (see above)."
fi
exit "$status"
