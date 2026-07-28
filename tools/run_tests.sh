#!/usr/bin/env bash
# Run the test suite. Needs no inference server, GPU, or network: every model call is served
# by a scripted fake or an in-memory HTTP transport.
#
# Usage: tools/run_tests.sh [--coverage] [pytest args...]
#   --coverage   measure statement AND branch coverage and print a report
#   any other arguments are forwarded to pytest (e.g. -k name, -x, a path)

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

coverage=0
pytest_args=()
for arg in "$@"; do
    case "$arg" in
        --coverage) coverage=1 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) pytest_args+=("$arg") ;;  # forwarded to pytest, not an error
    esac
done

python="$(project_python)"
assert_python_new_enough "$python"
has_module "$python" pytest || die "pytest is not installed; run tools/setup_python_env.sh"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "$coverage" == 1 ]]; then
    has_module "$python" coverage || die "coverage is not installed; run tools/setup_python_env.sh"
    # --branch, not statements alone: a 100% statement figure once hid four untaken branches,
    # two of which were real defects (recorded in the reference project's audit history).
    "$python" -m coverage run --branch --source="${REPO_ROOT}/src" -m pytest "${pytest_args[@]}"
    status=$?
    # `coverage report` exits non-zero when below .coveragerc's fail_under (100%). Honour that
    # exit code, not only pytest's, so a coverage regression fails the run instead of printing a
    # warning the caller ignores. A test failure still takes precedence in the reported status.
    "$python" -m coverage report --skip-covered
    report_status=$?
    [[ "$status" -eq 0 ]] && status="$report_status"
    exit "$status"
fi

exec "$python" -m pytest "${pytest_args[@]}"
