#!/usr/bin/env bash
# Build a pinned commit of llama.cpp (the llama-server binary tools/serve_models.sh launches) --
# the one dependency of this project that is a separate program, not a pip package, and so was
# not covered by requirements.txt/setup_python_env.sh. Idempotent: re-running with the same
# --commit against an already-built checkout is a fast no-op unless --force.
#
# Why this exists: this repo's own environment had a working llama-server built by hand at some
# point, with no record of which commit or build flags produced it -- exactly the "manual,
# undocumented step a person or agent has to remember" CLAUDE.md rules out. The default --commit
# below is that same commit, so re-running this script reproduces today's known-working binary.
#
# Usage: tools/build_llama_cpp.sh [--dir DIR] [--commit SHA] [--cuda-arch ARCH] [--no-cuda]
#                                 [--force]
#   --dir DIR        where to clone/build (default: ~/.local/share/llama.cpp)
#   --commit SHA     the llama.cpp commit to build (default: 91d2fc387529940230555abd297a8b5e99737d3f
#                     -- the commit this environment's own llama-server was already built from)
#   --cuda-arch ARCH  CMAKE_CUDA_ARCHITECTURES value (default: native -- CMake auto-detects the
#                     installed GPU's compute capability; pin explicitly, e.g. 120 for an RTX 5080,
#                     to cross-compile for a GPU not present on the build machine)
#   --no-cuda        build CPU-only (skip GGML_CUDA=ON) -- for a machine with no NVIDIA GPU/CUDA
#                     toolkit; tools/serve_models.sh works either way, just slower without CUDA
#   --force          rebuild even if --commit is already checked out and already built

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

repo_url="https://github.com/ggml-org/llama.cpp"
dir="${HOME}/.local/share/llama.cpp"
commit="91d2fc387529940230555abd297a8b5e99737d3f"
cuda_arch="native"
use_cuda=1
force=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir) dir="${2:?--dir needs a value}"; shift 2 ;;
        --commit) commit="${2:?--commit needs a value}"; shift 2 ;;
        --cuda-arch) cuda_arch="${2:?--cuda-arch needs a value}"; shift 2 ;;
        --no-cuda) use_cuda=0; shift ;;
        --force) force=1; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

[[ "$commit" =~ ^[0-9a-f]{7,40}$ ]] || die "--commit must be a git SHA (7-40 hex chars), got '$commit'"

require_command git "install git"
require_command cmake "install CMake (>= 3.14): https://cmake.org/install/"
if [[ "$use_cuda" -eq 1 ]]; then
    require_command nvcc \
        "install the CUDA toolkit (nvcc on PATH), or pass --no-cuda for a CPU-only build"
fi

binary="${dir}/build/bin/llama-server"
if [[ -d "${dir}/.git" ]]; then
    current_commit="$(git -C "$dir" rev-parse HEAD 2>/dev/null || echo "")"
    if [[ "$current_commit" == "$commit"* && -x "$binary" && "$force" -eq 0 ]]; then
        note "already built at ${commit} (${binary}) -- pass --force to rebuild"
        exit 0
    fi
    note "fetching ${repo_url} into existing checkout at ${dir}"
    git -C "$dir" fetch --quiet origin "$commit" \
        || die "could not fetch commit ${commit} from ${repo_url} into ${dir}"
    git -C "$dir" checkout --quiet "$commit" \
        || die "could not check out commit ${commit} in ${dir}"
else
    [[ -e "$dir" ]] && die "${dir} exists but is not a git checkout -- remove it or pass --dir"
    note "cloning ${repo_url} into ${dir}"
    git clone --quiet "$repo_url" "$dir" || die "could not clone ${repo_url} into ${dir}"
    git -C "$dir" checkout --quiet "$commit" \
        || die "could not check out commit ${commit} in ${dir} after cloning"
fi

cmake_args=(-B "${dir}/build" -S "$dir" -DCMAKE_BUILD_TYPE=Release)
if [[ "$use_cuda" -eq 1 ]]; then
    cmake_args+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch")
fi

note "configuring (cuda=${use_cuda}, cuda-arch=${cuda_arch})"
cmake "${cmake_args[@]}" || die "cmake configure failed (see above)"

note "building (this compiles llama.cpp + ggml from source; expect several minutes)"
cmake --build "${dir}/build" --config Release -j "$(nproc)" \
    || die "build failed (see above)"

[[ -x "$binary" ]] || die "build finished but ${binary} was not produced -- unexpected"
note "built: ${binary}"
note "add it to PATH, e.g.: export PATH=\"${dir}/build/bin:\$PATH\""
