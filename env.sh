#!/usr/bin/env bash
# Redirect framework and tool caches into a project-local .cache directory.
#
# The home directory on this machine has very little free space, so model
# downloads (CLIP weights via open_clip / Hugging Face) and pip's cache must
# not land in ~/.cache. Source this file each session, after activating the
# virtual environment:
#
#     source .venv/bin/activate
#     source env.sh
#
# Sourcing it more than once is safe.

# Directory of this script, so it works regardless of where it is sourced from.
_PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
_CACHE_DIR="${_PROJECT_ROOT}/.cache"

export XDG_CACHE_HOME="${_CACHE_DIR}"
export HF_HOME="${_CACHE_DIR}/huggingface"
export TORCH_HOME="${_CACHE_DIR}/torch"
export PIP_CACHE_DIR="${_CACHE_DIR}/pip"

mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}" "${PIP_CACHE_DIR}"

echo "Caches redirected to ${_CACHE_DIR}"
echo "  XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "  HF_HOME=${HF_HOME}"
echo "  TORCH_HOME=${TORCH_HOME}"
echo "  PIP_CACHE_DIR=${PIP_CACHE_DIR}"
