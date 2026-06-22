#!/usr/bin/env bash
# Quick environment check. Run after connecting, before working:
#
#     bash check_env.sh
#
# It confirms you are on a node that has the project virtual environment and
# that torch and CUDA are visible. /scratch is node-local on this cluster, so
# the venv exists only on the node where setup.sh was run; on any other node
# this prints a clear message and a fix instead of a bare ModuleNotFoundError.
set -o pipefail

_PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
_VENV_PY="${_PROJECT_ROOT}/.venv/bin/python"

echo "host    : $(hostname)"
echo "project : ${_PROJECT_ROOT}"

if [ ! -x "${_VENV_PY}" ]; then
    echo "ERROR: no virtual environment at ${_PROJECT_ROOT}/.venv on this node."
    echo "       /scratch is node-local; the venv lives only on the node where it was built."
    echo "       Rebuild it here with:  bash setup.sh"
    exit 1
fi

"${_VENV_PY}" - <<'PY'
import importlib.util
import sys

required = ("torch", "torchvision", "open_clip", "numpy", "pandas",
            "PIL", "h5py", "sklearn", "tqdm", "matplotlib")
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print("ERROR: missing packages:", ", ".join(missing))
    print("       Install them with:  python -m pip install -r requirements.txt")
    sys.exit(1)

import torch

print("python  :", sys.version.split()[0])
print("torch   :", torch.__version__)
if torch.cuda.is_available():
    print("cuda    : True -", torch.cuda.get_device_name(0))
else:
    print("cuda    : False (running on CPU)")
print("OK: environment ready.")
PY