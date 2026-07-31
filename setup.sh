#!/usr/bin/env bash
# One-time environment setup: create the virtual environment and install
# dependencies. Run once from the project root:
#
#     bash setup.sh
#
# After this, each working session only needs:
#
#     source .venv/bin/activate && source env.sh
#
# The script stops on the first error so a failed install is visible rather
# than silently ignored. pip is always invoked as `python -m pip` because the
# project path contains spaces, which breaks the shebang line of the generated
# `pip` shim; calling pip as a module avoids that.
set -eo pipefail

_PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${_PROJECT_ROOT}"

echo "Creating virtual environment at .venv"
python3 -m venv .venv

echo "Activating virtual environment"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Redirecting caches (env.sh)"
# shellcheck disable=SC1091
source env.sh

echo "Upgrading pip"
python -m pip install --upgrade pip

if [ -f requirements.lock.txt ]; then
    echo "Installing pinned requirements (requirements.lock.txt)"
    python -m pip install -r requirements.lock.txt
else
    echo "WARNING: requirements.lock.txt not found; installing unpinned"
    echo "         requirements.txt. Resolved versions may differ from the"
    echo "         environment that produced the stored results."
    python -m pip install -r requirements.txt
fi

echo
echo "Setup complete."
echo "Start each session with:"
echo "    source .venv/bin/activate && source env.sh"
