"""Run the E7b supersession overlay tests.

Usage, from the project root with the venv active:

    python -B tests/run_e7b_supersession.py

WHY THIS IS A SEPARATE RUNNER, and not another module inside run_all.py.

The task packet asked for the new module to be registered in
`tests/run_all.py`. That file is one of the paths listed in
`experiments/e10_capacity_360m/e10_common.py` SOURCE_PATHS. E10 is closed at
E10_PASS and its completed evidence is carried across the R3.1 live-binding
amendment, which records the exact source digest of those files. Adding an
import to run_all.py changes that digest, and the three E10 phase verifiers
then refuse to consume what they correctly see as stale output. That was
observed, not assumed: the edit moved the live digest from e666f101... to
71cbaaa4... and failed test_e10_phase1, phase2 and phase3.

Registering here would therefore have broken a closed experiment to publish
a withdrawal record, which is the opposite of what the withdrawal is for. So
this task follows the precedent the closure set with tests/run_closure.py and
VE-0, VE-1 and VE-2 followed with their own runners: a new file under tests/,
and no edit to the sealed one. Adding new files under tests/ is safe, because
SOURCE_PATHS names individual files rather than a directory glob.

The embargo source scan in run_all.py globs tests/*.py, so this test source is
still covered by it without any edit to the sealed file.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests import test_e7b_supersession  # noqa: E402

MODULES = (test_e7b_supersession,)


def main() -> int:
    failures = 0
    for module in MODULES:
        name = module.__name__.split(".")[-1]
        try:
            module.run()
        except Exception as error:
            failures += 1
            print(f"FAIL {name}: {error}")
        else:
            print(f"PASS {name}")
    if failures:
        print(f"{failures} test module(s) failed")
        return 1
    print("all E7b supersession test modules passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
