"""Run the VE-1 figure and table tests.

Usage, from the project root with the venv active:

    python -B tests/run_ve1.py

WHY THIS IS A SEPARATE RUNNER, and not another module inside run_all.py.

`tests/run_all.py` is one of the files listed in
`experiments/e10_capacity_360m/e10_common.py` SOURCE_PATHS. E10 is closed at
E10_PASS and its completed evidence is carried across the R3.1 live-binding
amendment, which records the exact source digest of those files. Adding an
import to run_all.py changes that digest, and the three E10 phase verifiers
then refuse to consume what they correctly see as stale output. The closure
shipped tests/run_closure.py for this reason and VE-0 shipped tests/run_ve0.py;
VE-1 follows the same pattern.

Adding new files under tests/ is safe, because SOURCE_PATHS names individual
files rather than a directory glob. The embargo source scan in run_all.py
globs tests/*.py, so this test source is still covered by it without any edit
to the sealed file.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests import test_ve1  # noqa: E402

MODULES = (test_ve1,)


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
    print("all VE-1 test modules passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
