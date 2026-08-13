"""Run the closure regression tests.

Usage, from the project root with the venv active:

    python -B tests/run_closure.py

WHY THIS IS A SEPARATE RUNNER, and not another module inside run_all.py.

`tests/run_all.py` is one of the files listed in
`experiments/e10_capacity_360m/e10_common.py` SOURCE_PATHS. E10 is closed at
E10_PASS and its completed evidence is carried across the R3.1 live-binding
amendment, which records the exact source digest of those files. Adding an
import to run_all.py changes that digest, and the three E10 phase verifiers
then refuse to consume what they correctly see as stale output. That was
observed, not assumed: the edit moved the live digest from
e666f101... to 71cbaaa4... and failed test_e10_phase1, phase2 and phase3.

The seal is doing its job. A closed experiment's test harness is part of what
was reviewed, so this closure leaves it untouched and ships its own runner
instead. Adding new files under tests/ is safe, because SOURCE_PATHS names
individual files rather than a directory glob.

The embargo source scan in run_all.py globs tests/*.py, so the closure test
sources are still covered by it without any edit to the sealed file.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests import test_closure  # noqa: E402

MODULES = (test_closure,)


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
    print("all closure test modules passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
