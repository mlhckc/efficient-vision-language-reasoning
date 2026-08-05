"""Run every regression test and report a single pass/fail summary.

Usage, from the project root with the venv active:

    python -B tests/run_all.py

The tests are read-only over the stored data, vocabularies and
checkpoints; nothing is written, trained or modified. The reproduction
test performs one brief evaluation-only forward pass and skips itself
when CUDA is unavailable. None of the tests opens the embargoed
clean-test target file; a source scan enforces that no test references
it at all.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests import (
    test_checkpoints,
    test_e8a,
    test_e8a_safety,
    test_g21_scorer,
    test_mask,
    test_reproduction,
    test_schemas,
    test_seeding,
    test_vocab,
)

MODULES = (
    test_schemas,
    test_vocab,
    test_mask,
    test_seeding,
    test_checkpoints,
    test_reproduction,
    test_e8a,
    test_g21_scorer,
    test_e8a_safety,
)

EMBARGOED_TOKEN = "test_" + "clean"


def embargo_source_scan() -> None:
    """No test source may reference the embargoed clean-test path."""
    for source in sorted(Path(__file__).parent.glob("*.py")):
        text = source.read_text()
        if source.name == Path(__file__).name:
            # This file assembles the token deliberately; skip it.
            continue
        if EMBARGOED_TOKEN in text:
            raise AssertionError(
                f"{source.name} references the embargoed clean-test path"
            )
    print("PASS embargo source scan")


def main() -> int:
    failures = 0
    embargo_source_scan()
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
    print("all test modules passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
