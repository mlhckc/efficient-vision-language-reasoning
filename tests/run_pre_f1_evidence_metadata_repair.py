"""Run the standalone pre-F1 evidence-metadata successor tests.

This runner stays separate from tests/run_all.py because the latter is a
byte-pinned source input of closed evidence packets.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests import test_pre_f1_evidence_metadata_repair  # noqa: E402


def main() -> int:
    try:
        test_pre_f1_evidence_metadata_repair.run()
    except Exception as error:
        print(f"FAIL pre-F1 evidence-metadata repair: {error}")
        return 1
    print("all pre-F1 evidence-metadata repair tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
