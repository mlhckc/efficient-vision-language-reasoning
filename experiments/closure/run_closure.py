"""Run the whole closure in dependency order.

    python -m experiments.closure.run_closure

Each stage is deterministic and idempotent, so re-running the whole sequence
reproduces every artefact byte for byte. The order matters: the row-level
reconstruction gates everything downstream, and the claim table reads the
contrast, statistics, efficiency and registry outputs.

reconstruct_rows returns a non-zero exit code when it records a blocker. That
is its designed signal, not a crash: the stage completed, and a family failed
the point-estimate reproduction gate and was stopped. The orchestrator
reports it and continues, because the remaining stages are built to exclude a
stopped family rather than to depend on it.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STAGES = [
    ("experiments.closure.reconstruct_rows",
     "Part 4: reconstruct row-level evidence behind the reproduction gate"),
    ("experiments.closure.build_intervals",
     "dependence-aware statistics over the reproduced rows"),
    ("experiments.closure.build_registry",
     "Output A: evidence registry, plus the E7b and E8B manifests"),
    ("experiments.closure.build_contrast_table",
     "Outputs B and F: contrasts, multiplicity and status"),
    ("experiments.closure.build_slice_table",
     "Output D: slice, type and reasoning-depth statistics"),
    ("experiments.closure.build_seed_table",
     "Output E: training-seed variability"),
    ("experiments.closure.build_efficiency_table",
     "efficiency closure table"),
    ("experiments.closure.build_claim_table",
     "Outputs G and H: claims, limitations and readiness"),
]


def main() -> int:
    blockers = False
    for name, description in STAGES:
        print(f"\n=== {name} : {description} ===")
        module = importlib.import_module(name)
        code = module.main()
        if code != 0:
            if name.endswith("reconstruct_rows"):
                blockers = True
                print(f"[recorded] {name} exit {code}: a reproduction gate "
                      f"fired and its family was stopped. See "
                      f"results/closure/C_reconstructed_evidence_manifest"
                      f".json blockers")
            else:
                print(f"[FAILED] {name} exit {code}")
                return code
    print("\nclosure complete"
          + (" with recorded blockers" if blockers else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
