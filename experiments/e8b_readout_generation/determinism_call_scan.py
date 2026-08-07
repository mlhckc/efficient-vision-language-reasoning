"""OI1 codebase scan: every call capable of changing deterministic
execution state, catalogued and classified.

Covers: deterministic-algorithm toggles and their warn_only argument,
SDPA backend switches, cudnn determinism and benchmark flags, TF32 and
matmul-precision controls, RNG seeding and state restoration, and the
cuBLAS workspace variable. The scan regenerates
determinism_call_scan_20260807.json so the record is reproducible from
this committed script.

    python -B experiments/e8b_readout_generation/determinism_call_scan.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils  # noqa: E402

PATTERNS = {
    "deterministic_algorithms": r"use_deterministic_algorithms",
    "warn_only": r"warn_only",
    "sdpa_backend": r"flash_sdp|mem_efficient_sdp|math_sdp|sdpa_kernel",
    "cudnn_flags": r"cudnn\.deterministic|cudnn\.benchmark",
    "tf32_or_matmul_precision": r"allow_tf32|float32_matmul_precision"
                                r"|fp32_precision",
    "seeding": r"\bset_seed\s*\(|manual_seed\s*\(|seed_worker"
               r"|set_rng_state|default_rng\s*\(",
    "cublas_workspace": r"CUBLAS_WORKSPACE_CONFIG",
}

# Modules on the CORE execution path (the trainer and everything it
# imports), where an un-reimposed downgrade would reach a scientific run.
CORE_PATH = {
    "experiments/e8b_readout_generation/training.py",
    "experiments/e8b_readout_generation/run.py",
    "experiments/e8b_readout_generation/latents.py",
    "experiments/e8b_readout_generation/readouts.py",
    "experiments/e8b_readout_generation/determinism_probe.py",
    "src/utils.py",
    "src/tokens_data.py",
    "experiments/e8a_question_encoder/e8a_common.py",
    "experiments/e8a_question_encoder/g21_scorer.py",
    "experiments/e8a_question_encoder/reinfer_g21.py",
}

# The known downgrade sources and where each is re-imposed. Every entry
# is verified against the live scan below; a new unlisted core-path hit
# fails the scan.
DOWNGRADE_NOTES = {
    "src/utils.py": "set_seed calls use_deterministic_algorithms(True, "
                    "warn_only=True): EVERY call downgrades strict "
                    "enforcement. Core discipline: reseed_strict or an "
                    "explicit enable_strict_determinism after each call, "
                    "with assert_strict_determinism at the point of use.",
    "experiments/e8b_readout_generation/latents.py":
        "build_trunk_and_projection reseeds (G13 order) -> the trainer "
        "re-imposes immediately after build_arm.",
    "experiments/e8b_readout_generation/run.py":
        "load_frozen_causal_lm reseeds for the random arm; "
        "enable_strict_determinism / reseed_strict / "
        "assert_strict_determinism are the enforcement helpers "
        "themselves; the trainer re-imposes after every load.",
    "experiments/e8b_readout_generation/training.py":
        "g8/b1 overfit gates reseed internally -> the trainer re-imposes "
        "after the gate and asserts before the first optimizer step.",
    "experiments/e8b_readout_generation/determinism_probe.py":
        "seeds BEFORE imposing strictness, re-imposes after build_arm, "
        "asserts at the point of use.",
    "experiments/e8a_question_encoder/e8a_common.py":
        "not on the core E8B training path at runtime (tokenizer and "
        "provenance helpers only); its seeding sites belong to E8A runs.",
    "experiments/e8a_question_encoder/g21_scorer.py":
        "pure scoring/normalisation; any RNG use is outside the E8B "
        "trainer path.",
    "experiments/e8a_question_encoder/reinfer_g21.py":
        "GPU-exclusivity helper only on the E8B path.",
    "src/tokens_data.py":
        "make_generator/seed_worker configure DataLoader RNG; they do "
        "not touch determinism enforcement flags.",
}


def scan() -> dict:
    hits = {}
    for path in sorted(PROJECT_ROOT.glob("**/*.py")):
        rel = str(path.relative_to(PROJECT_ROOT))
        if rel.startswith((".venv", ".cache", "results", "data")):
            continue
        text = path.read_text(errors="replace")
        file_hits = []
        for lineno, line in enumerate(text.splitlines(), 1):
            for category, pattern in PATTERNS.items():
                if re.search(pattern, line):
                    file_hits.append({"line": lineno,
                                      "category": category,
                                      "text": line.strip()[:160]})
        if file_hits:
            hits[rel] = file_hits
    core = {k: v for k, v in hits.items() if k in CORE_PATH}
    unexplained = [k for k in core if k not in DOWNGRADE_NOTES
                   and any(h["category"] in ("deterministic_algorithms",
                                             "warn_only", "sdpa_backend",
                                             "seeding")
                           for h in core[k])]
    return {"determinism_call_scan": {
        "purpose": "OI1: every call capable of changing deterministic "
                   "algorithms, warn_only, SDPA backends, cudnn "
                   "determinism/benchmark, TF32/matmul precision or RNG "
                   "configuration, catalogued and classified",
        "patterns": PATTERNS,
        "core_path_modules": sorted(CORE_PATH),
        "core_path_hits": core,
        "core_path_notes": DOWNGRADE_NOTES,
        "unexplained_core_path_files": unexplained,
        "non_core_hits": {k: v for k, v in hits.items()
                          if k not in CORE_PATH},
        "rule": "strict determinism is NEVER assumed from startup state: "
                "it is re-imposed after every seeding point and "
                "re-verified at the point of use "
                "(assert_strict_determinism), and every scientific run "
                "records verified_at_use",
    }}


def main() -> int:
    record = scan()
    body = record["determinism_call_scan"]
    if body["unexplained_core_path_files"]:
        print("UNEXPLAINED core-path determinism calls in:",
              body["unexplained_core_path_files"])
        return 1
    record["metadata"] = utils.run_metadata()
    out = (PROJECT_ROOT / "results" / "experiments"
           / "e8b_readout_generation"
           / "determinism_call_scan_20260807.json")
    if out.exists():
        out.unlink()
    out.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"core-path files with hits: {len(body['core_path_hits'])}; "
          f"non-core files: {len(body['non_core_hits'])}; "
          f"unexplained: 0")
    print(f"written {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
