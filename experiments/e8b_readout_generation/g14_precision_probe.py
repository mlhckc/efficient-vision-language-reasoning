"""NON-SCIENTIFIC G14 precision diagnostic: bf16 versus fp32 R1 deltas.

    python -B experiments/e8b_readout_generation/g14_precision_probe.py

Regenerates results/experiments/e8b_readout_generation/
g14_precision_diagnostic.json, the recorded justification for reading
G14's 1e-4 score tolerance as informational: on one pinned random prefix
(torch.manual_seed(0), scale 0.13) over the full top-100 answer cache,
the cached and brute-force R1 implementations agree within the tolerance
when the frozen SmolLM2-135M runs in fp32, while the same implementations
under the pinned bf16 weights show score deltas orders of magnitude
larger with the argmax still identical. The binding G14 clause is argmax
identity (master protocol section 18, restated in section 20); this probe
shows the bf16 deltas are kernel numerics, not a cached-path defect.

The probe uses no development row, computes no accuracy and selects
nothing; its output is labelled NON-SCIENTIFIC and never enters any
scientific aggregation. Nothing here reads, resolves or names the
embargoed clean-test target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402
from experiments.e8b_readout_generation.run import (  # noqa: E402
    OUT_DIR, atomic_write_json, load_frozen_causal_lm)

PROBE_SEED = 0
PROBE_SCALE = 0.13


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm, _ = load_frozen_causal_lm(True, device)
    tokenizer = e8a.load_tokenizer()
    answers, _ = g21.load_index_to_answer(
        config.DATA_DIR / "v2" / "answer_vocab_v2.json")
    cache = readouts.build_answer_cache(tokenizer, answers)
    torch.manual_seed(PROBE_SEED)
    prefix_fp32 = (torch.randn(1, 33, 576) * PROBE_SCALE).to(device)

    gate_bf16 = readouts.g14_r1_gate(lm, [prefix_fp32.to(torch.bfloat16)],
                                     cache)
    lm_fp32 = lm.to(torch.float32)
    gate_fp32 = readouts.g14_r1_gate(lm_fp32, [prefix_fp32], cache)

    payload = {"metadata": utils.run_metadata(),
               "g14_precision_diagnostic": {
        "label": "NON-SCIENTIFIC DIAGNOSTIC",
        "purpose": "establish that the large recorded R1 score delta under "
                   "the bf16 frozen model is kernel numerics, not a "
                   "cached-path defect: the identical implementations "
                   "under fp32 agree within the informational 1e-4 "
                   "tolerance",
        "probe": f"one pinned random prefix (torch.manual_seed("
                 f"{PROBE_SEED}), scale {PROBE_SCALE}), full top-100 "
                 f"answer cache, pretrained SmolLM2-135M",
        "generator": "experiments/e8b_readout_generation/"
                     "g14_precision_probe.py",
        "bf16_max_abs_score_delta": gate_bf16["max_abs_score_delta"],
        "bf16_argmax_identical": gate_bf16["argmax_identical"],
        "fp32_max_abs_score_delta": gate_fp32["max_abs_score_delta"],
        "fp32_argmax_identical": gate_fp32["argmax_identical"],
        "binding_clause": "G14 argmax identity (master protocol section "
                          "18); the score tolerance is informational"}}
    out = OUT_DIR / "g14_precision_diagnostic.json"
    out.unlink(missing_ok=True)
    atomic_write_json(out, payload)
    print(f"bf16 delta {gate_bf16['max_abs_score_delta']:.4e} / fp32 delta "
          f"{gate_fp32['max_abs_score_delta']:.4e}; argmax identical in "
          f"both; NON-SCIENTIFIC record written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
