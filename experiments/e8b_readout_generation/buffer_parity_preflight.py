"""NON-TRAINING preflight: execute and PERSIST the B2/B3 buffer parity.

The audit found `assert_lm_buffer_parity` wired but never executed in any
persisted artefact, so the claim that authorised B2/B3 construction
differs only in the intended pretrained-versus-random model state was not
independently reproducible. This module executes the comparison and
records everything a reviewer needs to reproduce it.

NO OPTIMIZER STEP IS AUTHORISED OR PERFORMED. There is no optimizer, no
backward pass and no parameter update anywhere in this file: it loads two
frozen models, inventories their buffers, compares them, and writes one
JSON record.

The embargoed clean-test target is never opened, listed, stat-ed, hashed
or resolved.

    python -B experiments/e8b_readout_generation/buffer_parity_preflight.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402
from experiments.e8b_readout_generation import training as e8b_training  # noqa: E402

OUT = e8b_run.OUT_DIR / "buffer_parity_preflight_20260807.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def classify(model, name: str) -> str:
    """persistent buffers appear in state_dict; non-persistent do not."""
    return "persistent" if name in model.state_dict() else "non-persistent"


def inventory(model) -> dict:
    """Full buffer inventory: name -> classification, dtype, shape and a
    value digest computed in float64 so the digest is dtype-independent
    and a bf16-truncated value cannot masquerade as its fp32 original."""
    out = {}
    for name, buffer in model.named_buffers():
        tensor = buffer.detach().cpu()
        out[name] = {
            "classification": classify(model, name),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "value_digest_float64": hashlib.sha256(
                tensor.to(torch.float64).numpy().tobytes()).hexdigest()
            if tensor.is_floating_point() else hashlib.sha256(
                tensor.numpy().tobytes()).hexdigest(),
            "first_values": [float(v) for v in
                             tensor.flatten()[:4].to(torch.float64)]
            if tensor.is_floating_point() else None}
    return out


def main() -> int:
    utils.set_seed(0)
    e8b_run.enable_strict_determinism()
    e8b_run.pin_fp32_precision()
    determinism = e8b_run.assert_strict_determinism()

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(e8b_run.MODEL_REPO,
                                     revision=e8b_run.MODEL_REVISION)

    arms = {}
    for arm, pretrained in (("B3", True), ("B2", False)):
        lm, provenance = e8b_run.load_frozen_causal_lm(pretrained,
                                                       device=None)
        e8b_run.enable_strict_determinism()   # the random load reseeds
        arms[arm] = {
            "arm_model": provenance["arm_model"],
            "construction": provenance["construction"],
            "pinned_revision": provenance["pinned_revision"],
            "parameter_count": provenance["parameter_count"],
            "parameter_state_sha256": provenance["state_dict_sha256"],
            "config_sha256": provenance["config_sha256"],
            "config_dtype_normalised": provenance["config_dtype_normalised"],
            "attn_implementation": provenance["attn_implementation"],
            "rotary_fp32_restoration":
                provenance["rotary_fp32_restoration"],
            "buffer_inventory": inventory(lm)}
        del lm

    # The reconstruction reference: a rotary module built from the SAME
    # pinned configuration, independent of either arm.
    reference = None
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding)
    module = LlamaRotaryEmbedding(config=cfg)
    reference = {name: {
        "dtype": str(t.detach().cpu().dtype),
        "value_digest_float64": hashlib.sha256(
            t.detach().cpu().to(torch.float64).numpy().tobytes()).hexdigest()}
        for name, t in module.named_buffers()}

    # The parity verdict itself, from the production helper.
    verdict = e8b_run.assert_lm_buffer_parity(
        arms["B3"]["buffer_inventory"], arms["B2"]["buffer_inventory"],
        label_a="B3", label_b="B2")

    both = arms["B3"]["buffer_inventory"]
    matches_reference = all(
        f"model.rotary_emb.{n}" not in both
        or both[f"model.rotary_emb.{n}"]["value_digest_float64"]
        == r["value_digest_float64"]
        for n, r in reference.items())
    non_persistent = [n for n, v in both.items()
                      if v["classification"] == "non-persistent"]
    parameters_differ = (arms["B3"]["parameter_state_sha256"]
                         != arms["B2"]["parameter_state_sha256"])

    record = {"metadata": utils.run_metadata(),
              "e8b_buffer_parity_preflight": {
        "dated_utc": "2026-08-07",
        "NON_TRAINING": True,
        "optimizer_steps": 0,
        "scope": "loads the two frozen arms, inventories and compares "
                 "every buffer, and records the rotary reconstruction "
                 "evidence. No optimizer, no backward pass, no parameter "
                 "update, no checkpoint written.",
        "determinism_verified_at_use": determinism,
        "arms": arms,
        "rotary_reconstruction_reference": {
            "source": "LlamaRotaryEmbedding(config=<pinned config>), "
                      "built independently of either arm from the SAME "
                      "pinned configuration",
            "config_sha256": arms["B3"]["config_sha256"],
            "buffers": reference,
            "both_arms_match_reference": bool(matches_reference)},
        "non_persistent_buffers": non_persistent,
        "parity_verdict": verdict,
        "parameters_differ_as_intended": bool(parameters_differ),
        "conclusion": (
            "REPRODUCIBLE CLAIM: authorised B2/B3 construction differs "
            "ONLY in the intended pretrained-versus-random parameter "
            "state. Every buffer, persistent and non-persistent, is "
            "byte-identical across the pair by dtype, shape and float64 "
            "value digest; the rotary buffers of both arms reproduce a "
            "reference rebuilt from the same pinned configuration; and "
            "the parameter state digests differ, as they must."
            if (verdict["all_identical"] and matches_reference
                and parameters_differ) else
            "PARITY NOT ESTABLISHED - see parity_verdict"),
        "code_hashes": {
            "run.py": sha256_text(
                (Path(e8b_run.__file__)).read_text()),
            "training.py": sha256_text(
                (Path(e8b_training.__file__)).read_text()),
            "this_module": sha256_text(Path(__file__).read_text())},
        "protocol_family": e8b_run.PROTOCOL_FAMILY,
        "clean_test_accessed": False}}
    if OUT.exists():
        OUT.unlink()
    OUT.write_text(json.dumps(record, indent=2, default=str) + "\n")
    body = record["e8b_buffer_parity_preflight"]
    print(f"  buffers compared : {verdict['buffers_compared']}")
    print(f"  non-persistent   : {non_persistent}")
    print(f"  all identical    : {verdict['all_identical']}")
    print(f"  match reference  : {matches_reference}")
    print(f"  params differ    : {parameters_differ}")
    print(f"  verdict          : {body['conclusion'][:60]}...")
    print(f"written {OUT.name}")
    return 0 if (verdict["all_identical"] and matches_reference
                 and parameters_differ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
