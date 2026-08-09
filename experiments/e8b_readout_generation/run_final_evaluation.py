"""Orchestration driver: the frozen E8B evaluation over one frozen cell.

ORCHESTRATION ONLY. Every scientific decision -- R1 ranking, R2
constrained decoding, R3 bounded generation, the answer cache and trie,
the intervention constructions, the neutral means, the derangement, both
denominators, raw and G21-normalised scoring, out-of-vocabulary handling
and the batched readout path -- lives in final_evaluation.py,
batched_readouts.py, readouts.py and run.py and is CALLED here, never
reimplemented. This module chooses which frozen checkpoint to evaluate,
proves it is the frozen one, and records what came back.

ARM-AWARE, following the canonical plan. B2 and B3 receive R1, R2 and R3
under all four frozen conditions. B1 is evaluated AS A CLASSIFIER under
the same four conditions and nothing else: section 4 of the plan states
that the E8B likelihood treatment is not applied to B1, and plan section
13 says "evaluate B1 as a classifier". No R1/R2/R3 semantics are invented
for it. B1's conditions reuse the frozen intervention constructor by
wrapping the loader, so the intervention logic is the shared one.

The B1 wrapper is deliberately a generator over the SAME five-tuple the
loader yields, so b1_canonical_predictions runs unmodified with its own
fp32 and autocast gates intact.

NO TRAINING. No optimizer, no backward pass, no parameter update.
THE EMBARGOED CLEAN TEST IS NEVER TOUCHED.

    python -B experiments/e8b_readout_generation/run_final_evaluation.py \\
        --cell ARM SCALE SEED
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402
from experiments.e8b_readout_generation import training as e8b_training  # noqa: E402
from experiments.e8b_readout_generation import final_evaluation as fe  # noqa: E402
from experiments.e8b_readout_generation import validate_evaluation as ve  # noqa: E402

OUT_DIR = e8b_run.OUT_DIR
V2_DIR = config.DATA_DIR / "v2"
MANIFEST_RECORD = OUT_DIR / "e8b_core_analysis_20260809.json"
MANIFEST_BASIS_COMMIT = "1b4ea42"
BATCH_SIZE = 64


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluator_digest() -> str:
    """Digest of the scientific evaluation sources this driver calls, so
    a result can be tied to the implementation that produced it."""
    digest = hashlib.sha256()
    for name in ("final_evaluation.py", "batched_readouts.py",
                 "readouts.py", "run.py", "training.py"):
        digest.update(name.encode())
        digest.update(hashlib.sha256(
            (Path(__file__).parent / name).read_bytes()).digest())
    return digest.hexdigest()


def frozen_entry(arm: str, scale: str, seed: int) -> dict:
    """The manifest row for this cell, with every binding re-verified
    against disk. Any mismatch aborts before a GPU is touched."""
    if not MANIFEST_RECORD.exists():
        sys.exit(f"{MANIFEST_RECORD.name} is missing; evaluation binds "
                 f"exclusively to the frozen 18-cell manifest")
    analysis = json.loads(MANIFEST_RECORD.read_text())["e8b_core_analysis"]
    if analysis["basis_commit"] != MANIFEST_BASIS_COMMIT:
        sys.exit(f"the manifest names basis commit "
                 f"{analysis['basis_commit']!r}, not "
                 f"{MANIFEST_BASIS_COMMIT!r}")
    entries = [e for e in analysis["matrix_manifest"]
               if (e["arm"], e["scale"], e["seed"]) == (arm, scale, seed)]
    if len(entries) != 1:
        sys.exit(f"{arm}/{scale}/seed{seed} is not a unique manifest cell")
    entry = entries[0]

    if entry["arm"] != arm or entry["scale"] != scale \
            or entry["seed"] != seed:
        sys.exit("manifest identity mismatch")
    if entry["protocol_family"] != e8b_run.PROTOCOL_FAMILY:
        sys.exit(f"protocol family mismatch: {entry['protocol_family']!r}")
    if entry["clean_test_accessed"] is not False:
        sys.exit("the manifest records clean-test access; refusing")

    checkpoint = PROJECT_ROOT / entry["checkpoints"]["canonical"]["path"]
    if not checkpoint.exists():
        sys.exit(f"frozen checkpoint missing: {checkpoint}")
    live = sha256_file(checkpoint)
    if live != entry["checkpoints"]["canonical"]["sha256"]:
        sys.exit(f"CHECKPOINT HASH MISMATCH for {arm}/{scale}/seed{seed}: "
                 f"{live[:16]} on disk against "
                 f"{entry['checkpoints']['canonical']['sha256'][:16]} in "
                 f"the frozen manifest. Refusing to evaluate an artefact "
                 f"that has changed.")
    result_path = PROJECT_ROOT / entry["result_record"]["path"]
    if sha256_file(result_path) != entry["result_record"]["sha256"]:
        sys.exit(f"RESULT RECORD HASH MISMATCH for {result_path.name}")
    record = json.loads(result_path.read_text())["e8b_core_cell"]
    if record["clean_test_accessed"] is not False:
        sys.exit("the result record reports clean-test access; refusing")
    if record["recipe_sha256"] != e8b_training.recipe_sha256(
            e8b_training.build_core_recipe(arm, scale, seed)):
        sys.exit("recipe hash does not rebuild")
    # The selection rule is arm-aware and is asserted, not assumed: B2/B3
    # must be the epoch-22 checkpoint, B1 its own best-on-dev one.
    if arm in ("B2", "B3"):
        if record["canonical_checkpoint_rule"] != "epoch_22" \
                or record["canonical_epoch"] != 22:
            sys.exit(f"{arm} must evaluate its epoch-22 checkpoint")
        if not checkpoint.name.endswith("_canonical_ep22.pt"):
            sys.exit(f"{arm} checkpoint is not the epoch-22 artefact")
    else:
        if "best development accuracy" not in \
                record["canonical_checkpoint_rule"]:
            sys.exit("B1 must evaluate its frozen section-7.3 "
                     "best-on-development checkpoint")
        if not checkpoint.name.endswith("_canonical_best.pt"):
            sys.exit("B1 checkpoint is not the canonical_best artefact")
    entry["_checkpoint_path"] = checkpoint
    entry["_record"] = record
    return entry


def build_model(arm: str, seed: int, checkpoint: Path, lm, device):
    """The frozen arm, restored from the frozen checkpoint. Construction
    goes through run.build_arm, the same builder training used."""
    dropout = 0.1
    if arm == "B1":
        trunk, readout = e8b_run.build_arm("B1", seed, dropout, None)
        model = e8b_training.B1Classifier(trunk, readout)
    else:
        model, _ = e8b_run.build_arm(arm, seed, dropout, lm)
    e8b_run.enable_strict_determinism()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload = state["state"] if "state" in state else state
    model.load_state_dict(payload)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def intervened_batches(loader, condition: str, context: dict, device):
    """The loader, with the FROZEN intervention constructor applied.

    Yields the same five-tuple the loader yields, so a downstream
    evaluator runs unmodified. The intervention itself is
    run.intervention_inputs -- the shared constructor the language-model
    path uses -- so B1 and B2/B3 receive the same construction.
    """
    for images, questions, question_ids, mask, labels in loader:
        images = images.to(device)
        questions = questions.to(device)
        mask = mask.to(device)
        deranged = context.get("deranged_batch")
        if condition == "shuffled_image":
            deranged = context["deranged_lookup"](question_ids).to(device)
        image_in, question_in, mask_in = e8b_run.intervention_inputs(
            condition, images, questions, mask,
            context["neutral_image"], context["neutral_question"],
            context["neutral_mask"], deranged_image_tokens=deranged)
        yield image_in, question_in, question_ids, mask_in, labels


def score_lm_condition(result, subset, answers, in_mask) -> dict:
    """Scoring, delegated wholesale to final_evaluation."""
    gold_strings = list(subset["answer"])
    labels = np.array(subset["label"])
    r1 = fe.score_closed(np.array(result["r1_pred"])[in_mask],
                         labels[in_mask], answers)
    r2 = fe.score_closed(np.array(result["r2_pred"])[in_mask],
                         labels[in_mask], answers)
    r3_in = fe.score_open(
        [t for t, keep in zip(result["r3_text"], in_mask) if keep],
        [g for g, keep in zip(gold_strings, in_mask) if keep])
    r3_raw = fe.score_open(result["r3_text"], gold_strings)
    outcomes = [readouts.r3_outcome(
        {"overlong": o, "empty": e, "text": t}, set(answers))
        for t, o, e in zip(result["r3_text"], result["r3_overlong"],
                           result["r3_empty"])]
    strip = lambda d: {k: v for k, v in d.items()  # noqa: E731
                       if not k.endswith("hits")}
    return {
        "R1_in_vocabulary": strip(r1),
        "R2_in_vocabulary": strip(r2),
        "R3_in_vocabulary": strip(r3_in),
        "R1_raw_denominator": fe.score_closed_raw_denominator(
            result["r1_pred"], gold_strings, answers,
            in_vocabulary=in_mask),
        "R2_raw_denominator": fe.score_closed_raw_denominator(
            result["r2_pred"], gold_strings, answers,
            in_vocabulary=in_mask),
        "R3_raw_denominator": strip(r3_raw),
        "R3_outcomes": readouts.summarise_r3_outcomes(outcomes),
        "seconds": result["seconds"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", nargs=3, required=True,
                        metavar=("ARM", "SCALE", "SEED"))
    args = parser.parse_args()
    arm, scale, seed = args.cell[0], args.cell[1], int(args.cell[2])

    utils.set_seed(0)
    e8b_run.enable_strict_determinism()
    e8b_run.pin_fp32_precision()
    determinism = e8b_run.assert_strict_determinism()

    entry = frozen_entry(arm, scale, seed)
    checkpoint = entry["_checkpoint_path"]
    run_name = entry["run"]
    out_path = OUT_DIR / f"{run_name}_final_evaluation.json"
    if out_path.exists():
        sys.exit(f"{out_path.name} already exists; scientific artefacts "
                 f"are never overwritten. Remove it deliberately if a "
                 f"re-evaluation is authorised.")

    # The resource gate, as frozen. The final evaluation is ALREADY
    # inside committed_not_yet_spent_hours for this identity, so nothing
    # is added here: adding it again would double-count reserved work
    # and compare a committed estimate against post-projection headroom
    # as though it were new.
    gate = e8b_run.per_identity_gate(arm, 0.0)
    if gate["fires"]:
        sys.exit(f"RESOURCE GATE REFUSES: {gate['fire_reason']}")
    print(f"[GATE] {gate['identity']}: {gate['already_charged_hours']} h "
          f"charged, {gate['committed_not_yet_spent_hours']} h committed "
          f"(this evaluation is inside that), headroom "
          f"{gate['headroom_hours']} h, fires False")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = pd.read_csv(V2_DIR / "dev_raw.csv",
                      dtype={"questionId": str, "imageId": str},
                      keep_default_na=False, na_filter=False)
    raw["label"] = raw["label"].astype(int)
    raw["in_vocabulary"] = raw["in_vocabulary"].astype(bool)
    stores = ve.ExtendedTokenStores()
    coverage = stores.covers(list(raw["questionId"]), list(raw["imageId"]))
    if not coverage["complete"]:
        sys.exit("the extended stores do not cover the raw denominator")

    answers, vocabulary = g21.load_index_to_answer(
        V2_DIR / "answer_vocab_v2.json")
    lm, provenance, tokenizer, cache, trie = None, None, None, None, None
    if arm != "B1":
        lm, provenance = e8b_run.load_frozen_causal_lm(
            arm == "B3", device=None)
        e8b_run.enable_strict_determinism()
        e8b_run.promote_lm_to_fp32(lm)
        lm = lm.to(device)
        tokenizer = e8a.load_tokenizer()
        cache = readouts.build_answer_cache(tokenizer, answers)
        trie = readouts.build_trie(cache)

    model = build_model(arm, seed, checkpoint, lm, device)
    loader = torch.utils.data.DataLoader(
        ve.RawRowDataset(raw, stores), batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=ve.collate)
    context = fe.build_intervention_context(
        loader, raw, stores.image_vector, device,
        e8b_training.canonical_prefix)

    in_mask = raw["in_vocabulary"].to_numpy()
    labels = raw["label"].to_numpy()
    normalisation_check = fe.assert_normalisation_disjoint(
        [g for g, keep in zip(list(raw["answer"]), in_mask) if not keep],
        answers)

    started = time.time()
    conditions, per_row_hits = {}, {}
    for condition in fe.CONDITIONS:
        mark = time.time()
        if arm == "B1":
            batches = intervened_batches(loader, condition, context,
                                         device)
            predicted, seen_labels, _ = \
                e8b_training.b1_canonical_predictions(model, batches,
                                                      device)
            if not np.array_equal(seen_labels, labels):
                sys.exit(f"{condition}: B1 label stream is not aligned "
                         f"with the raw denominator")
            closed = fe.score_closed(predicted[in_mask], labels[in_mask],
                                     answers)
            conditions[condition] = {
                "classifier_in_vocabulary":
                    {k: v for k, v in closed.items()
                     if not k.endswith("hits")},
                "classifier_raw_denominator":
                    fe.score_closed_raw_denominator(
                        predicted, list(raw["answer"]), answers,
                        in_vocabulary=in_mask),
                "seconds": round(time.time() - mark, 2),
                "readouts_defined_for_this_arm":
                    "classifier only; the canonical plan does not define "
                    "R1/R2/R3 for B1 and none is invented here"}
            per_row_hits[condition] = (predicted == labels).astype(np.uint8)
        else:
            result = fe.evaluate_condition(
                model, lm, loader, cache, trie, tokenizer, device,
                condition, context)
            conditions[condition] = score_lm_condition(
                result, raw, answers, in_mask)
            per_row_hits[condition] = (
                np.array(result["r1_pred"]) == labels).astype(np.uint8)
        print(f"  {condition:16s} {conditions[condition]['seconds']:8.1f} s")

    elapsed = time.time() - started
    npz_path = OUT_DIR / f"{run_name}_final_eval_per_row.npz"
    if npz_path.exists():
        sys.exit(f"{npz_path.name} already exists; refusing to overwrite")
    np.savez_compressed(npz_path, in_vocabulary=in_mask, labels=labels,
                        **{f"r1_hits_{c}": v
                           for c, v in per_row_hits.items()})

    body = {
        "dated_utc": "2026-08-09",
        "NO_TRAINING": True,
        "cell": [arm, scale, seed], "run": run_name,
        "manifest_basis_commit": MANIFEST_BASIS_COMMIT,
        "checkpoint": {
            "path": str(checkpoint.relative_to(PROJECT_ROOT)),
            "sha256": entry["checkpoints"]["canonical"]["sha256"],
            "rule": entry["canonical_checkpoint_rule"],
            "epoch": entry["canonical_epoch"]},
        "result_record_sha256": entry["result_record"]["sha256"],
        "evaluator_source_digest": evaluator_digest(),
        "driver": "experiments/e8b_readout_generation/"
                  "run_final_evaluation.py (orchestration only)",
        "protocol_family": entry["protocol_family"],
        "denominators": {
            "raw_rows": int(len(raw)),
            "in_vocabulary_rows": int(in_mask.sum()),
            "out_of_vocabulary_rows": int((~in_mask).sum()),
            "image_clusters_raw": int(raw["imageId"].nunique()),
            "coverage": round(float(in_mask.mean()), 6),
            "restriction_optimisation": "REJECTED and not used; every "
                                        "readout runs on the full raw "
                                        "denominator"},
        "conditions": conditions,
        "normalisation_check": normalisation_check,
        "determinism": determinism,
        "frozen_lm": provenance,
        "per_row_artefact": {
            "path": str(npz_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(npz_path)},
        "wall_seconds": round(elapsed, 2),
        "resource_gate_before": gate,
        "clean_test_accessed": False}
    e8b_run.atomic_write_json(
        out_path, {"metadata": utils.run_metadata(),
                   "e8b_final_evaluation": body})
    print(f"[EVAL DONE] {run_name}: {elapsed / 3600:.3f} h -> "
          f"{out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
