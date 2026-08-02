"""E8A Phase-1B: arm A0p at train_40k, seed 0 only.

A0p is the interface-matched CLIP question-token control of E8A plan section 2
and master protocol section 8.0: the same token-level CLIP question sequence
the stored A0 reasoner consumes, plus **one trainable Linear(512, 512)**, so
that `A1 - A0p` differs in the question representation source and nothing else.

    CLIP question tokens [B, L, 512]
      -> valid-token selection and key-padding mask
      -> one trainable token-wise Linear(512, 512), 262,656 parameters
      -> [B, L, 512]
      -> the unmodified latent-query reasoner of src/reasoner.py
      -> 100-way answer classifier

A0p is **not** the stored A0 artefact. A0 has no projection and 21,099,620
trainable parameters; A0p has 21,362,276. The stored A0 checkpoint is never
loaded, never used as an initialisation, and never overwritten.

Every gate, training and evaluation function is imported from run_pilot.py and
used unchanged, so A0p executes the same code path as A1 and A1r. That is the
strongest available form of interface matching: the arms differ in their
question store and in the projection's input width, and in nothing else.

Usage, from the project root with the venv active:

    python -B experiments/e8a_question_encoder/run_a0p.py --gates
    python -B experiments/e8a_question_encoder/run_a0p.py --train

Scope, from the Phase-1B authorisation: A0p only, train_40k only, seed 0 only.
Seeds 1 and 2, train_250k, the full core matrix, SmolLM2-360M, E8B, E9, the
clean test, F1 and F2 are all outside this script. A1 and A1r are NOT retrained.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import run_pilot  # noqa: E402

ARM = e8a.A0P_ARM
SEED = run_pilot.PILOT_SEED
SCALE = run_pilot.PILOT_SCALE

# A0p writes only to these paths. Asserted before anything runs, so an A0, A1
# or A1r artefact cannot be overwritten.
PROTECTED_PREFIXES = ("e8a_A1_", "e8a_A1r_", "reasoner_seed", "overfit_A1")


def assert_namespace_isolation() -> dict:
    """A0p must not write over any A0, A1 or A1r artefact."""
    a0_dir = config.RESULTS_DIR / "experiments" / "v3_01_reasoner"
    a0_checkpoints = sorted(p.name
                            for p in (a0_dir / "checkpoints").glob("*.pt"))
    a0_hashes = {p.name: e8a.sha256_file(p)
                 for p in sorted((a0_dir / "checkpoints").glob("reasoner_*.pt"))}
    ours = {
        "checkpoint": e8a.OUT_DIR / "checkpoints"
        / f"e8a_{ARM}_{SCALE}_seed{SEED}.pt",
        "gates": e8a.OUT_DIR / f"gates_{ARM}.json",
        "pilot": e8a.OUT_DIR / f"pilot_{ARM}.json",
        "correctness": e8a.OUT_DIR / f"correctness_{ARM}_seed{SEED}.npz",
    }
    for name, path in ours.items():
        assert not any(str(path.name).startswith(p)
                       for p in PROTECTED_PREFIXES), (name, path)
    print(f"[PASS] namespace isolation: A0p writes only "
          f"{sorted(p.name for p in ours.values())}")
    return {"a0p_outputs": {k: str(v.relative_to(PROJECT_ROOT))
                            for k, v in ours.items()},
            "a0_checkpoints_present": a0_checkpoints,
            "a0_checkpoint_sha256_before": a0_hashes,
            "a1_a1r_sha256_before": {
                p.name: e8a.sha256_file(p)
                for p in sorted(e8a.OUT_DIR.rglob("*A1*")) if p.is_file()},
            "passed": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gates", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if not (args.gates or args.train):
        parser.error("pass --gates or --train")

    utils.set_seed()
    device = utils.get_device()
    e8a.OUT_DIR.mkdir(parents=True, exist_ok=True)

    gate_record: dict = {}

    def persist_and_reraise(error):
        """Canonical section 18: a gate failure is recorded verbatim."""
        gate_record["failed_with"] = f"{type(error).__name__}: {error}"
        utils.save_json({"metadata": utils.run_metadata(),
                         "e8a_a0p_gates_partial": gate_record},
                        e8a.OUT_DIR / f"FAILED_gates_{ARM}.json")
        print(f"\nGATE FAILURE recorded to results/experiments/"
              f"e8a_question_encoder/FAILED_gates_{ARM}.json")
        raise error

    try:
        recipe = e8a.gate_g0_recipe()
        gate_record.update({
        "arm": ARM,
        "recipe": recipe,
        "g17_embargo": run_pilot.gate_g17_embargo(),
        "g12_read_only": run_pilot.gate_g12_read_only(),
        "namespace_isolation": assert_namespace_isolation(),
        })
        gate_record["g1_g18_vocabulary"] = run_pilot.gate_g1_g18_vocabulary(
            [e8a.V2_DIR / f"{SCALE}.csv", e8a.V2_DIR / "dev.csv"])
        gate_record["g15_manifests"] = run_pilot.gate_g15_manifests(
            [e8a.V2_DIR / f"{SCALE}.csv", e8a.V2_DIR / "dev.csv",
             e8a.V2_DIR / "answer_vocab_v2.json",
             e8a.CLIP_TOKEN_DIR / "image_tokens.h5",
             e8a.CLIP_TOKEN_DIR / "question_tokens.h5",
             config.RESULTS_DIR / "experiments" / "v3_01_reasoner"
             / "results.json"])

        print("loading stores")
        images = e8a.ImageTokenStore()
        questions = {ARM: e8a.open_clip_question_store()}
        store = questions[ARM]
        assert store.states.shape[1] == e8a.CLIP_QUESTION_WIDTH
        assert "EOT position + 1" in store.attrs["length_convention"]
        assert int(store.lengths.max()) <= e8a.TOKEN_BUDGET_L, \
            int(store.lengths.max())
        gate_record["store_attributes"] = {ARM: dict(store.attrs)}
        print(f"  CLIP question store: {store.states.shape[0]:,} packed token "
              f"rows at width {store.states.shape[1]}, "
              f"{len(store.row_of):,} questionIds")

        # G13 over all three arms at once, so the trunk identity that makes the
        # three-way comparison meaningful is asserted rather than assumed.
        gate_record["g13_construction"] = run_pilot.gate_g13_construction(
            recipe["dropout"], SEED, arms=("A0p", "A1", "A1r"))

        dataset = e8a.E8ATokenDataset(e8a.V2_DIR / f"{SCALE}.csv", images, store)
        probe_loader = DataLoader(dataset, batch_size=recipe["batch_size"],
                                  shuffle=False, collate_fn=e8a.collate_e8a)
        e8a.prepare_encoder_for_build(ARM)
        probe_model = e8a.build_e8a_model(e8a.arm_d_question(ARM),
                                          recipe["dropout"], SEED)
        gate_record["g4_g5_forward_and_mask"] = run_pilot.gate_g4_g5(
            probe_model, probe_loader, device)
        parameters = e8a.parameter_report(probe_model.to("cpu"))
        assert parameters["trainable_projection"] == e8a.A0P_PROJECTION_PARAMETERS
        assert parameters["trainable_total"] == 21_362_276, parameters
        print(f"[PASS] A0p trainable parameters {parameters['trainable_total']:,} "
              f"= {parameters['trainable_reasoner_trunk']:,} trunk + "
              f"{parameters['trainable_projection']:,} projection, matching the "
              f"canonical Linear(512, 512) figure of "
              f"{e8a.A0P_PROJECTION_PARAMETERS:,}")
        gate_record["parameters"] = parameters
        del probe_model
        torch.cuda.empty_cache()

        gate_record["scheduler_formula"] = run_pilot.gate_scheduler_formula(
            math.ceil(len(dataset) / recipe["batch_size"]),
            recipe["final_max_epochs"], recipe["warmup_frac"],
            recipe["learning_rate"])

        gate_record["g8_tiny_overfit"] = {
            ARM: run_pilot.gate_g8_tiny_overfit(ARM, recipe["dropout"], SEED,
                                                dataset, device)}

        print("\n=== pinned intervention tensors ===")
        neutral_image, neutral_image_provenance = e8a.build_neutral_image_tokens(
            images)
        gate_record["pinned_neutral_image"] = neutral_image_provenance
        print(f"  neutral image tokens {neutral_image.shape} over "
              f"{neutral_image_provenance['n_training_images']} training images, "
              f"sha256 {neutral_image_provenance['sha256'][:16]}...")
        neutral_question = {ARM: e8a.build_neutral_question_states_clip(device)}
        provenance = neutral_question[ARM][2]
        gate_record["pinned_neutral_question"] = {ARM: provenance}
        print(f"  neutral question states, arm {ARM}: "
              f"{provenance['valid_positions']} valid of L={provenance['L']} "
              f"({provenance['length_convention']}), sha256 "
              f"{provenance['sha256'][:16]}...")

    except Exception as error:                       # noqa: BLE001
        persist_and_reraise(error)

    utils.save_json({"metadata": utils.run_metadata(),
                     "e8a_a0p_gates": gate_record},
                    e8a.OUT_DIR / f"gates_{ARM}.json")
    if args.gates:
        print("\ngates complete; no training run (--gates)")
        return 0

    run = run_pilot.train_arm(ARM, recipe, SEED, images, questions, device)
    utils.save_json({"metadata": utils.run_metadata(seed=SEED),
                     "e8a_a0p_run": run},
                    e8a.OUT_DIR / f"run_partial_{ARM}.json")
    evaluation = run_pilot.evaluate_arm(ARM, recipe, SEED, run, images,
                                        questions, neutral_image,
                                        neutral_question, device)

    after = {p.name: e8a.sha256_file(p) for p in sorted(
        (config.RESULTS_DIR / "experiments" / "v3_01_reasoner"
         / "checkpoints").glob("reasoner_*.pt"))}
    assert after == gate_record["namespace_isolation"][
        "a0_checkpoint_sha256_before"], "an A0 checkpoint changed"
    a1_after = {p.name: e8a.sha256_file(p)
                for p in sorted(e8a.OUT_DIR.rglob("*A1*")) if p.is_file()}
    assert a1_after == gate_record["namespace_isolation"][
        "a1_a1r_sha256_before"], "an A1 or A1r artefact changed"
    print(f"[PASS] every stored A0 checkpoint and all {len(a1_after)} A1/A1r "
          f"artefacts are byte-identical after the run")

    record = {
        "metadata": utils.run_metadata(seed=SEED),
        "e8a_a0p_pilot": {
            "phase": "Phase 1B bounded pilot",
            "arm": ARM,
            "scope": {"arms": [ARM], "scale": SCALE, "seeds": [SEED],
                      "not_run": ["A0p seed 1", "A0p seed 2",
                                  "A0p train_250k", "any further A1 or A1r "
                                  "run", "the full E8A core matrix",
                                  "SmolLM2-360M", "E8B", "E9", "clean test",
                                  "F1", "F2"]},
            "interface": {
                "description": "frozen CLIP question tokens [B, L, 512] -> "
                               "valid-token selection and key-padding mask -> "
                               "one trainable token-wise Linear(512, 512) -> "
                               "[B, L, 512] -> unmodified latent-query "
                               "reasoner -> 100-way classifier",
                "pooled": False,
                "reasoner_source_file_modified": False,
                "differs_from_A0_in": "exactly one respect: the presence of "
                                      "the trainable Linear(512, 512), "
                                      "262,656 parameters. A0 has no "
                                      "projection and 21,099,620 trainable "
                                      "parameters.",
                "differs_from_A1_in": "exactly one respect: the question "
                                      "representation source, and hence the "
                                      "projection's input width, 512 against "
                                      "576.",
            },
            "fixed_question_caveat":
                "the pinned neutral question is the CLIP token states of the "
                "string \"question\", which is 3 valid positions (SOT, the "
                "word, EOT) against a mean of about 12.2 CLIP tokens for real "
                "questions. normal minus fixed-question therefore confounds "
                "question content with a sequence-length change, and it is "
                "NOT a pure semantic contribution. The asymmetry across arms "
                "is larger still: A0p has 3 valid positions where A1 and A1r "
                "have 1, because each arm's neutral sequence comes from its "
                "own tokenizer, exactly as canonical section 11.1 specifies.",
            "system_comparison_label":
                "Matched system comparisons under a recipe historically "
                "selected on the CLIP-question-token reasoner. This selection "
                "asymmetry favours A0p and makes the comparison conservative "
                "with respect to the SLM arm.",
            "claim_scope":
                "a result about the frozen model under the pre-registered "
                "post-final-norm token-sequence interface at L = 32. Never "
                "generalised to 'small language models do not help', nor to "
                "'small language models help'.",
            "gates": gate_record,
            "run": run,
            "evaluation": evaluation,
            "a0_artefacts_unchanged": after,
            "a1_a1r_artefacts_unchanged": a1_after,
            "clean_test_accessed": False,
        },
    }
    utils.save_json(record, e8a.OUT_DIR / f"pilot_{ARM}.json")
    print(f"\nA0p pilot record written to "
          f"results/experiments/e8a_question_encoder/pilot_{ARM}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
