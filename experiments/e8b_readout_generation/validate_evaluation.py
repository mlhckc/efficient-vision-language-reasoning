"""NON-SCIENTIFIC validation of the final evaluation pipeline.

Runs every path the pipeline will take after a core cell completes --
R1, R2 and R3, on both denominators, scored both ways, under all four
matched conditions -- against a FROZEN EXPLORATORY checkpoint from the
abandoned search, on a small bounded subset.

The point is to prove the machinery executes and produces
well-formed, self-consistent records. It is NOT a result. The checkpoint
belongs to the superseded BF16 search family, carries no core protocol
family, and every artefact written here is marked NON_SCIENTIFIC so the
promotion guard refuses it. No accuracy from this run may appear
anywhere near a core result.

NO OPTIMIZER STEP. No training, no backward pass, no parameter update.

THE EMBARGOED CLEAN TEST IS NEVER TOUCHED.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils, tokens_data  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402
from experiments.e8b_readout_generation import training as e8b_training  # noqa: E402
from experiments.e8b_readout_generation import final_evaluation as fe  # noqa: E402

V2_DIR = PROJECT_ROOT / "data" / "v2"
TOKEN_DIR = config.DATA_DIR / "v3" / "tokens"
EXPLORATORY = (e8b_run.OUT_DIR / "checkpoints"
               / "e8b_B3_train_40k_seed0_search1_best.pt")
RECORD = e8b_run.OUT_DIR / "evaluation_pipeline_validation_20260807.json"


class ExtendedTokenStores(tokens_data.TokenStores):
    """The pinned stores plus the raw-denominator extension.

    Kept here rather than in src/tokens_data.py so the pinned stores and
    the module that reads them stay exactly as the core cells hash them.
    The extension is consulted only for identifiers the pinned stores do
    not have, so a row that exists in both always resolves to the pinned
    value and the training path can never see a different number."""

    def __init__(self):
        super().__init__()
        self.extension_question_tokens = None
        self.extension_image_tokens = None
        self.extension_question_index = {}
        self.extension_image_row = {}
        question_path = TOKEN_DIR / "question_tokens_raw_dev_extension.h5"
        image_path = TOKEN_DIR / "image_tokens_raw_dev_extension.h5"
        if question_path.exists():
            with h5py.File(question_path, "r") as store:
                ids = [i.decode("utf-8") for i in store["ids"][:]]
                self.extension_question_tokens = store["tokens"][:]
                self._ext_offsets = store["offsets"][:]
                self._ext_lengths = store["lengths"][:]
            self.extension_question_index = {
                qid: index for index, qid in enumerate(ids)}
        if image_path.exists():
            with h5py.File(image_path, "r") as store:
                ids = [i.decode("utf-8") for i in store["ids"][:]]
                self.extension_image_tokens = store["tokens"][:]
            self.extension_image_row = {
                image_id: index for index, image_id in enumerate(ids)}

    def image_vector(self, image_id: str):
        if image_id in self.image_row:
            return self.image_tokens[self.image_row[image_id]]
        return self.extension_image_tokens[
            self.extension_image_row[image_id]]

    def question_vector(self, qid: str):
        if qid in self.question_index:
            offset, length = self.question_span(qid)
            return self.question_tokens[offset:offset + length]
        index = self.extension_question_index[qid]
        offset = int(self._ext_offsets[index])
        length = int(self._ext_lengths[index])
        return self.extension_question_tokens[offset:offset + length]

    def covers(self, question_ids, image_ids) -> dict:
        missing_q = [q for q in question_ids
                     if q not in self.question_index
                     and q not in self.extension_question_index]
        missing_i = [i for i in image_ids
                     if i not in self.image_row
                     and i not in self.extension_image_row]
        return {"missing_questions": missing_q,
                "missing_images": missing_i,
                "complete": not missing_q and not missing_i}


class RawRowDataset(torch.utils.data.Dataset):
    """Rows of the raw manifest served from the extended stores."""

    def __init__(self, frame, stores: ExtendedTokenStores):
        self.frame = frame.reset_index(drop=True)
        self.stores = stores

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = torch.from_numpy(
            np.asarray(self.stores.image_vector(row["imageId"]),
                       dtype=np.float32))
        question = torch.from_numpy(
            np.asarray(self.stores.question_vector(row["questionId"]),
                       dtype=np.float32))
        return (image, question, row["questionId"],
                question.shape[0], int(row["label"]))


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    lengths = [b[3] for b in batch]
    width = max(lengths)
    questions = torch.zeros(len(batch), width, batch[0][1].shape[1])
    mask = torch.ones(len(batch), width, dtype=torch.bool)
    for index, item in enumerate(batch):
        questions[index, :lengths[index]] = item[1]
        mask[index, :lengths[index]] = False
    ids = [b[2] for b in batch]
    labels = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return images, questions, ids, mask, labels


def main(rows: int = 64) -> int:
    utils.set_seed(0)
    e8b_run.enable_strict_determinism()
    e8b_run.pin_fp32_precision()
    determinism = e8b_run.assert_strict_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not EXPLORATORY.exists():
        sys.exit(f"no exploratory checkpoint at {EXPLORATORY.name}")

    raw = pd.read_csv(V2_DIR / "dev_raw.csv",
                      dtype={"questionId": str, "imageId": str},
                      keep_default_na=False, na_filter=False)
    raw["label"] = raw["label"].astype(int)
    raw["in_vocabulary"] = raw["in_vocabulary"].astype(bool)
    stores = ExtendedTokenStores()
    coverage = stores.covers(list(raw["questionId"]), list(raw["imageId"]))
    if not coverage["complete"]:
        sys.exit(f"the extended stores do not cover the raw denominator: "
                 f"{len(coverage['missing_questions'])} questions and "
                 f"{len(coverage['missing_images'])} images missing")

    # A bounded subset that DELIBERATELY spans both denominators, so the
    # out-of-vocabulary path is exercised rather than skipped.
    in_vocab = raw[raw["in_vocabulary"]].head(rows // 2)
    out_vocab = raw[~raw["in_vocabulary"]].head(rows // 2)
    subset = pd.concat([in_vocab, out_vocab]).reset_index(drop=True)

    lm, provenance = e8b_run.load_frozen_causal_lm(True, device=None)
    e8b_run.enable_strict_determinism()
    e8b_run.promote_lm_to_fp32(lm)
    lm = lm.to(device)
    tokenizer = e8a.load_tokenizer()
    answers, vocabulary = g21.load_index_to_answer(
        V2_DIR / "answer_vocab_v2.json")
    cache = readouts.build_answer_cache(tokenizer, answers)
    trie = readouts.build_trie(cache)

    model, _ = e8b_run.build_arm("B3", 0, 0.1, lm)
    e8b_run.enable_strict_determinism()
    state = torch.load(EXPLORATORY, map_location="cpu",
                       weights_only=False)
    model.load_state_dict(state["state"] if "state" in state else state)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    loader = torch.utils.data.DataLoader(
        RawRowDataset(subset, stores), batch_size=16, shuffle=False,
        collate_fn=collate)

    # The SHARED builder, so the validated path and the core path are
    # the same path.
    context = fe.build_intervention_context(
        loader, subset, stores.image_vector, device,
        e8b_training.canonical_prefix)
    neutral_image = context["neutral_image"]
    neutral_question = context["neutral_question"]
    derangement = context["derangement"]

    conditions = {}
    for condition in fe.CONDITIONS:
        started = time.time()
        conditions[condition] = fe.evaluate_condition(
            model, lm, loader, cache, trie, tokenizer, device,
            condition, context)
        print(f"  {condition:16s} {time.time() - started:6.1f} s")

    # --- scoring, both denominators, both scorers ---
    index_to_answer = answers
    gold_strings = list(subset["answer"])
    labels = np.array(subset["label"])
    in_mask = np.array(subset["in_vocabulary"])
    # The precondition for the normalised coverage identity, CHECKED on
    # this data before any raw-denominator number is derived from it.
    normalisation_check = fe.assert_normalisation_disjoint(
        [g for g, keep in zip(gold_strings, in_mask) if not keep],
        index_to_answer)
    scored = {}
    for condition, result in conditions.items():
        r1 = fe.score_closed(np.array(result["r1_pred"])[in_mask],
                             labels[in_mask], index_to_answer)
        r2 = fe.score_closed(np.array(result["r2_pred"])[in_mask],
                             labels[in_mask], index_to_answer)
        r3_in = fe.score_open(
            [t for t, keep in zip(result["r3_text"], in_mask) if keep],
            [g for g, keep in zip(gold_strings, in_mask) if keep])
        r3_raw = fe.score_open(result["r3_text"], gold_strings)
        outcomes = [readouts.r3_outcome(
            {"overlong": o, "empty": e, "text": t}, set(answers))
            for t, o, e in zip(result["r3_text"], result["r3_overlong"],
                               result["r3_empty"])]
        scored[condition] = {
            "R1_in_vocabulary": {k: v for k, v in r1.items()
                                 if not k.endswith("hits")},
            "R2_in_vocabulary": {k: v for k, v in r2.items()
                                 if not k.endswith("hits")},
            # HB3b: DIRECT over every row, with the coverage identity
            # kept only as a cross-check that must agree.
            "R1_raw_denominator": fe.score_closed_raw_denominator(
                result["r1_pred"], gold_strings, index_to_answer,
                in_vocabulary=in_mask),
            "R2_raw_denominator": fe.score_closed_raw_denominator(
                result["r2_pred"], gold_strings, index_to_answer,
                in_vocabulary=in_mask),
            "R3_in_vocabulary": {k: v for k, v in r3_in.items()
                                 if not k.endswith("hits")},
            "R3_raw_denominator": {k: v for k, v in r3_raw.items()
                                   if not k.endswith("hits")},
            "R3_outcomes": readouts.summarise_r3_outcomes(outcomes),
            "seconds": result["seconds"]}

    # --- the paired contrast machinery, exercised on a synthetic pair ---
    normal = conditions["normal"]
    hits_a = (np.array(normal["r1_pred"]) == labels)
    hits_b = (np.array(conditions["fixed_image"]["r1_pred"]) == labels)
    contrast = fe.paired_contrast(hits_a, hits_b, list(subset["imageId"]),
                                  label_a="normal",
                                  label_b="fixed_image", bootstrap=2000)

    # --- per-row records ---
    per_row = e8b_run.OUT_DIR / "validation_per_row_20260807.npz"
    if per_row.exists():
        per_row.unlink()
    np.savez_compressed(
        per_row,
        questionId=np.array(subset["questionId"], dtype="S32"),
        imageId=np.array(subset["imageId"], dtype="S32"),
        label=labels, in_vocabulary=in_mask,
        **{f"{c}_r1": np.array(conditions[c]["r1_pred"])
           for c in fe.CONDITIONS},
        **{f"{c}_r2": np.array(conditions[c]["r2_pred"])
           for c in fe.CONDITIONS},
        **{f"{c}_r3": np.array(conditions[c]["r3_text"], dtype=object)
           for c in fe.CONDITIONS})

    record = {"metadata": utils.run_metadata(),
              "e8b_evaluation_pipeline_validation": {
        "NON_SCIENTIFIC": True,
        "dated_utc": "2026-08-07",
        "status": "NON-SCIENTIFIC IMPLEMENTATION VALIDATION - NOT A "
                  "RESULT",
        "why_not_a_result": "the checkpoint is a FROZEN EXPLORATORY "
                            "artefact of the permanently abandoned "
                            "eight-point search. It carries no core "
                            "protocol family, it was trained under the "
                            "superseded BF16 design, and it was "
                            "selected by a rule the fixed-endpoint "
                            "amendment withdrew. Every accuracy below "
                            "exists only to show the machinery runs, "
                            "and none may be quoted, ranked or placed "
                            "beside a core result.",
        "checkpoint": {"path": str(EXPLORATORY.relative_to(PROJECT_ROOT)),
                       "sha256": hashlib.sha256(
                           EXPLORATORY.read_bytes()).hexdigest()},
        "rows_evaluated": int(len(subset)),
        "in_vocabulary_rows": int(in_mask.sum()),
        "out_of_vocabulary_rows": int((~in_mask).sum()),
        "conditions_executed": list(fe.CONDITIONS),
        "readouts_executed": list(fe.READOUTS),
        "normalisation_check": normalisation_check,
        "scored": scored,
        "paired_contrast_demo": contrast,
        "derangement": derangement,
        "neutral_inputs": {
            "basis": fe.NEUTRAL_BASIS,
            "image_sha256": fe.sha256_array(
                neutral_image.cpu().numpy()),
            "question_sha256": fe.sha256_array(
                neutral_question.cpu().numpy())},
        "per_row_records": str(per_row.relative_to(PROJECT_ROOT)),
        "determinism_verified_at_use": determinism,
        "frozen_model": {"arm_model": provenance["arm_model"],
                         "pinned_revision": provenance["pinned_revision"]},
        "optimizer_steps": 0,
        "clean_test_accessed": False}}
    if RECORD.exists():
        RECORD.unlink()
    RECORD.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"\nwritten {RECORD.name} (NON_SCIENTIFIC)")
    for condition in fe.CONDITIONS:
        s = scored[condition]
        print(f"  {condition:16s} R1 {s['R1_in_vocabulary']['raw_exact']:.4f}"
              f"  R2 {s['R2_in_vocabulary']['raw_exact']:.4f}"
              f"  R3raw {s['R3_raw_denominator']['raw_exact']:.4f}"
              f"  invalid {s['R3_outcomes']['counts'].get('overlong', 0)}"
              f"/{s['R3_outcomes']['counts'].get('empty', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
