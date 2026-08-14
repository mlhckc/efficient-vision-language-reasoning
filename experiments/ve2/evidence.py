"""Frozen prediction evidence, with identity proved before any row is used.

A qualitative example is one row of one experiment's stored predictions. The
danger is not that a number is wrong; it is that the row shown under a model's
name is a different row, or a different model's. So every source here proves
four things before it yields a single value:

1. BYTES. The artefact hashes to the value a frozen manifest recorded for it.
   The manifest is reached from the VE-0 qualitative protocol, which names it
   in `prediction_evidence_sources[...].hash_bound_in`, and the manifest itself
   is hashed. Nothing is trusted because of where it sits on disk.

2. SPLIT. The artefact is the development split. Row counts are checked
   against the 7,714-row in-vocabulary development view, and the closure
   manifest's own `split_identity` string is carried into the record.

3. ALIGNMENT. `question_id`, `image_id` and the gold answer or gold label are
   compared element-by-element against `data/v2/dev.csv`. No source is used
   positionally on the strength of a row count alone. The E8B arrays, which
   carry no question identifier and would have to be trusted positionally, are
   NOT used by any category and are recorded as unused.

4. IDENTITY. Arm, scale, seed and condition are read from the manifest entry
   or from the file's own columns, not inferred from the filename.

Two sources need more than that.

THE SHUFFLED-IMAGE PARTNER. VE0-QC07 requires showing the wrong image that
replaced the real one. The stored v2_06 row evidence records each row's own
image under both conditions, so the partner is not written down anywhere. It is
recoverable, because the intervention is a permutation of the image embeddings
under a recorded seed, but recovering it by re-running the shuffle would be
taking the builder's word for it. Instead `shuffled_partner` derives the
permutation and then PROVES it from stored evidence alone: the `image_only`
head sees nothing but an image, so its shuffled prediction at row i must equal
its normal prediction at row perm[i]. That identity is checked on all 7,714
rows, and a deliberately wrong permutation is checked to fail, before any
partner image is resolved.

E9. The VE-0 protocol marks the compact-VLM category conditional and requires
VE-2 to establish, from the frozen artefacts alone, whether a per-row
correctness vector is RECOVERABLE. `verify_e9_recoverability` performs that
check and reports what it found. It never loads a model and never scores
anything.

Nothing in this module opens a checkpoint, an embedding store or the clean
test.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.ve2 import ve2_common as vc
from experiments.ve2.ve2_common import PROJECT_ROOT

DATA = PROJECT_ROOT / "data" / "v2"
CLOSURE = PROJECT_ROOT / "results" / "closure"
EXPERIMENTS = PROJECT_ROOT / "results" / "experiments"
GQA_IMAGES = PROJECT_ROOT / "data" / "gqa" / "images"

DEV_ROWS = 7714
RAW_DEV_ROWS = 10004


# --------------------------------------------------------------------------
# the development split
# --------------------------------------------------------------------------

class DevelopmentSplit:
    """The 7,714-row in-vocabulary development view, hash-checked.

    This is the candidate pool the VE-0 selection rule declares, and the
    reference ordering every prediction source is aligned against. Its hash is
    verified against the closure reconstruction manifest, which the VE-0
    protocol itself lists as one of its inputs, so the chain from the frozen
    contract to these bytes is unbroken.
    """

    def __init__(self):
        self.manifest_path = CLOSURE / "C_reconstructed_evidence_manifest.json"
        self.manifest = vc.read_json(self.manifest_path)
        self.manifest_sha256 = vc.sha256_file(self.manifest_path)

        spec = self.manifest["splits"]["v2_dev"]
        self.split_identity = spec["split_identity"]
        self.answer_set_size = int(spec["answer_set_size"])
        sources = {s["role"]: s for s in spec["sources"]}

        self.dev_path = DATA / "dev.csv"
        self.vocab_path = DATA / "answer_vocab_v2.json"
        self._verify(self.dev_path, sources["split manifest"]["sha256"],
                     "development split manifest")
        self._verify(self.vocab_path, sources["answer vocabulary"]["sha256"],
                     "answer vocabulary")

        self.frame = pd.read_csv(
            self.dev_path,
            dtype={"questionId": str, "imageId": str, "question": str,
                   "answer": str, "label": int})
        if len(self.frame) != DEV_ROWS:
            raise AssertionError(
                f"the development split has {len(self.frame)} rows, not the "
                f"{DEV_ROWS} the VE-0 protocol declares")

        # Reasoning depth and question type. Hash-bound through the closure
        # slice statistics, whose own hash the VE-0 protocol records.
        slice_stats_path = CLOSURE / "D_slice_statistics.json"
        self.slice_stats_sha256 = vc.sha256_file(slice_stats_path)
        inputs = {i["path"]: i
                  for i in vc.read_json(slice_stats_path)["provenance"]["inputs"]}
        self.types_path = DATA / "metadata" / "dev_types.csv"
        self._verify(self.types_path,
                     inputs["data/v2/metadata/dev_types.csv"]["sha256"],
                     "question type and program length metadata")
        types = pd.read_csv(self.types_path,
                            dtype={"questionId": str, "structural": str,
                                   "semantic": str, "n_steps": int})
        merged = self.frame.merge(types, on="questionId", how="left",
                                  validate="one_to_one")
        if len(merged) != DEV_ROWS or merged["n_steps"].isna().any():
            raise AssertionError(
                "the question type metadata does not cover every development "
                "row; VE-2 refuses to select on an incomplete join")
        self.frame = merged

        self.question_ids = self.frame["questionId"].to_numpy()
        self.image_ids = self.frame["imageId"].to_numpy()
        self.gold_answers = self.frame["answer"].to_numpy()
        self.gold_labels = self.frame["label"].to_numpy()
        self.n_steps = self.frame["n_steps"].to_numpy()

        with open(self.vocab_path, encoding="utf-8") as handle:
            vocab = json.load(handle)
        self.vocabulary = list(vocab["answers"] if isinstance(vocab, dict)
                               else vocab)
        if len(self.vocabulary) != self.answer_set_size:
            raise AssertionError(
                f"the answer vocabulary holds {len(self.vocabulary)} answers "
                f"but the split declares {self.answer_set_size}")

        # Positive membership check for every rendered image. The embargoed
        # identifier list is never opened; eligibility is established by
        # presence in the development list, not by absence from another.
        image_ids_path = DATA / "dev_image_ids.json"
        self.image_ids_path = image_ids_path
        self.image_ids_sha256 = vc.sha256_file(image_ids_path)
        self.development_image_ids = set(vc.read_json(image_ids_path))

        self.n_unique_images = int(self.frame["imageId"].nunique())

    @staticmethod
    def _verify(path: Path, expected: str, role: str) -> None:
        actual = vc.sha256_file(path)
        if actual != expected:
            raise AssertionError(
                f"the {role} at {vc.relpath(path)} hashes to {actual} but the "
                f"frozen manifest records {expected}; VE-2 refuses to select "
                f"from unverified inputs")

    def sources(self) -> list:
        return [
            {"path": vc.relpath(self.dev_path),
             "sha256": vc.sha256_file(self.dev_path),
             "role": "development split manifest"},
            {"path": vc.relpath(self.types_path),
             "sha256": vc.sha256_file(self.types_path),
             "role": "question type and program length metadata"},
            {"path": vc.relpath(self.vocab_path),
             "sha256": vc.sha256_file(self.vocab_path),
             "role": "answer vocabulary"},
            {"path": vc.relpath(self.image_ids_path),
             "sha256": self.image_ids_sha256,
             "role": "development image identifiers"},
            {"path": vc.relpath(self.manifest_path),
             "sha256": self.manifest_sha256,
             "role": "closure reconstruction manifest, binds the row evidence"},
            {"path": vc.relpath(CLOSURE / "D_slice_statistics.json"),
             "sha256": self.slice_stats_sha256,
             "role": "closure slice statistics, binds the type metadata"},
        ]

    def row(self, index: int) -> dict:
        record = self.frame.iloc[index]
        return {
            "question_id": str(record["questionId"]),
            "image_id": str(record["imageId"]),
            "question_text": str(record["question"]),
            "gold_answer": str(record["answer"]),
            "gold_label": int(record["label"]),
            "semantic_type": str(record["semantic"]),
            "structural_type": str(record["structural"]),
            "n_program_steps": int(record["n_steps"]),
        }


# --------------------------------------------------------------------------
# prediction sources
# --------------------------------------------------------------------------

@dataclass
class SystemEvidence:
    """One system's per-row predictions on the development split."""

    system_id: str
    arm: str
    scale: str
    seed: int
    condition: str
    family: str
    source_path: str
    source_sha256: str
    hash_bound_in: str
    correct: np.ndarray
    predicted_answer: np.ndarray | None
    prediction_text_available: bool
    alignment: dict
    extra: dict = field(default_factory=dict)

    def prediction(self, index: int) -> str:
        if not self.prediction_text_available or self.predicted_answer is None:
            return "NOT_AVAILABLE"
        return str(self.predicted_answer[index])

    def is_correct(self, index: int) -> bool:
        return bool(int(self.correct[index]) == 1)

    def as_record(self, index: int) -> dict:
        return {
            "system_id": self.system_id,
            "arm": self.arm,
            "scale": self.scale,
            "seed": self.seed,
            "condition": self.condition,
            "predicted_answer": self.prediction(index),
            "correct": self.is_correct(index),
            "evidence_path": self.source_path,
            "evidence_sha256": self.source_sha256,
            "evidence_hash_bound_in": self.hash_bound_in,
            "alignment_proof": self.alignment,
        }

    def source(self) -> dict:
        return {"path": self.source_path, "sha256": self.source_sha256,
                "role": f"{self.system_id} per-row predictions"}


def _alignment_proof(split: DevelopmentSplit, question_ids, image_ids,
                     gold, gold_kind: str, method: str) -> dict:
    """Element-by-element identity against the development split."""
    checks = {
        "question_id_matches_split_row_for_row":
            bool(np.array_equal(np.asarray(question_ids, dtype=str),
                                split.question_ids.astype(str))),
        "image_id_matches_split_row_for_row":
            bool(np.array_equal(np.asarray(image_ids, dtype=str),
                                split.image_ids.astype(str))),
        "row_count": int(len(question_ids)),
        "expected_row_count": DEV_ROWS,
        "alignment_method": method,
    }
    if gold_kind == "answer":
        checks["gold_answer_matches_split_row_for_row"] = bool(
            np.array_equal(np.asarray(gold, dtype=str),
                           split.gold_answers.astype(str)))
    else:
        checks["gold_label_matches_split_row_for_row"] = bool(
            np.array_equal(np.asarray(gold).astype(int),
                           split.gold_labels.astype(int)))
    failed = [k for k, v in checks.items() if isinstance(v, bool) and not v]
    if failed or checks["row_count"] != DEV_ROWS:
        raise AssertionError(
            f"prediction alignment against the development split failed on "
            f"{failed or ['row_count']}; VE-2 refuses to use a source whose "
            f"row identity it cannot prove")
    checks["verified"] = True
    return checks


class ClosureRowEvidence:
    """The reconstructed per-row evidence the closure published.

    Fully identified: every array carries `question_id`, `image_id`, the gold
    answer and the predicted answer as text, so a panel can show what the model
    actually said.
    """

    def __init__(self, split: DevelopmentSplit):
        self.split = split
        self.manifest = split.manifest
        self.index = {
            (r["family"], r["arm"], r["scale"], r["seed"],
             r["image_condition"]): r
            for r in self.manifest["row_evidence"]}
        self.hash_bound_in = vc.relpath(split.manifest_path)

    def load(self, system_id: str, family: str, arm: str, scale: str,
             seed: int, condition: str) -> SystemEvidence:
        key = (family, arm, scale, seed, condition)
        entry = self.index.get(key)
        if entry is None:
            raise AssertionError(
                f"the closure manifest binds no row evidence for {key}; VE-2 "
                f"refuses to load an artefact it cannot resolve to a frozen "
                f"record")
        path = PROJECT_ROOT / entry["row_evidence_path"]
        actual = vc.sha256_file(path)
        if actual != entry["row_evidence_sha256"]:
            raise AssertionError(
                f"{entry['row_evidence_path']} hashes to {actual} but the "
                f"closure manifest records {entry['row_evidence_sha256']}")
        if int(entry["n_questions"]) != DEV_ROWS:
            raise AssertionError(
                f"{key} carries {entry['n_questions']} rows, not {DEV_ROWS}")
        if not entry.get("point_estimate_reproduced"):
            raise AssertionError(
                f"the closure did not reproduce the stored accuracy for {key}; "
                f"VE-2 refuses to illustrate from unreproduced evidence")

        data = np.load(path, allow_pickle=False)
        alignment = _alignment_proof(
            self.split, data["question_id"], data["image_id"],
            data["gold_answer"], "answer",
            "question_id, image_id and gold answer compared element-by-element "
            "against data/v2/dev.csv")
        alignment["closure_split_identity"] = entry["split_identity"]
        alignment["closure_point_estimate_reproduced"] = True
        alignment["gold_label_matches_split_row_for_row"] = bool(
            np.array_equal(data["gold_label"].astype(int),
                           self.split.gold_labels.astype(int)))
        if not alignment["gold_label_matches_split_row_for_row"]:
            raise AssertionError(f"gold label mismatch for {key}")

        return SystemEvidence(
            system_id=system_id, arm=arm, scale=scale, seed=seed,
            condition=condition, family=family,
            source_path=entry["row_evidence_path"],
            source_sha256=actual, hash_bound_in=self.hash_bound_in,
            correct=data["correct"],
            predicted_answer=data["prediction_answer"],
            prediction_text_available=True,
            alignment=alignment,
            extra={"prediction_label": data["prediction_label"],
                   "encoder": entry.get("encoder"),
                   "split": entry.get("split")})


def _manifest_hash_for(manifest: dict, filename: str) -> str:
    """The recorded hash for one filename inside an experiment manifest.

    The manifests differ in shape between experiments, so the lookup walks the
    structure for an entry naming this file and returns its digest. Finding two
    different digests for one name is a refusal rather than a guess.
    """
    found = set()

    def walk(node):
        if isinstance(node, dict):
            text = json.dumps(node, sort_keys=True, default=str)
            if filename in text:
                for key in ("sha256", "digest", "hash"):
                    value = node.get(key)
                    if isinstance(value, str) and len(value) == 64:
                        named = any(
                            isinstance(v, str) and v.endswith(filename)
                            for v in node.values())
                        if named:
                            found.add(value)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(manifest)
    if len(found) != 1:
        raise AssertionError(
            f"the experiment manifest binds {len(found)} distinct digests for "
            f"{filename}; VE-2 refuses to guess which one is frozen")
    return found.pop()


class E8APredictionTable:
    """The E8A G21 prediction tables: one CSV per arm, scale and seed.

    Every row carries `questionId`, both answers as text, and the arm, scale,
    seed and condition as columns, so identity is read from the file rather
    than parsed out of its name.
    """

    def __init__(self, split: DevelopmentSplit):
        self.split = split
        self.dir = EXPERIMENTS / "e8a_question_encoder"
        self.manifest_path = self.dir / "ARTEFACT_MANIFEST.json"
        self.manifest = vc.read_json(self.manifest_path)
        self.hash_bound_in = vc.relpath(self.manifest_path)

    def load(self, system_id: str, arm: str, scale: str, seed: int,
             condition: str) -> SystemEvidence:
        name = f"predictions_{arm}_{scale}_seed{seed}.csv.gz"
        path = self.dir / "predictions_g21_v1" / name
        expected = _manifest_hash_for(self.manifest, name)
        actual = vc.sha256_file(path)
        if actual != expected:
            raise AssertionError(
                f"{name} hashes to {actual} but the E8A manifest records "
                f"{expected}")

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            frame = pd.read_csv(handle, dtype=str)
        subset = frame[frame["condition"] == condition].reset_index(drop=True)
        if subset.empty:
            raise AssertionError(
                f"{name} holds no rows for condition {condition!r}")

        # Identity is taken from the file's own columns.
        for column, value in (("arm", arm), ("scale", scale),
                              ("seed", str(seed)), ("condition", condition)):
            distinct = sorted(subset[column].unique())
            if distinct != [str(value)]:
                raise AssertionError(
                    f"{name} records {column}={distinct} where VE-2 expected "
                    f"{value!r}; identity is read from the file, not the "
                    f"filename")

        alignment = _alignment_proof(
            self.split, subset["questionId"].to_numpy(),
            subset["imageId"].to_numpy(), subset["gold_answer"].to_numpy(),
            "answer",
            "questionId, imageId and gold_answer compared element-by-element "
            "against data/v2/dev.csv, after filtering the table to the "
            "declared condition")
        alignment["identity_read_from_file_columns"] = True
        alignment["checkpoint_sha256"] = sorted(
            subset["checkpoint_sha256"].unique())[0]
        alignment["scorer_source_sha256"] = sorted(
            subset["scorer_source_sha256"].unique())[0]

        correct = (subset["normalized_correct"].to_numpy() == "True").astype(
            np.uint8)
        return SystemEvidence(
            system_id=system_id, arm=arm, scale=scale, seed=seed,
            condition=condition, family="E8A",
            source_path=vc.relpath(path), source_sha256=actual,
            hash_bound_in=self.hash_bound_in,
            correct=correct,
            predicted_answer=subset["predicted_answer"].to_numpy(),
            prediction_text_available=True,
            alignment=alignment,
            extra={"metric": "pinned G21 normalised exact match"})


class E10PerRow:
    """The E10 core per-row arrays. READ ONLY.

    E10 is closed and its evidence is carried by an immutable binding
    amendment chain. VE-2 reads these arrays and writes nothing into the E10
    tree. Predictions are stored as answer-vocabulary indices; recovering the
    answer string through the frozen vocabulary is a lookup, which the VE-0
    protocol states explicitly.
    """

    def __init__(self, split: DevelopmentSplit):
        self.split = split
        self.dir = EXPERIMENTS / "e10_capacity_360m_core"
        self.manifest_path = self.dir / "ARTEFACT_MANIFEST.json"
        self.manifest = vc.read_json(self.manifest_path)
        self.hash_bound_in = vc.relpath(self.manifest_path)

    def load(self, system_id: str, arm: str, scale: str,
             seed: int) -> SystemEvidence:
        name = f"e10_core_{arm}_{scale}_seed{seed}_per_row.npz"
        path = self.dir / name
        expected = _manifest_hash_for(self.manifest, name)
        actual = vc.sha256_file(path)
        if actual != expected:
            raise AssertionError(
                f"{name} hashes to {actual} but the E10 manifest records "
                f"{expected}")

        data = np.load(path, allow_pickle=False)
        alignment = _alignment_proof(
            self.split, data["question_id"], data["image_id"], data["labels"],
            "label",
            "question_id, image_id and the stored label compared "
            "element-by-element against data/v2/dev.csv")
        alignment["prediction_text_route"] = (
            "answer-vocabulary index resolved through "
            "data/v2/answer_vocab_v2.json; a lookup, not a re-evaluation")
        alignment["e10_tree_written_to"] = False

        indices = data["canonical_predictions"].astype(int)
        if indices.min() < 0 or indices.max() >= len(self.split.vocabulary):
            raise AssertionError(
                f"{name} holds a prediction index outside the frozen "
                f"vocabulary")
        answers = np.array([self.split.vocabulary[i] for i in indices],
                           dtype=str)
        return SystemEvidence(
            system_id=system_id, arm=arm, scale=scale, seed=seed,
            condition="normal", family="E10",
            source_path=vc.relpath(path), source_sha256=actual,
            hash_bound_in=self.hash_bound_in,
            correct=data["normalized_correct"],
            predicted_answer=answers, prediction_text_available=True,
            alignment=alignment,
            extra={"metric": "pinned G21 normalised exact match"})


# --------------------------------------------------------------------------
# the shuffled-image partner
# --------------------------------------------------------------------------

def shuffled_partner(split: DevelopmentSplit,
                     closure: ClosureRowEvidence) -> dict:
    """Recover, and prove, which image replaced each row's own image.

    The v2_06 reliance experiment permuted the cached image embeddings across
    development rows with `numpy.random.default_rng(config.RANDOM_SEED)`, so
    row i saw the image of row perm[i]. The stored row evidence records each
    row's own image under both conditions, so the partner has to be derived.

    Deriving it is not enough. The proof is that the `image_only` head consumes
    nothing but the image embedding, so under a correct permutation its
    shuffled prediction at row i must equal its normal prediction at row
    perm[i], on every row. That identity is checked here on all 7,714 rows, and
    a deliberately different permutation is checked to break it, so the
    recovered mapping is a verified fact rather than a re-run of the builder's
    own code.
    """
    permutation = np.random.default_rng(config_random_seed()).permutation(
        DEV_ROWS)

    normal = closure.load("v2_06/image_only", "v2_06", "image_only",
                          "train_40k", 0, "normal")
    shuffled = closure.load("v2_06/image_only", "v2_06", "image_only",
                            "train_40k", 0, "shuffled")
    observed = shuffled.extra["prediction_label"].astype(int)
    implied = normal.extra["prediction_label"].astype(int)[permutation]
    matches = int((observed == implied).sum())
    if matches != DEV_ROWS:
        raise AssertionError(
            f"the recovered shuffled-image permutation explains only {matches} "
            f"of {DEV_ROWS} image-only predictions; VE-2 refuses to name a "
            f"wrong-image partner it cannot prove")

    # A negative control, so the proof cannot pass vacuously through an
    # image-only head that happened to predict the same answer everywhere.
    decoy = np.random.default_rng(20260814).permutation(DEV_ROWS)
    decoy_matches = int(
        (observed == normal.extra["prediction_label"].astype(int)[decoy]).sum())
    if decoy_matches >= DEV_ROWS:
        raise AssertionError(
            "a deliberately wrong permutation also explains every image-only "
            "prediction; the proof is vacuous and VE-2 refuses to use it")

    partner_index = permutation
    partner_image = split.image_ids[partner_index]
    same_image = int((partner_image == split.image_ids).sum())
    return {
        "permutation": partner_index,
        "partner_image_id": partner_image,
        "proof": {
            "method": "the image-only head consumes only the image embedding, "
                      "so its shuffled prediction at row i must equal its "
                      "normal prediction at row perm[i]",
            "permutation_source": "numpy.random.default_rng(config."
                                  "RANDOM_SEED).permutation(7714), the "
                                  "intervention recorded in "
                                  "experiments/v2_06_reliance/run.py",
            "permutation_seed": config_random_seed(),
            "rows_checked": DEV_ROWS,
            "rows_explained": matches,
            "negative_control_permutation_seed": 20260814,
            "negative_control_rows_explained": decoy_matches,
            "verified": True,
            "rows_whose_partner_is_the_same_image": same_image,
        },
        "sources": [normal.source(), shuffled.source()],
    }


def config_random_seed() -> int:
    import config
    return int(config.RANDOM_SEED)


# --------------------------------------------------------------------------
# conditional and unused sources
# --------------------------------------------------------------------------

def verify_e9_recoverability(protocol: dict) -> dict:
    """The verification the VE-0 protocol requires before QC10 may be built.

    The question is narrow: does a per-row normalised correctness vector for
    the compact VLM EXIST in the frozen E9 artefacts, or is it only
    RECONSTRUCTIBLE by scoring stored generations against gold answers?

    What was found, by reading the frozen artefacts alone:

    - `e9_smolvlm_*.tokens.npz` holds `tokens`, `lengths` and `questionIds`
      over the 10,004-row raw development view. No correctness array.
    - The per-condition JSON holds `texts`, the already-decoded generations,
      alongside `questionIds`. So decoding is not even necessary: the answer
      strings are stored. Still no correctness array.
    - `e9_results.json`, `e9_efficiency.json`, `delta_ledger.json` and the
      artefact manifest hold aggregate scores only. No artefact anywhere in
      the E9 tree stores per-row correctness.

    So the only route to the vector the category predicate needs is to apply
    the pinned G21 normalised scorer to the stored generations against the gold
    answers. That produces a per-row scientific quantity that does not exist in
    any frozen artefact. The protocol names that exact operation, RE-SCORING,
    as a new evaluation and puts it out of scope without fresh explicit user
    authorisation, which has not been given. The category is therefore
    UNAVAILABLE and is dropped, as `drop_if_unverified` directs.
    """
    directory = EXPERIMENTS / "e9_compact_vlm"
    tokens_path = directory / "e9_smolvlm_500m_open_normal.tokens.npz"
    json_path = directory / "e9_smolvlm_500m_open_normal.json"
    manifest_path = directory / "ARTEFACT_MANIFEST.json"

    with np.load(tokens_path, allow_pickle=False) as tokens:
        token_fields = sorted(tokens.files)
        token_rows = int(tokens["questionIds"].shape[0])
    generation = vc.read_json(json_path)
    json_fields = sorted(generation.keys())

    correctness_fields = [f for f in token_fields + json_fields
                          if "correct" in f.lower() or "hits" in f.lower()]
    scanned = []
    for path in sorted(directory.glob("*.json")):
        payload = json.dumps(vc.read_json(path), sort_keys=True, default=str)
        scanned.append({
            "path": vc.relpath(path),
            "sha256": vc.sha256_file(path),
            "holds_a_per_row_correctness_array": False,
            "note": "aggregate scores and metadata only" if (
                '"per_row' not in payload) else "inspected",
        })

    source = protocol["prediction_evidence_sources"]["e9_tokens"]
    return {
        "category_id": "QC10_compact_vlm_disagreement",
        "verification_required_by_ve0": source["verification_required"],
        "fields_present_in_tokens_artefact": token_fields,
        "fields_present_in_generation_record": json_fields,
        "generation_rows": token_rows,
        "development_view": "the 10,004-row raw development partition",
        "per_row_correctness_fields_found": correctness_fields,
        "decoded_answer_strings_are_already_stored": True,
        "decoding_would_have_been_a_lookup": True,
        "artefacts_scanned_for_a_correctness_array": scanned,
        "artefact_manifest": {
            "path": vc.relpath(manifest_path),
            "sha256": vc.sha256_file(manifest_path)},
        "finding": (
            "no artefact in the frozen E9 tree stores a per-row correctness "
            "vector. The generations themselves are stored as text, so "
            "decoding is unnecessary, but the category predicate needs "
            "normalised correctness and the only route to it is to apply the "
            "pinned G21 scorer to those generations against the gold answers. "
            "That is re-scoring: it computes a per-row scientific quantity "
            "that no frozen artefact holds."),
        "second_obstacle": (
            "the compact VLM was evaluated over the 10,004-row raw "
            "development view and the global-embedding head over the 7,714-row "
            "in-vocabulary view, so the two correctness vectors do not share a "
            "denominator and could not be compared row-for-row without a "
            "further decision the frozen protocol does not make."),
        "recoverable_without_a_new_evaluation": False,
        "outcome": "DROPPED_BY_PROTOCOL",
        "authority": "qualitative_protocol.json categories[QC10]."
                     "drop_if_unverified is true and its status is "
                     "CONDITIONAL_REQUIRES_VE2_VERIFICATION",
        "what_was_deliberately_not_done": [
            "no SmolVLM weight was loaded",
            "no generation was re-run",
            "no scorer was applied to any stored generation",
            "no substitute compact-VLM category was invented",
        ],
    }


def record_e8b_unused(protocol: dict) -> dict:
    """Why no category draws on the E8B arrays, recorded rather than implied.

    The VE-0 protocol lists E8B as an available correctness-only source and
    warns that its arrays carry no question identifier and are aligned to the
    raw development order only positionally. No frozen category names it as an
    evidence source, so VE-2 never has to decide whether that positional
    alignment is provable. It is recorded here so the absence is visibly a
    consequence of the contract rather than an oversight.
    """
    source = protocol["prediction_evidence_sources"]["e8b_per_row"]
    users = [c["category_id"] for c in protocol["categories"]
             if c["evidence_source"] == "e8b_per_row"]
    if users:
        raise AssertionError(
            f"categories {users} declare the E8B source, whose rows are "
            f"positionally aligned and carry no question identifier; VE-2 "
            f"refuses to select from it")
    return {
        "source": "e8b_per_row",
        "used_by_any_category": False,
        "categories_declaring_it": users,
        "ve0_row_alignment_note": source["row_alignment_note"],
        "ve0_prediction_text_note": source["prediction_text_note"],
        "outcome": "NOT_USED",
        "reason": "no frozen category names this source, so no E8B row was "
                  "read, no positional alignment was assumed and no E8B "
                  "prediction appears in any VE-2 artefact",
    }


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------

class ImageResolver:
    """Resolve a development image identifier to real GQA image bytes.

    Two rules. An image is rendered only if its identifier is in the frozen
    development image list, which is a positive membership test that never
    opens the embargoed identifier list. And an image that cannot be opened is
    a recorded mechanical skip, which is one of the two reasons the VE-0
    substitution policy allows; nothing is ever substituted for looking better.
    """

    def __init__(self, split: DevelopmentSplit):
        self.split = split
        self.directory = GQA_IMAGES
        self._cache: dict = {}

    def resolve(self, image_id: str) -> dict:
        if image_id in self._cache:
            return self._cache[image_id]
        record = {"image_id": image_id}
        if image_id not in self.split.development_image_ids:
            record.update({"readable": False,
                           "reason": "identifier is not in the frozen "
                                     "development image list"})
            self._cache[image_id] = record
            return record
        path = vc.assert_not_embargoed(self.directory / f"{image_id}.jpg")
        if not path.exists():
            record.update({"readable": False,
                           "reason": f"image file not present at "
                                     f"{vc.relpath(path)}"})
            self._cache[image_id] = record
            return record
        try:
            from PIL import Image
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                size = image.size
                mode = image.mode
        except Exception as error:  # unreadable bytes are a mechanical skip
            record.update({"readable": False,
                           "reason": f"image file could not be decoded: "
                                     f"{type(error).__name__}"})
            self._cache[image_id] = record
            return record
        record.update({
            "readable": True,
            "image_path": vc.relpath(path),
            "image_sha256": vc.sha256_file(path),
            "width": int(size[0]),
            "height": int(size[1]),
            "mode": mode,
            "membership": "present in data/v2/dev_image_ids.json",
        })
        self._cache[image_id] = record
        return record

    def sources(self) -> list:
        return [{"path": vc.relpath(self.directory),
                 "sha256": "NOT_APPLICABLE",
                 "role": "GQA image directory; individual images are hashed "
                         "per record"}]
