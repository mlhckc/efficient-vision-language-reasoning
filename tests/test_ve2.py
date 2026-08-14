"""VE-2 validation: the qualitative evidence layer.

What these tests are for. VE-2 shows individual development questions beside
real photographs and states what each system answered. The risk is not an
arithmetic error; it is that a panel shows the wrong image, the wrong gold
answer, or one model's prediction under another model's name, and that nothing
catches it because the picture looks fine. So the suite works backwards from
every rendered string to the frozen artefact it came from.

The groups below, in the order they run:

  contract    the frozen VE-0 protocol is unchanged, the salt is the recorded
              one, and the implemented rank rule is the recorded formula
  selection   the ordering reproduces exactly, is independent of everything
              except salt, category and question identifier, and every skip
              carries one of the two mechanical reasons the policy allows
  identity    question, image, gold answer, type metadata, program length,
              prediction string and correctness each match their source
  images      every rendered image is a real development image, resolved from
              its identifier, hashed, and never a substitute
  scope       development only, no embargoed reference anywhere in the VE-2
              tree, no new evaluation output, no unverified source used
  captions    every caption carries its illustrative-status and limitation
              sentence and no overclaim survives the guard
  provenance  the chain from artefact to source hash resolves, and a rebuild
              reproduces the content

This module is deliberately NOT imported by tests/run_all.py. That file is
inside the E10 SOURCE_PATHS digest, and adding an import to it would move the
digest and make the three E10 phase verifiers refuse their own sealed output.
The closure shipped tests/run_closure.py for this reason, VE-0 shipped
tests/run_ve0.py and VE-1 shipped tests/run_ve1.py; VE-2 follows the same
pattern with tests/run_ve2.py. The embargo source scan in run_all.py globs
tests/*.py, so this file is still covered by it without any edit to the sealed
file.
"""

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.ve2 import evidence, records, selection  # noqa: E402
from experiments.ve2 import ve2_common as vc  # noqa: E402
from experiments.ve2.ve2_common import Contract  # noqa: E402

VE2 = PROJECT_ROOT / "results" / "ve2"
DEV = PROJECT_ROOT / "data" / "v2" / "dev.csv"
TYPES = PROJECT_ROOT / "data" / "v2" / "metadata" / "dev_types.csv"

# The embargoed stem, assembled from two halves so this source can be scanned
# for the literal without matching itself.
EMBARGOED = "test_" + "clean"

# The salt recorded in the frozen VE-0 protocol. Written out here so a change
# to the protocol is a test failure rather than a silently different gallery.
EXPECTED_SALT = "P2607-VE0-QUALITATIVE-SALT-20260813-v1"
EXPECTED_PROTOCOL_DIGEST = (
    "c6ca3aece6f877afefd6d46afd99ab05b60e8d1a5afa172568666dae7150b76d")

DEV_ROWS = 7714


def _load(name: str) -> dict:
    return json.loads((VE2 / name).read_text(encoding="utf-8"))


def _records() -> list:
    return _load("qualitative_records.json")["records"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------

def test_frozen_protocol_is_unchanged() -> None:
    """The qualitative contract VE-2 selected under is the accepted one."""
    contract = Contract()
    _check(contract.protocol["content_sha256"] == EXPECTED_PROTOCOL_DIGEST,
           "the VE-0 qualitative protocol's content digest has moved")
    _check(vc.content_digest(contract.protocol) == EXPECTED_PROTOCOL_DIGEST,
           "the VE-0 qualitative protocol does not hash to its own recorded "
           "content digest")


def test_salt_is_the_frozen_one() -> None:
    """The salt is read from the contract, is the recorded value, and is
    certified as frozen before any example was inspected."""
    contract = Contract()
    _check(contract.salt == EXPECTED_SALT,
           f"the selection salt is {contract.salt!r}, not the frozen "
           f"{EXPECTED_SALT!r}")
    _check(contract.selection_rule["salt_frozen_before_inspection"] is True,
           "the protocol no longer certifies the salt as frozen before "
           "inspection")
    registry = _load("selection_registry.json")
    _check(registry["salt"] == EXPECTED_SALT,
           "the published selection registry records a different salt")
    _check(EXPECTED_SALT not in
           (PROJECT_ROOT / "experiments" / "ve2" / "ve2_common.py").read_text(),
           "VE-2 source holds its own copy of the salt; it must read the "
           "frozen contract instead")


def test_rank_formula_matches_the_contract() -> None:
    contract = Contract()
    vc.assert_rank_formula_matches(contract)
    key = vc.deterministic_rank_key(EXPECTED_SALT, "QC02_all_wrong", "12345")
    expected = hashlib.sha256(
        f"{EXPECTED_SALT}|QC02_all_wrong|12345".encode("utf-8")).hexdigest()
    _check(key == expected, "the rank key is not SHA-256 over "
                            "salt|category|question_id")


def test_every_category_predicate_matches_its_contract() -> None:
    contract = Contract()
    for category_id in contract.protocol["available_categories"]:
        selection.assert_predicate_matches(contract.category(category_id))
        records.assert_category_systems_match(contract.category(category_id))


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def test_recorded_ranks_recompute() -> None:
    """Every published rank key recomputes from the frozen rule."""
    for record in _records():
        expected = vc.deterministic_rank_key(
            EXPECTED_SALT, record["category_id"], record["question_id"])
        _check(record["selection_rank_key"] == expected,
               f"{record['qualitative_id']} carries a rank key that does not "
               f"recompute")
        _check(record["qualitative_id"] ==
               f"VE2-QUAL-{record['category_id']}-"
               f"{record['selection_rank']:02d}",
               f"{record['qualitative_id']} does not follow the schema's "
               f"identifier format")


def test_selection_reproduces_from_the_published_pools() -> None:
    """The published selection is what the frozen rule returns.

    For each category the whole development split is re-ordered by the frozen
    rank and the published examples are checked to be the lowest-ranked rows
    of that ordering that satisfy the category's own predicate, with the
    balanced category checked within each direction.
    """
    contract = Contract()
    registry = {c["category_id"]: c
                for c in _load("selection_registry.json")["categories"]}
    split = evidence.DevelopmentSplit()
    by_category: dict = {}
    for record in _records():
        by_category.setdefault(record["category_id"], []).append(record)

    for category_id, published in by_category.items():
        entry = registry[category_id]
        pool_ids = _recompute_pool(category_id, split)
        _check(len(pool_ids) == entry["candidate_pool_size"],
               f"{category_id} pool is {len(pool_ids)} rows, the registry "
               f"records {entry['candidate_pool_size']}")
        ordered = sorted(
            pool_ids,
            key=lambda q: (vc.deterministic_rank_key(EXPECTED_SALT,
                                                     category_id, q), q))
        rank_of = {q: i + 1 for i, q in enumerate(ordered)}
        for record in published:
            _check(record["question_id"] in rank_of,
                   f"{record['qualitative_id']} is not in its category's "
                   f"recomputed candidate pool")
            _check(rank_of[record["question_id"]] == record["selection_rank"],
                   f"{record['qualitative_id']} publishes rank "
                   f"{record['selection_rank']} but recomputes to "
                   f"{rank_of[record['question_id']]}")

        chosen_ranks = sorted(r["selection_rank"] for r in published)
        if entry["balance_requirement"] == "NOT_APPLICABLE":
            _check(chosen_ranks == list(range(1, len(published) + 1)),
                   f"{category_id} did not take the lowest ranks in order and "
                   f"recorded no skip: ranks {chosen_ranks}")
        else:
            _check(len(published) == 2,
                   f"{category_id} declares a balance requirement but "
                   f"published {len(published)} examples")
            directions = {r["systems"][0]["correct"] for r in published}
            _check(directions == {True, False},
                   f"{category_id} published both examples in the same "
                   f"direction, which its balance requirement forbids")
            for record in published:
                lower = [q for q in ordered
                         if rank_of[q] < record["selection_rank"]]
                same = _direction_members(split, record["systems"][0]["correct"])
                _check(not [q for q in lower if q in same],
                       f"{record['qualitative_id']} is not the lowest-ranked "
                       f"row in its direction")


def _recompute_pool(category_id: str, split) -> set:
    """Re-evaluate one category's predicate independently of the builder."""
    frame = split.frame
    qid = frame["questionId"].to_numpy()

    def closure(family, arm, condition="normal"):
        path = (PROJECT_ROOT / "results" / "closure" / "row_evidence" / family /
                f"{family}__{arm}__train_40k__seed0__{condition}.npz")
        return np.load(path, allow_pickle=False)

    if category_id in ("QC01_all_correct", "QC02_all_wrong"):
        want = 1 if category_id == "QC01_all_correct" else 0
        mask = np.ones(DEV_ROWS, dtype=bool)
        for arm in ("question_only", "concat", "fusion"):
            mask &= closure("v2_02", arm)["correct"] == want
    elif category_id == "QC03_fusion_right_concat_wrong":
        mask = (closure("v2_02", "fusion")["correct"] == 1) \
            & (closure("v2_02", "concat")["correct"] == 0)
    elif category_id == "QC04_representation_improvement":
        mask = (closure("e2", "fusion")["correct"] == 1) \
            & (closure("v2_02", "fusion")["correct"] == 0)
    elif category_id == "QC05_question_side_pretraining":
        mask = _e8a("A1") & ~_e8a("A1r")
    elif category_id == "QC06_answer_side_disagreement":
        mask = _e10("B4") != _e10("B4r")
    elif category_id == "QC07_visual_reliance":
        mask = (closure("v2_06", "fusion", "normal")["correct"] == 1) \
            & (closure("v2_06", "fusion", "shuffled")["correct"] == 0)
    elif category_id == "QC08_short_reasoning":
        mask = frame["n_steps"].to_numpy() <= 2
    elif category_id == "QC09_deep_reasoning_failure":
        mask = (frame["n_steps"].to_numpy() >= 4) \
            & (closure("v2_02", "fusion")["correct"] == 0)
    else:
        raise AssertionError(f"no independent predicate for {category_id}")
    return {str(q) for q in qid[np.asarray(mask, dtype=bool)]}


def _e8a(arm: str) -> np.ndarray:
    path = (PROJECT_ROOT / "results" / "experiments" / "e8a_question_encoder" /
            "predictions_g21_v1" / f"predictions_{arm}_train_40k_seed0.csv.gz")
    frame = pd.read_csv(path, dtype=str)
    frame = frame[frame["condition"] == "normal"].reset_index(drop=True)
    return (frame["normalized_correct"].to_numpy() == "True")


def _e10(arm: str) -> np.ndarray:
    path = (PROJECT_ROOT / "results" / "experiments" /
            "e10_capacity_360m_core" /
            f"e10_core_{arm}_train_40k_seed0_per_row.npz")
    return np.load(path, allow_pickle=False)["normalized_correct"].astype(int)


def _direction_members(split, b4_correct: bool) -> set:
    b4, b4r = _e10("B4"), _e10("B4r")
    mask = (b4 == 1) if b4_correct else (b4r == 1)
    mask = mask & (b4 != b4r)
    return {str(q) for q in split.question_ids[mask]}


def test_rank_depends_on_nothing_but_salt_category_and_question() -> None:
    """A different image, model or outcome cannot move a rank.

    The rank key is a pure function of three strings. This checks the
    consequence directly: the ordering of a fixed pool is unchanged when the
    question identifiers keep their values but arrive in a different order,
    and changes when the category changes.
    """
    ids = [f"{i:09d}" for i in range(200)]
    forward = selection.order_pool(EXPECTED_SALT, "QC02_all_wrong",
                                   range(len(ids)), ids)
    backward = selection.order_pool(EXPECTED_SALT, "QC02_all_wrong",
                                    reversed(range(len(ids))), ids)
    _check([c.question_id for c in forward] ==
           [c.question_id for c in backward],
           "the ordering depends on the order candidates were supplied in")
    other = selection.order_pool(EXPECTED_SALT, "QC01_all_correct",
                                 range(len(ids)), ids)
    _check([c.question_id for c in forward] != [c.question_id for c in other],
           "two different categories produce the same ordering, so the "
           "category is not entering the rank key")


def test_a_skip_needs_a_mechanical_reason() -> None:
    """A skip for any other reason is refused, not recorded."""
    ids = ["000000001", "000000002"]
    candidates = selection.order_pool(EXPECTED_SALT, "QC02_all_wrong",
                                      range(2), ids)
    allowed, _ = selection.select(
        candidates, 1,
        lambda c: (selection.SKIP_IMAGE_UNREADABLE, "unreadable")
        if c.rank == 1 else None)
    _check(allowed[0][0].rank == 2, "an allowed skip did not advance the rank")
    _check(allowed[0][1][0]["reason"] == selection.SKIP_IMAGE_UNREADABLE,
           "the skip was not recorded with its reason")

    try:
        selection.select(candidates, 1,
                         lambda c: ("this one reads better", "prettier image"))
    except AssertionError as error:
        _check("refuses to skip" in str(error),
               "an aesthetic skip was refused for the wrong reason")
    else:
        raise AssertionError(
            "VE-2 accepted an aesthetic skip reason; the substitution policy "
            "allows only a mechanical one")


def test_published_skips_are_all_mechanical() -> None:
    registry = _load("selection_registry.json")
    for category in registry["categories"]:
        for skip in category["skips"]:
            _check(skip["reason"] in selection.ALLOWED_SKIP_REASONS,
                   f"{category['category_id']} records a skip for "
                   f"{skip['reason']!r}, which the frozen policy does not "
                   f"allow")
            _check(skip.get("detail"),
                   f"{category['category_id']} records a skip with no detail")
    for record in _records():
        for skip in record["skipped_ranks"]:
            _check(skip["reason"] in selection.ALLOWED_SKIP_REASONS,
                   f"{record['qualitative_id']} records a non-mechanical skip")


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------

def test_every_record_matches_the_development_split() -> None:
    """Question, image, gold answer, types and program length are the split's."""
    dev = pd.read_csv(DEV, dtype={"questionId": str, "imageId": str})
    dev = dev.set_index("questionId")
    types = pd.read_csv(TYPES, dtype={"questionId": str}).set_index("questionId")
    for record in _records():
        question = record["question_id"]
        _check(question in dev.index,
               f"{record['qualitative_id']} names a question that is not in "
               f"the development split")
        row, kind = dev.loc[question], types.loc[question]
        for field, actual, expected in (
                ("image_id", record["image_id"], row["imageId"]),
                ("question_text", record["question_text"], row["question"]),
                ("gold_answer", record["gold_answer"], row["answer"]),
                ("semantic_type", record["semantic_type"], kind["semantic"]),
                ("structural_type", record["structural_type"],
                 kind["structural"]),
                ("n_program_steps", record["n_program_steps"],
                 int(kind["n_steps"]))):
            _check(actual == expected,
                   f"{record['qualitative_id']} {field} is {actual!r} but the "
                   f"development split records {expected!r}")


def test_every_prediction_matches_its_frozen_source() -> None:
    """Each published prediction and correctness is read back from its file."""
    dev = pd.read_csv(DEV, dtype={"questionId": str})
    position = {q: i for i, q in enumerate(dev["questionId"])}
    vocabulary = json.loads(
        (PROJECT_ROOT / "data" / "v2" / "answer_vocab_v2.json").read_text())
    vocabulary = (vocabulary["answers"] if isinstance(vocabulary, dict)
                  else vocabulary)
    cache: dict = {}
    checked = 0
    for record in _records():
        index = position[record["question_id"]]
        for system in record["systems"]:
            path = PROJECT_ROOT / system["evidence_path"]
            _check(_sha256(path) == system["evidence_sha256"],
                   f"{system['evidence_path']} does not hash to the value the "
                   f"record carries")
            if path.suffix == ".npz":
                if str(path) not in cache:
                    cache[str(path)] = np.load(path, allow_pickle=False)
                data = cache[str(path)]
                _check(str(data["question_id"][index]) == record["question_id"],
                       f"{system['system_id']} row {index} is a different "
                       f"question")
                if "prediction_answer" in data.files:
                    answer = str(data["prediction_answer"][index])
                    correct = bool(data["correct"][index] == 1)
                else:
                    answer = vocabulary[
                        int(data["canonical_predictions"][index])]
                    correct = bool(data["normalized_correct"][index] == 1)
            else:
                if str(path) not in cache:
                    frame = pd.read_csv(path, dtype=str)
                    cache[str(path)] = frame[
                        frame["condition"] == system["condition"]
                    ].reset_index(drop=True)
                frame = cache[str(path)]
                _check(frame["questionId"].iloc[index] == record["question_id"],
                       f"{system['system_id']} row {index} is a different "
                       f"question")
                answer = str(frame["predicted_answer"].iloc[index])
                correct = frame["normalized_correct"].iloc[index] == "True"
            _check(answer == system["predicted_answer"],
                   f"{record['qualitative_id']} {system['system_id']} "
                   f"publishes {system['predicted_answer']!r} but the source "
                   f"holds {answer!r}")
            _check(correct == system["correct"],
                   f"{record['qualitative_id']} {system['system_id']} "
                   f"publishes correct={system['correct']} but the source "
                   f"holds {correct}")
            checked += 1
    _check(checked >= 18, f"only {checked} predictions were checked")


def test_predictions_are_row_aligned_not_positionally_assumed() -> None:
    """Every source used carries a verified element-by-element alignment."""
    for record in _records():
        for system in record["systems"]:
            proof = system["alignment_proof"]
            _check(proof.get("verified") is True,
                   f"{system['system_id']} carries no verified alignment")
            _check(proof["question_id_matches_split_row_for_row"],
                   f"{system['system_id']} question identifiers do not match "
                   f"the split row for row")
            _check(proof["image_id_matches_split_row_for_row"],
                   f"{system['system_id']} image identifiers do not match the "
                   f"split row for row")
            _check(proof["row_count"] == DEV_ROWS,
                   f"{system['system_id']} does not cover the development "
                   f"split")


def test_no_e8b_row_is_used() -> None:
    """The one source whose alignment is positional is not used anywhere."""
    registry = _load("selection_registry.json")
    unused = {entry["source"] for entry in
              registry["unused_prediction_sources"]}
    _check("e8b_per_row" in unused,
           "the E8B source is not recorded as unused")
    text = json.dumps(_records())
    _check("e8b" not in text.lower(),
           "an E8B artefact appears in a qualitative record, but its rows "
           "carry no question identifier and are only positionally aligned")


def test_compact_vlm_category_is_dropped_not_approximated() -> None:
    """QC10 is recorded unavailable, with the reason, and nothing invented."""
    registry = _load("selection_registry.json")
    dropped = {d["category_id"]: d for d in registry["dropped_categories"]}
    _check("QC10_compact_vlm_disagreement" in dropped,
           "the conditional compact-VLM category is neither populated nor "
           "recorded as dropped")
    finding = dropped["QC10_compact_vlm_disagreement"]
    _check(finding["recoverable_without_a_new_evaluation"] is False,
           "the category was dropped without establishing that its evidence "
           "is unrecoverable")
    _check(finding["outcome"] == "DROPPED_BY_PROTOCOL",
           f"unexpected outcome {finding['outcome']}")
    ids = {r["category_id"] for r in _records()}
    _check("QC10_compact_vlm_disagreement" not in ids,
           "a dropped category produced records anyway")
    text = json.dumps(_records()) + json.dumps(_load("captions.json"))
    _check("smolvlm" not in text.lower(),
           "a compact-VLM prediction appears in a VE-2 artefact")


def test_the_wrong_image_partner_is_proved_not_assumed() -> None:
    """QC07's substituted image is recovered with a check on all 7,714 rows."""
    registry = _load("selection_registry.json")
    proof = registry["shuffled_partner_recovery"]
    _check(proof["verified"] is True, "the partner recovery is not verified")
    _check(proof["rows_explained"] == proof["rows_checked"] == DEV_ROWS,
           f"the recovered permutation explains {proof['rows_explained']} of "
           f"{proof['rows_checked']} rows")
    _check(proof["negative_control_rows_explained"] < DEV_ROWS,
           "a deliberately wrong permutation explains every row, so the proof "
           "is vacuous")
    for record in _records():
        if not record["category_id"].startswith("QC07"):
            continue
        partners = [s["displayed_image"] for s in record["systems"]
                    if not s["displayed_image"]["is_the_questions_own_image"]]
        _check(len(partners) == 1,
               f"{record['qualitative_id']} does not show exactly one "
               f"wrong-image partner")
        _check(partners[0]["image_id"] != record["image_id"],
               f"{record['qualitative_id']} shows the question's own image as "
               f"the image that replaced it")


def test_records_are_schema_exact_and_distinct() -> None:
    contract = Contract()
    seen = {}
    for record in _records():
        records.assert_schema_exact(record, contract.record_schema)
        _check(record["qualitative_id"] not in seen,
               f"{record['qualitative_id']} is used by two records")
        seen[record["qualitative_id"]] = record["question_id"]
        _check(record["placement"] ==
               contract.category(record["category_id"])["placement"],
               f"{record['qualitative_id']} placement does not match its "
               f"category")


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------

def test_every_rendered_image_is_a_real_development_image() -> None:
    """Resolved from the identifier, hashed, and present in the split."""
    development = set(json.loads(
        (PROJECT_ROOT / "data" / "v2" / "dev_image_ids.json").read_text()))
    checked = 0
    for record in _records():
        shown = [{"image_id": record["image_id"],
                  "image_path": record["image_path"],
                  "image_sha256": record["image_sha256"]}]
        shown += [s["displayed_image"] for s in record["systems"]]
        for image in shown:
            _check(image["image_id"] in development,
                   f"{image['image_id']} is not a development image")
            path = PROJECT_ROOT / image["image_path"]
            _check(path.exists(), f"{image['image_path']} does not exist")
            _check(path.name == f"{image['image_id']}.jpg",
                   f"{image['image_path']} is not the file for image "
                   f"{image['image_id']}: an image was substituted")
            _check(_sha256(path) == image["image_sha256"],
                   f"{image['image_path']} does not hash to the value the "
                   f"record carries")
            checked += 1
    _check(checked >= 18, f"only {checked} images were checked")


def test_images_are_read_at_a_usable_size() -> None:
    from PIL import Image
    for record in _records():
        with Image.open(PROJECT_ROOT / record["image_path"]) as image:
            width, height = image.size
        _check(min(width, height) >= 100,
               f"{record['image_path']} is {width}x{height}, too small to "
               f"render legibly")


# --------------------------------------------------------------------------
# scope
# --------------------------------------------------------------------------

def test_no_embargoed_reference_anywhere_in_the_ve2_tree() -> None:
    """No VE-2 source or artefact names, resolves or hashes the embargoed set."""
    for source in sorted((PROJECT_ROOT / "experiments" / "ve2").glob("*.py")):
        text = source.read_text(encoding="utf-8")
        occurrences = text.count(EMBARGOED)
        allowed = text.count(f'EMBARGOED_NAME = "{EMBARGOED}"')
        _check(occurrences == allowed,
               f"{source.name} references the embargoed material outside the "
               f"single guard constant")
    for artefact in sorted(VE2.rglob("*")):
        if artefact.is_dir() or artefact.suffix not in (".json",):
            continue
        _check(EMBARGOED not in artefact.read_text(encoding="utf-8"),
               f"{artefact.name} names the embargoed material")


def test_the_embargo_guard_actually_refuses() -> None:
    """A negative test, so the firewall is known to fire rather than assumed."""
    try:
        vc.assert_not_embargoed(f"data/v2/{EMBARGOED}_targets.csv")
    except AssertionError:
        pass
    else:
        raise AssertionError("the clean-test path guard did not refuse")
    try:
        vc.assert_payload_not_embargoed({"path": f"{EMBARGOED}_targets.csv"})
    except AssertionError:
        pass
    else:
        raise AssertionError("the clean-test payload guard did not refuse")


def test_scope_confirmations_are_all_negative() -> None:
    manifest = _load("VE2_MANIFEST.json")
    scope = manifest["scope_confirmations"]
    for key, value in scope.items():
        if key == "gpu_hours_charged":
            _check(value == 0.0, "VE-2 charged GPU hours")
            continue
        _check(value is False, f"VE-2 manifest records {key} as {value!r}")
    _check(manifest["provenance"]["evaluated_anything"] is False,
           "VE-2 provenance claims an evaluation")
    _check(manifest["provenance"]["rescored_anything"] is False,
           "VE-2 provenance claims a re-scoring")


def test_ve2_wrote_only_under_its_own_directory() -> None:
    """Nothing was written into a sealed tree."""
    registry = _load("provenance_registry.json")
    outputs = [output["path"]
               for entry in registry["entries"] for output in entry["outputs"]]
    outputs += [value["path"] for value in registry["registry_outputs"].values()]
    for path in outputs:
        _check(path.startswith("results/ve2/"),
               f"VE-2 recorded an output outside results/ve2: {path}")


def test_the_overclaim_guard_refuses() -> None:
    """Negative tests for the interpretation guard."""
    for text in ("the model grounds the object in the image",
                 "this proves the fusion head understands the scene",
                 "an out-of-distribution compositional generalisation result"):
        try:
            vc.assert_no_prohibited_interpretation(text, "test")
        except AssertionError:
            continue
        raise AssertionError(f"the overclaim guard accepted {text!r}")
    # The sanctioned disclaimers must still pass.
    for text in ("It is not proof of grounding, and the partner is shown.",
                 "A single example proves nothing about the population."):
        vc.assert_no_prohibited_interpretation(text, "test")


# --------------------------------------------------------------------------
# captions
# --------------------------------------------------------------------------

def test_every_caption_carries_its_limitation() -> None:
    payload = _load("captions.json")
    _check(payload["record_count"] == len(payload["records"]),
           "the caption record count disagrees with the records")
    for caption in payload["records"]:
        for part in ("message", "evidence", "illustrative_status"):
            _check(caption.get(part), f"{caption['artefact_id']} has no {part}")
        _check("illustrates an aggregate finding" in
               caption["illustrative_status"],
               f"{caption['artefact_id']} does not state its illustrative "
               f"status")
        _check("not itself" in caption["illustrative_status"],
               f"{caption['artefact_id']} does not deny population-level "
               f"status")
        if caption["kind"] != "dropped_category":
            _check(caption.get("limitation"),
                   f"{caption['artefact_id']} has no limitation")
        vc.assert_no_prohibited_interpretation(caption["caption"],
                                               caption["artefact_id"])
        vc.assert_no_prohibited_terminology(caption["caption"],
                                            caption["artefact_id"])


def test_every_selected_example_has_a_caption() -> None:
    captions = {c["artefact_id"] for c in _load("captions.json")["records"]}
    for record in _records():
        _check(f"VE2-CAP-{record['qualitative_id']}" in captions,
               f"{record['qualitative_id']} has no caption")


def test_the_mandatory_caveat_is_on_every_page() -> None:
    contract = Contract()
    caveat = contract.specification["mandatory_caption_caveat"]
    for name in ("ve2_gallery_main", "ve2_gallery_appendix"):
        sidecar = json.loads(
            (VE2 / "galleries" / f"{name}.data.json").read_text())
        _check(sidecar["mandatory_caption_caveat"] == caveat,
               f"{name} does not carry the VE-0 mandatory caveat verbatim")
        _check(sidecar["development_set_only"] is True,
               f"{name} does not declare development-set-only")


# --------------------------------------------------------------------------
# what was drawn
# --------------------------------------------------------------------------

def test_the_galleries_drew_exactly_the_records() -> None:
    """Every string on a page is the record's, and no label is offset."""
    by_id = {r["qualitative_id"]: r for r in _records()}
    drawn = 0
    for name in ("ve2_gallery_main", "ve2_gallery_appendix"):
        sidecar = json.loads(
            (VE2 / "galleries" / f"{name}.data.json").read_text())
        for page in sidecar["pages"]:
            for example in page["examples"]:
                record = by_id[example["qualitative_id"]]
                _check(record["category_id"] == page["category_id"],
                       f"{example['qualitative_id']} is drawn on the wrong "
                       f"category page")
                _check(example["question_text"] == record["question_text"],
                       "a drawn question is not the record's")
                _check(example["gold_answer"] == record["gold_answer"],
                       "a drawn gold answer is not the record's")
                _check(example["image_ids_drawn"][0] == record["image_id"],
                       "a drawn image is not the record's")
                published = {s["system_id"]: s for s in record["systems"]}
                _check({d["system_id"] for d in example["predictions"]} ==
                       set(published),
                       f"{example['qualitative_id']} drew a different set of "
                       f"systems")
                for prediction in example["predictions"]:
                    system = published[prediction["system_id"]]
                    _check(prediction["predicted_answer"] ==
                           system["predicted_answer"],
                           f"{example['qualitative_id']} drew "
                           f"{prediction['predicted_answer']!r} against "
                           f"{prediction['system_id']}, which answered "
                           f"{system['predicted_answer']!r}")
                    _check(prediction["correct"] == system["correct"],
                           "a drawn correctness is not the record's")
                    _check((prediction["mark_drawn"] == "✓") ==
                           system["correct"],
                           "the correctness mark contradicts the stored "
                           "correctness")
                    _check((prediction["word_drawn"] == "correct") ==
                           system["correct"],
                           "the correctness word contradicts the stored "
                           "correctness")
                    _check(prediction["display_label"] ==
                           system["display_label"],
                           "a prediction is drawn under a different label than "
                           "the record's")
                drawn += 1
    _check(drawn == len(by_id),
           f"{drawn} examples were drawn but {len(by_id)} records exist")


def test_placement_follows_the_frozen_specification() -> None:
    contract = Contract()
    main = set(contract.protocol["main_text_categories"])
    appendix = set(contract.protocol["appendix_categories"])
    for name, expected in (("ve2_gallery_main", main),
                           ("ve2_gallery_appendix", appendix)):
        sidecar = json.loads(
            (VE2 / "galleries" / f"{name}.data.json").read_text())
        drawn_categories = {p["category_id"] for p in sidecar["pages"]}
        _check(drawn_categories <= expected,
               f"{name} draws categories the frozen protocol places "
               f"elsewhere: {sorted(drawn_categories - expected)}")


# --------------------------------------------------------------------------
# provenance and rebuild
# --------------------------------------------------------------------------

def test_every_source_hash_resolves() -> None:
    registry = _load("provenance_registry.json")
    for entry in registry["entries"]:
        for source in entry["prediction_sources"]:
            if source["sha256"] == "NOT_APPLICABLE":
                continue
            path = PROJECT_ROOT / source["path"]
            _check(path.exists(), f"{source['path']} does not exist")
            _check(_sha256(path) == source["sha256"],
                   f"{source['path']} does not hash to the recorded value")
        for output in entry["outputs"]:
            path = PROJECT_ROOT / output["path"]
            _check(path.exists(), f"{output['path']} does not exist")
            _check(_sha256(path) == output["sha256"],
                   f"{output['path']} does not hash to the recorded value")


def test_the_provenance_chain_is_complete() -> None:
    """Artefact to category to rank to question to image to source to HEAD."""
    registry = _load("provenance_registry.json")
    required = {"ve2_artefact_id", "category_id", "selection_rank",
                "question_id", "image_id", "prediction_source_path",
                "prediction_source_sha256", "repository_head",
                "builder_sha256", "output_sha256"}
    _check(set(registry["chain"]) == required,
           "the declared provenance chain is not the one the task requires")
    questions = {r["question_id"] for r in _records()}
    recorded = {q for entry in registry["entries"]
                for q in entry["question_ids"]}
    _check(questions == recorded,
           "the provenance registry does not cover every selected question")
    for entry in registry["entries"]:
        _check(len(entry["repository_head"]) == 40,
               f"{entry['artefact_id']} records no repository HEAD")
        _check(len(entry["builder_sha256"]) == 64,
               f"{entry['artefact_id']} records no builder hash")


def test_content_digests_are_self_consistent() -> None:
    for name in ("qualitative_records.json", "selection_registry.json",
                 "captions.json", "visual_qa.json", "provenance_registry.json",
                 "VE2_MANIFEST.json"):
        payload = _load(name)
        _check(vc.content_digest(payload) == payload["content_sha256"],
               f"{name} does not hash to its own recorded content digest")


def test_rebuild_reproduces_the_scientific_content() -> None:
    """A second build from the same inputs reproduces every artefact.

    Content digests are compared for the artefacts that carry no output path,
    and the gallery bytes are compared directly. The gallery sidecars, the
    provenance registry and the manifest legitimately record where they were
    written, so they are compared after the rebuild directory is normalised
    back to the committed one.
    """
    from experiments.ve2 import run_ve2

    temporary = Path(tempfile.mkdtemp(prefix="ve2-rebuild-"))
    try:
        run_ve2.build(temporary)
        for name in ("qualitative_records.json", "selection_registry.json",
                     "captions.json", "visual_qa.json"):
            rebuilt = json.loads((temporary / name).read_text())
            _check(rebuilt["content_sha256"] == _load(name)["content_sha256"],
                   f"{name} did not reproduce on rebuild")
        for name in ("ve2_gallery_main", "ve2_gallery_appendix"):
            for suffix in (".pdf", ".png"):
                if suffix == ".png":
                    continue
                original = VE2 / "galleries" / f"{name}{suffix}"
                rebuilt = temporary / "galleries" / f"{name}{suffix}"
                _check(_sha256(original) == _sha256(rebuilt),
                       f"{name}{suffix} is not byte-reproducible")
            original = (VE2 / "galleries" / f"{name}.data.json").read_text()
            rebuilt = (temporary / "galleries"
                       / f"{name}.data.json").read_text()
            normalised = rebuilt.replace(vc.relpath(temporary),
                                         vc.relpath(VE2))
            _check(json.loads(normalised)["pages"] ==
                   json.loads(original)["pages"],
                   f"{name} drew different content on rebuild")
        for name in ("ve2_gallery_main_QC02_all_wrong.png",
                     "ve2_gallery_appendix_QC06_answer_side_disagreement.png"):
            _check(_sha256(VE2 / "galleries" / name) ==
                   _sha256(temporary / "galleries" / name),
                   f"{name} is not byte-reproducible")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def test_manifest_counts_agree_with_the_artefacts() -> None:
    manifest = _load("VE2_MANIFEST.json")
    counts = manifest["counts"]
    published = _records()
    _check(counts["examples_selected"] == len(published),
           "the manifest example count disagrees with the records")
    _check(counts["unique_questions"] == len({r["question_id"]
                                              for r in published}),
           "the manifest unique-question count is wrong")
    _check(counts["categories_populated"] +
           counts["categories_dropped"] == counts["categories_in_protocol"],
           "populated plus dropped categories do not account for the protocol")
    _check(counts["main_text_examples"] + counts["appendix_examples"] ==
           counts["examples_selected"],
           "main-text plus appendix examples do not account for the total")


# --------------------------------------------------------------------------
# scientific invariance
# --------------------------------------------------------------------------

def test_ve0_and_ve1_are_untouched() -> None:
    """VE-2 changed no frozen quantitative artefact.

    Each VE-0 and VE-1 artefact still hashes to its own recorded content
    digest, which is the same self-consistency the VE-1 suite checks. A VE-2
    build that had edited one of them would break this.
    """
    for directory in ("ve0", "ve1"):
        root = PROJECT_ROOT / "results" / directory
        for path in sorted(root.rglob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "content_sha256" not in payload:
                continue
            _check(vc.content_digest(payload) == payload["content_sha256"],
                   f"{path.relative_to(PROJECT_ROOT)} no longer hashes to its "
                   f"own recorded content digest; VE-2 must not have touched "
                   f"it")


def test_ve1_deferred_this_specification_to_ve2() -> None:
    blocked = json.loads(
        (PROJECT_ROOT / "results" / "ve1" /
         "blocked_specifications.json").read_text())
    deferred = [entry for entry in blocked["entries"]
                if entry["status"] == "DEFERRED_TO_VE2"]
    _check([entry["specification_id"] for entry in deferred] ==
           [vc.SPECIFICATION_ID],
           f"VE-1 deferred "
           f"{[e['specification_id'] for e in deferred]} to VE-2, not just "
           f"{vc.SPECIFICATION_ID}")
    manifest = _load("VE2_MANIFEST.json")
    _check(manifest["populates_specification"] == vc.SPECIFICATION_ID,
           "the VE-2 manifest populates a different specification")


TESTS = (
    test_frozen_protocol_is_unchanged,
    test_salt_is_the_frozen_one,
    test_rank_formula_matches_the_contract,
    test_every_category_predicate_matches_its_contract,
    test_recorded_ranks_recompute,
    test_selection_reproduces_from_the_published_pools,
    test_rank_depends_on_nothing_but_salt_category_and_question,
    test_a_skip_needs_a_mechanical_reason,
    test_published_skips_are_all_mechanical,
    test_every_record_matches_the_development_split,
    test_every_prediction_matches_its_frozen_source,
    test_predictions_are_row_aligned_not_positionally_assumed,
    test_no_e8b_row_is_used,
    test_compact_vlm_category_is_dropped_not_approximated,
    test_the_wrong_image_partner_is_proved_not_assumed,
    test_records_are_schema_exact_and_distinct,
    test_every_rendered_image_is_a_real_development_image,
    test_images_are_read_at_a_usable_size,
    test_no_embargoed_reference_anywhere_in_the_ve2_tree,
    test_the_embargo_guard_actually_refuses,
    test_scope_confirmations_are_all_negative,
    test_ve2_wrote_only_under_its_own_directory,
    test_the_overclaim_guard_refuses,
    test_every_caption_carries_its_limitation,
    test_every_selected_example_has_a_caption,
    test_the_mandatory_caveat_is_on_every_page,
    test_the_galleries_drew_exactly_the_records,
    test_placement_follows_the_frozen_specification,
    test_every_source_hash_resolves,
    test_the_provenance_chain_is_complete,
    test_content_digests_are_self_consistent,
    test_rebuild_reproduces_the_scientific_content,
    test_manifest_counts_agree_with_the_artefacts,
    test_ve0_and_ve1_are_untouched,
    test_ve1_deferred_this_specification_to_ve2,
)


def run() -> None:
    for test in TESTS:
        test()
    print(f"  {len(TESTS)} VE-2 checks passed")


if __name__ == "__main__":
    run()
