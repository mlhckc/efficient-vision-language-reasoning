"""Build the whole VE-2 package in one deterministic pass.

    python -m experiments.ve2.run_ve2

`--out-dir` exists so the validation suite can rebuild into an isolated
directory and compare content without touching the committed artefacts.

Order of operations, and why it is this order:

1. Load and hash the frozen VE-0 contract. A changed contract stops the build
   before anything is selected.
2. Load and prove every prediction source. Alignment failures stop the build
   before a pool exists, so no candidate can be formed from an unverified row.
3. Run the conditional verification the protocol requires for E9, and record
   why no category reads the E8B arrays.
4. Build candidate pools from the frozen predicates, order them by the frozen
   rank, and take the lowest ranks. Skips are recorded with a mechanical
   reason; the pool is built before the order and the order before any image
   is opened.
5. Render, caption, and record provenance.

Nothing here trains, evaluates, measures, scores or selects a checkpoint. The
only bytes it writes live under results/ve2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.ve2 import captions, evidence, gallery, records, selection
from experiments.ve2 import ve2_common as vc
from experiments.ve2 import visual_qa
from experiments.ve2.ve2_common import Contract, Recorder

BUILDER = "experiments/ve2/run_ve2.py"


# --------------------------------------------------------------------------
# evidence wiring
# --------------------------------------------------------------------------

def load_systems(split: evidence.DevelopmentSplit) -> tuple:
    """Every system any frozen category names, each proving its own identity."""
    closure = evidence.ClosureRowEvidence(split)
    e8a = evidence.E8APredictionTable(split)
    e10 = evidence.E10PerRow(split)

    systems = {
        "v2_02/question_only": closure.load(
            "v2_02/question_only", "v2_02", "question_only", "train_40k", 0,
            "normal"),
        "v2_02/concat": closure.load(
            "v2_02/concat", "v2_02", "concat", "train_40k", 0, "normal"),
        "v2_02/fusion": closure.load(
            "v2_02/fusion", "v2_02", "fusion", "train_40k", 0, "normal"),
        "e2/fusion": closure.load(
            "e2/fusion", "e2", "fusion", "train_40k", 0, "normal"),
        "v2_06/fusion@normal": closure.load(
            "v2_06/fusion@normal", "v2_06", "fusion", "train_40k", 0,
            "normal"),
        "v2_06/fusion@shuffled": closure.load(
            "v2_06/fusion@shuffled", "v2_06", "fusion", "train_40k", 0,
            "shuffled"),
        "E8A/A1": e8a.load("E8A/A1", "A1", "train_40k", 0, "normal"),
        "E8A/A1r": e8a.load("E8A/A1r", "A1r", "train_40k", 0, "normal"),
        "E10/B4": e10.load("E10/B4", "B4", "train_40k", 0),
        "E10/B4r": e10.load("E10/B4r", "B4r", "train_40k", 0),
    }
    return systems, closure


def assert_category_identity(contract: Contract, systems: dict) -> None:
    """Each category's declared scale, seed and condition are what was loaded."""
    for category_id, system_ids in records.CATEGORY_SYSTEMS.items():
        category = contract.category(category_id)
        records.assert_category_systems_match(category)
        for system_id in system_ids:
            loaded = systems[system_id]
            if loaded.scale != category["scale"]:
                raise AssertionError(
                    f"{category_id} declares scale {category['scale']} but "
                    f"{system_id} was loaded at {loaded.scale}")
            if loaded.seed != category["seed"]:
                raise AssertionError(
                    f"{category_id} declares seed {category['seed']} but "
                    f"{system_id} was loaded at seed {loaded.seed}")


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def select_category(contract: Contract, category_id: str, split, systems: dict,
                    resolver: evidence.ImageResolver, partner: dict,
                    sources: dict) -> dict:
    """Pool, order, take, record. One category."""
    category = contract.category(category_id)
    mask = selection.candidate_mask(category, systems, split.n_steps)
    pool = [int(i) for i in np.flatnonzero(mask)]

    def eligible(candidate):
        """Only the two mechanical reasons the frozen policy allows."""
        recheck = bool(mask[candidate.index])
        if not recheck:
            return (selection.SKIP_PREDICATE_FAILED_ON_RECHECK,
                    "the row no longer satisfies the frozen category "
                    "predicate on re-check")
        image = resolver.resolve(split.image_ids[candidate.index])
        if not image["readable"]:
            return (selection.SKIP_IMAGE_UNREADABLE, image["reason"])
        if category_id == "QC07_visual_reliance":
            partner_image = resolver.resolve(
                partner["partner_image_id"][candidate.index])
            if not partner_image["readable"]:
                return (selection.SKIP_IMAGE_UNREADABLE,
                        f"the wrong-image partner this row was shown is not "
                        f"readable: {partner_image['reason']}")
        return None

    wanted = contract.examples_per_category
    balance = selection.balance_directions(category, pool, systems)

    # The category's deterministic ordering is computed once, over its whole
    # candidate pool. A category that declares balanced directions selects
    # within each direction by FILTERING this ordering, so every example still
    # carries its rank in the category's single ordering. That matters for more
    # than tidiness: ranking within each direction separately gave both QC06
    # examples rank 1 and therefore the same qualitative_id, which the
    # uniqueness guard in `build` now refuses outright.
    full_order = selection.order_pool(contract.salt, category_id, pool,
                                      split.question_ids)
    groups = []

    if balance is None:
        chosen, trailing = selection.select(full_order, wanted, eligible)
        groups.append({"direction_id": "ALL", "label": "no balance "
                       "requirement is declared for this category",
                       "pool_size": len(pool), "chosen": chosen,
                       "trailing_skips": trailing})
    else:
        if wanted % len(balance["directions"]) != 0:
            raise AssertionError(
                f"{category_id} asks for {wanted} examples across "
                f"{len(balance['directions'])} balanced directions, which "
                f"does not divide evenly; VE-2 refuses to produce an "
                f"unbalanced gallery")
        per_direction = wanted // len(balance["directions"])
        for direction in balance["directions"]:
            members = set(direction["indices"])
            candidates = [c for c in full_order if c.index in members]
            chosen, trailing = selection.select(candidates, per_direction,
                                                eligible)
            groups.append({"direction_id": direction["direction_id"],
                           "label": direction["label"],
                           "pool_size": len(direction["indices"]),
                           "chosen": chosen, "trailing_skips": trailing})

    collisions = selection.detect_rank_collisions(full_order)

    selected, registry_rows = [], []
    for group in groups:
        for candidate, skipped in group["chosen"]:
            image = resolver.resolve(split.image_ids[candidate.index])
            partner_record = None
            if category_id == "QC07_visual_reliance":
                partner_id = partner["partner_image_id"][candidate.index]
                resolved = resolver.resolve(partner_id)
                partner_record = {
                    "image_id": str(partner_id),
                    "image_path": resolved["image_path"],
                    "image_sha256": resolved["image_sha256"],
                    "recovered_by": partner["proof"]["method"],
                    "proof": partner["proof"],
                }
            record = records.build_record(
                category, candidate, skipped, split, systems, image,
                partner_record, sources)
            records.assert_schema_exact(record, contract.record_schema)
            selected.append(record)
            registry_rows.append({
                "qualitative_id": record["qualitative_id"],
                "direction_id": group["direction_id"],
                "direction_label": group["label"],
                "selection_rank": candidate.rank,
                "selection_rank_key": candidate.rank_key,
                "question_id": candidate.question_id,
                "image_id": record["image_id"],
                "skipped_ranks": skipped,
            })

    if len(selected) != wanted:
        raise AssertionError(
            f"{category_id} yielded {len(selected)} examples where the frozen "
            f"rule asks for {wanted}")

    return {
        "category_id": category_id,
        "status": "POPULATED",
        "placement": category["placement"],
        "predicate": category["predicate"],
        "evidence_source": category["evidence_source"],
        "scale": category["scale"],
        "seed": category["seed"],
        "condition": category["condition"],
        "systems": records.CATEGORY_SYSTEMS[category_id],
        "candidate_pool_size": len(pool),
        "candidate_pool_share_of_development_split":
            round(len(pool) / evidence.DEV_ROWS, 6),
        "examples_requested": wanted,
        "examples_selected": len(selected),
        "balance_requirement": (balance["requirement"] if balance
                                else "NOT_APPLICABLE"),
        "direction_pool_sizes": [{"direction_id": g["direction_id"],
                                  "pool_size": g["pool_size"]}
                                 for g in groups],
        "skips": [s for group in groups
                  for _, skipped in group["chosen"] for s in skipped],
        "skip_count": sum(len(skipped) for group in groups
                          for _, skipped in group["chosen"]),
        "rank_key_collisions": collisions,
        "selected": registry_rows,
        "records": selected,
    }


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(out_dir: Path | None = None) -> dict:
    contract = Contract()
    vc.assert_rank_formula_matches(contract)
    recorder = Recorder(contract)
    root = Path(out_dir) if out_dir else vc.VE2_DIR
    gallery_dir = root / "galleries"

    split = evidence.DevelopmentSplit()
    systems, _ = load_systems(split)
    assert_category_identity(contract, systems)
    resolver = evidence.ImageResolver(split)
    partner = evidence.shuffled_partner(split, evidence.ClosureRowEvidence(split))

    e9_finding = evidence.verify_e9_recoverability(contract.protocol)
    e8b_note = evidence.record_e8b_unused(contract.protocol)

    sources = {
        "development_split": split.sources(),
        "prediction_sources": sorted(
            {(s.source_path, s.source_sha256, s.hash_bound_in)
             for s in systems.values()}),
        "shuffled_partner_recovery": partner["sources"],
    }
    sources["prediction_sources"] = [
        {"path": p, "sha256": h, "hash_bound_in": b}
        for p, h, b in sources["prediction_sources"]]

    available = [c for c in contract.protocol["available_categories"]]
    dropped = {}
    for category_id, category in contract.categories.items():
        if category_id in available:
            continue
        if category["evidence_source"] == "e9_tokens":
            dropped[category_id] = e9_finding
        else:
            raise AssertionError(
                f"{category_id} is not in the available list and VE-2 has no "
                f"recorded finding for it")

    results = {}
    for category_id in available:
        results[category_id] = select_category(
            contract, category_id, split, systems, resolver, partner, sources)

    records_by_category = {k: v["records"] for k, v in results.items()}
    _assert_records_are_distinct(records_by_category)
    main_ids = [c for c in contract.protocol["main_text_categories"]
                if c in results]
    appendix_ids = [c for c in contract.protocol["appendix_categories"]
                    if c in results]

    galleries = {}
    galleries["main"] = gallery.render(
        "ve2_gallery_main", "Qualitative development examples",
        main_ids, contract, records_by_category, gallery_dir)
    galleries["appendix"] = gallery.render(
        "ve2_gallery_appendix",
        "Qualitative development examples, appendix",
        appendix_ids, contract, records_by_category, gallery_dir)

    caption_records = []
    for gallery_key, ids in (("main", main_ids), ("appendix", appendix_ids)):
        gallery_id = galleries[gallery_key]["artefact_id"]
        for category_id in ids:
            caption_records.append(captions.page_caption(
                contract, category_id, records_by_category[category_id],
                gallery_id))
    for category_id in available:
        for record in records_by_category[category_id]:
            caption_records.append(captions.example_caption(contract, record))
    for category_id, finding in dropped.items():
        caption_records.append(captions.dropped_caption(contract, category_id,
                                                        finding))

    for key, payload in galleries.items():
        recorder.record(
            artefact_id=payload["artefact_id"], kind="gallery",
            outputs=payload["outputs"], builder=BUILDER,
            sources=(sources["prediction_sources"]
                     + [s for s in sources["development_split"]]),
            categories=[p["category_id"] for p in payload["sidecar"]["pages"]],
            question_ids=[e["question_id"] for p in payload["sidecar"]["pages"]
                          for e in p["examples"]],
            notes={"placement": payload["sidecar"]["placement"],
                   "pages": payload["sidecar"]["page_count"]})

    outputs = {}
    outputs["records"] = _write_records(root, contract, results, sources)
    outputs["registry"] = _write_registry(root, contract, results, dropped,
                                          e8b_note, partner)
    outputs["captions"] = _write_captions(root, contract, caption_records)
    outputs["visual_qa"] = _write_visual_qa(root, contract, galleries)
    outputs["provenance"] = _write_provenance(root, contract, recorder,
                                              outputs, sources)
    outputs["manifest"] = _write_manifest(root, contract, results, dropped,
                                          galleries, caption_records, outputs,
                                          e8b_note, partner, sources)
    return {"contract": contract, "results": results, "dropped": dropped,
            "galleries": galleries, "captions": caption_records,
            "outputs": outputs, "root": root, "recorder": recorder,
            "split": split}


def _assert_records_are_distinct(records_by_category: dict) -> None:
    """No two records may share an identifier, and none may repeat a row.

    The first implementation ranked each of QC06's balanced directions
    separately, so both of its examples came out as rank 1 and collided on
    `VE2-QUAL-QC06_answer_side_disagreement-01`. Two different questions
    carrying one identifier is a provenance failure: a caption, a record and a
    gallery row could no longer be matched to each other. The ranking was
    repaired and this guard is what makes the repair permanent.
    """
    seen_ids: dict = {}
    for category_id, records in sorted(records_by_category.items()):
        seen_questions: dict = {}
        for record in records:
            identifier = record["qualitative_id"]
            if identifier in seen_ids:
                raise AssertionError(
                    f"two qualitative records share the identifier "
                    f"{identifier}: questions {seen_ids[identifier]} and "
                    f"{record['question_id']}")
            seen_ids[identifier] = record["question_id"]
            question = record["question_id"]
            if question in seen_questions:
                raise AssertionError(
                    f"{category_id} selected question {question} twice, as "
                    f"{seen_questions[question]} and {identifier}")
            seen_questions[question] = identifier


def _finalise(payload: dict, contract: Contract, sources: list) -> dict:
    payload["ve0_source_artefacts"] = contract.source_inputs()
    payload["ve0_specification_id"] = vc.SPECIFICATION_ID
    payload["development_set_only"] = True
    payload["clean_test_accessed"] = False
    payload["provenance"] = vc.provenance(BUILDER, inputs=sources)
    payload["content_sha256"] = vc.content_digest(payload)
    return payload


def _all_sources(sources: dict) -> list:
    return (sources["development_split"] + sources["prediction_sources"]
            + sources["shuffled_partner_recovery"])


def _write_records(root: Path, contract: Contract, results: dict,
                   sources: dict) -> dict:
    every = [r for value in results.values() for r in value["records"]]
    payload = _finalise({
        "title": "VE-2 qualitative records",
        "ve2_output": "qualitative_records",
        "rule": "one record per selected example, carrying exactly the "
                "nineteen fields the frozen VE-0 record schema declares. "
                "Every value is copied from a hash-checked source; no number "
                "and no prediction string is written by hand.",
        "record_schema_sha256": contract.hashes["I"]["sha256"],
        "record_count": len(every),
        "records": sorted(every, key=lambda r: r["qualitative_id"]),
    }, contract, _all_sources(sources))
    path = root / "qualitative_records.json"
    return {"path": vc.relpath(path), "sha256": vc.write_json(path, payload),
            "content_sha256": payload["content_sha256"]}


def _write_registry(root: Path, contract: Contract, results: dict,
                    dropped: dict, e8b_note: dict, partner: dict) -> dict:
    payload = _finalise({
        "title": "VE-2 selection and skip registry",
        "ve2_output": "selection_registry",
        "rule": "the complete audit of the selection: every category's "
                "candidate pool size, the frozen rank of every example taken, "
                "every rank skipped with its mechanical reason, and every "
                "category not populated with the finding that dropped it.",
        "selection_rule": contract.selection_rule,
        "salt": contract.salt,
        "salt_frozen_before_inspection": True,
        "salt_source": "results/ve0/qualitative_protocol.json, read at build "
                       "time and never copied into VE-2 source",
        "allowed_skip_reasons": list(selection.ALLOWED_SKIP_REASONS),
        "appearance_played_no_part": (
            "the rank key is SHA-256 over the salt, the category identifier "
            "and the question identifier. It cannot see the image, the model, "
            "the prediction or the outcome, and the candidate pool is built "
            "before the order is computed and the order before any image file "
            "is opened."),
        "categories_populated": len(results),
        "categories_dropped": len(dropped),
        "total_examples_selected": sum(len(v["records"])
                                       for v in results.values()),
        "total_skips": sum(v["skip_count"] for v in results.values()),
        "categories": sorted(
            [{k: v for k, v in value.items() if k != "records"}
             for value in results.values()],
            key=lambda c: c["category_id"]),
        "dropped_categories": sorted(dropped.values(),
                                     key=lambda d: d["category_id"]),
        "unused_prediction_sources": [e8b_note],
        "shuffled_partner_recovery": partner["proof"],
    }, contract, _all_sources({"development_split": [],
                               "prediction_sources": [],
                               "shuffled_partner_recovery":
                                   partner["sources"]}))
    path = root / "selection_registry.json"
    return {"path": vc.relpath(path), "sha256": vc.write_json(path, payload),
            "content_sha256": payload["content_sha256"]}


def _write_captions(root: Path, contract: Contract, records: list) -> dict:
    payload = _finalise({
        "title": "VE-2 canonical caption records",
        "ve2_output": "captions",
        "rule": "every caption carries four parts: the MESSAGE, the EVIDENCE "
                "behind it, its ILLUSTRATIVE STATUS, and the LIMITATION. The "
                "illustrative-status sentence is mandatory and states that "
                "the panel illustrates an aggregate finding and is not itself "
                "population-level statistical evidence.",
        "record_count": len(records),
        "records": sorted(records, key=lambda r: r["artefact_id"]),
    }, contract, [])
    path = root / "captions.json"
    return {"path": vc.relpath(path), "sha256": vc.write_json(path, payload),
            "content_sha256": payload["content_sha256"]}


def _write_visual_qa(root: Path, contract: Contract, galleries: dict) -> dict:
    payload = _finalise({
        "title": "VE-2 visual quality-assurance record",
        "ve2_output": "visual_qa",
        "rule": "every rendered page was displayed and inspected against the "
                "checklist below. Findings are kept after repair, because a "
                "defect found and fixed is the evidence that the pass was "
                "real.",
        "checklist": visual_qa.CHECKLIST,
        "pages_inspected": sum(len(g["sidecar"]["pages"])
                               for g in galleries.values()),
        "records": sorted(visual_qa.RECORDS, key=lambda r: r["artefact_id"]),
        "programmatic_validation_is_not_sufficient":
            "the validation suite proves that the drawn strings match the "
            "records and that the records match their frozen sources. It "
            "cannot see a prediction label sitting against the wrong system, "
            "an image squashed out of its aspect ratio, or a line of text "
            "running off the page, which is why this pass exists.",
    }, contract, [])
    path = root / "visual_qa.json"
    return {"path": vc.relpath(path), "sha256": vc.write_json(path, payload),
            "content_sha256": payload["content_sha256"]}


def _write_provenance(root: Path, contract: Contract, recorder: Recorder,
                      outputs: dict, sources: dict) -> dict:
    payload = _finalise({
        "title": "VE-2 provenance registry",
        "ve2_output": "provenance_registry",
        "rule": "every artefact resolves through the chain the task requires: "
                "VE-2 artefact identifier, qualitative category, deterministic "
                "rank, question identifier, image identifier, frozen "
                "prediction source, source hash, repository HEAD, builder "
                "hash and output hash.",
        "chain": ["ve2_artefact_id", "category_id", "selection_rank",
                  "question_id", "image_id", "prediction_source_path",
                  "prediction_source_sha256", "repository_head",
                  "builder_sha256", "output_sha256"],
        "entry_count": len(recorder.entries),
        "entries": sorted(recorder.entries, key=lambda e: e["artefact_id"]),
        "prediction_sources": sources["prediction_sources"],
        "development_split_sources": sources["development_split"],
        "registry_outputs": {k: v for k, v in outputs.items()},
    }, contract, _all_sources(sources))
    path = root / "provenance_registry.json"
    return {"path": vc.relpath(path), "sha256": vc.write_json(path, payload),
            "content_sha256": payload["content_sha256"]}


def _write_manifest(root: Path, contract: Contract, results: dict,
                    dropped: dict, galleries: dict, caption_records: list,
                    outputs: dict, e8b_note: dict, partner: dict,
                    sources: dict) -> dict:
    payload = _finalise({
        "title": "VE-2 manifest: deterministic qualitative development "
                 "evidence",
        "ve2_output": "manifest",
        "rule": "VE-2 populates the one VE-0 specification VE-1 deferred to "
                "it, VE0-FIG-10, from frozen development evidence and nothing "
                "else. It trains nothing, evaluates nothing, measures "
                "nothing, scores nothing, selects no checkpoint and reads no "
                "clean-test material.",
        "populates_specification": vc.SPECIFICATION_ID,
        "counts": {
            "categories_in_protocol": contract.protocol["category_count"],
            "categories_available": len(contract.protocol[
                "available_categories"]),
            "categories_populated": len(results),
            "categories_dropped": len(dropped),
            "examples_selected": sum(len(v["records"])
                                     for v in results.values()),
            "main_text_examples": sum(
                len(v["records"]) for v in results.values()
                if v["placement"] == "MAIN_TEXT"),
            "appendix_examples": sum(
                len(v["records"]) for v in results.values()
                if v["placement"] == "APPENDIX"),
            "unique_questions": len({r["question_id"] for v in results.values()
                                     for r in v["records"]}),
            "unique_images_rendered": len(
                {r["image_id"] for v in results.values()
                 for r in v["records"]}
                | {s["displayed_image"]["image_id"]
                   for v in results.values() for r in v["records"]
                   for s in r["systems"]}),
            "skips_recorded": sum(v["skip_count"] for v in results.values()),
            "caption_records": len(caption_records),
            "prediction_sources_read": len(sources["prediction_sources"]),
        },
        "galleries": {key: {"artefact_id": g["artefact_id"],
                            "placement": g["sidecar"]["placement"],
                            "pages": g["sidecar"]["page_count"],
                            "examples": g["sidecar"]["example_count"],
                            "outputs": g["outputs"]}
                      for key, g in galleries.items()},
        "dropped_categories": [
            {"category_id": k, "outcome": v["outcome"],
             "authority": v["authority"], "finding": v["finding"]}
            for k, v in sorted(dropped.items())],
        "unused_prediction_sources": [e8b_note],
        "shuffled_partner_recovery_verified": partner["proof"]["verified"],
        "registry_outputs": outputs,
        "scope_confirmations": {
            "trained_anything": False,
            "evaluated_anything": False,
            "measured_anything": False,
            "rescored_anything": False,
            "generated_any_prediction": False,
            "selected_any_checkpoint": False,
            "read_any_checkpoint": False,
            "read_any_embedding_store": False,
            "clean_test_accessed": False,
            "clean_test_path_resolved": False,
            "clean_test_value_present": False,
            "gpu_used": False,
            "gpu_hours_charged": 0.0,
            "wrote_under_results_closure": False,
            "wrote_under_results_experiments": False,
            "wrote_under_results_ve0": False,
            "wrote_under_results_ve1": False,
            "modified_tests_run_all": False,
            "changed_any_ve1_figure_or_table": False,
            "changed_any_scientific_aggregate": False,
            "f1_started": False,
            "f2_started": False,
        },
        "clean_test_governance": (
            "The clean-test contents were never inspected or used for "
            "development, model selection, or reporting decisions. Its bytes "
            "were mechanically read once by an independent reviewer integrity-"
            "hash command on 13 August 2026. VE-2 did not open, hash, stat or "
            "parse the clean-test target, and no VE-2 module resolves its "
            "path."),
        "determinism_contract": {
            "tier_1_scientific_content": "content_sha256 over each artefact's "
                                         "payload with provenance and the "
                                         "digest itself removed is invariant "
                                         "given unchanged inputs.",
            "tier_2_provenance_bearing_bytes": "the full-file hash covers the "
                                               "provenance block, which "
                                               "records the repository HEAD "
                                               "and moves when artefacts are "
                                               "committed.",
            "galleries": "PDF and PNG bytes are deterministic: no wall-clock "
                         "metadata is written, the hash salt is pinned and "
                         "fonts are embedded as TrueType. Each gallery also "
                         "emits exactly what it drew as a JSON sidecar, which "
                         "is what the rebuild check compares.",
        },
    }, contract, _all_sources(sources))
    path = root / "VE2_MANIFEST.json"
    return {"path": vc.relpath(path), "sha256": vc.write_json(path, payload),
            "content_sha256": payload["content_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None,
                        help="write into this directory instead of "
                             "results/ve2; used by the validation suite")
    args = parser.parse_args()
    result = build(Path(args.out_dir) if args.out_dir else None)
    print(f"VE-2 built into {vc.relpath(result['root'])}")
    print(f"  categories populated : {len(result['results'])}")
    print(f"  categories dropped   : {len(result['dropped'])}")
    print(f"  examples selected    : "
          f"{sum(len(v['records']) for v in result['results'].values())}")
    print(f"  skips recorded       : "
          f"{sum(v['skip_count'] for v in result['results'].values())}")
    print(f"  captions             : {len(result['captions'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
