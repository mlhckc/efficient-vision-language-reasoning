"""Output A/B: the canonical evidence inventory.

One row per quantity that is allowed to appear in the dissertation, the
supervisor presentation or an appendix, plus explicit rows for the quantities
that are NOT allowed to appear. Every number is read from a frozen artefact
that the independent closure review verified; nothing is recomputed here.

The reporting boundaries attached to each row come from policy_evidence.py.
The figure, table, section and slide back-references are filled in later by
run_ve0, from the authored specifications, so the inventory and the
specifications can never disagree about who consumes what.
"""

from __future__ import annotations

from . import policy_evidence as pol
from . import ve0_common as vc

SPLITS = {
    "v2_dev": {
        "dataset": "GQA, image-disjoint V2 development partition",
        "split": "development",
        "n_questions": 7714,
        "n_unique_images": 768,
        "answer_set_size": 100,
        "note": "the in-vocabulary top-100 view; 768 of the 777 development "
                "images are represented once the vocabulary gate is applied",
    },
    "v2_1000_dev": {
        "dataset": "GQA, image-disjoint V2 development partition",
        "split": "development",
        "n_questions": 9823,
        "n_unique_images": 776,
        "answer_set_size": 1000,
        "note": "the E3 top-1000 view. Never paired with the top-100 view",
    },
    "v2_dev_raw": {
        "dataset": "GQA, image-disjoint V2 development partition",
        "split": "development",
        "n_questions": 10004,
        "n_unique_images": 777,
        "answer_set_size": None,
        "note": "the raw development partition before the vocabulary gate, "
                "used by E8B's raw denominators and by E9",
    },
}


def _split_for(family: str, n_questions, answer_set_size) -> str:
    if answer_set_size == 1000 or family.lower() == "e3":
        return "v2_1000_dev"
    if n_questions == 10004:
        return "v2_dev_raw"
    return "v2_dev"


def _role(family: str) -> str:
    return pol.FAMILY_ROLE.get(family, pol.FAMILY_ROLE.get(family.upper(),
                                                           "SUPPORTING"))


def _stage(family: str) -> str:
    return pol.FAMILY_STAGE.get(family,
                                pol.FAMILY_STAGE.get(family.upper(),
                                                     "S00_OFF_SEQUENCE"))


def _metric_id(family: str) -> str:
    return pol.FAMILY_METRIC.get(family,
                                 pol.FAMILY_METRIC.get(family.upper(),
                                                       "UNKNOWN"))


def _limitation(family: str, key: str | None = None) -> str:
    parts = [pol.UNIVERSAL_LIMITATION]
    family_note = pol.FAMILY_LIMITATION.get(
        family, pol.FAMILY_LIMITATION.get(family.upper()))
    if family_note:
        parts.append(family_note)
    if key and key in pol.CONTRAST_LIMITATION:
        parts.append(pol.CONTRAST_LIMITATION[key])
    return " ".join(parts)


def _boundaries(comparison_class: str) -> tuple[str, str]:
    entry = pol.COMPARISON_CLASSES[comparison_class]
    return entry["allowed_claim"], entry["forbidden_stronger_claim"]


def _blank_row() -> dict:
    """Every field exists on every row. Absence is explicit, never silent."""
    return {
        "evidence_id": None,
        "evidence_class": None,
        "experiment_family": None,
        "story_stage": None,
        "scientific_role": None,
        "source_key": None,
        "quantity": None,
        "canonical_source_path": vc.NOT_APPLICABLE,
        "canonical_source_sha256": vc.NOT_APPLICABLE,
        "row_level_evidence_paths": [],
        "row_level_sha256": [],
        "checkpoint_identity": [],
        "checkpoint_sha256": [],
        "split": vc.NOT_APPLICABLE,
        "dataset": vc.NOT_APPLICABLE,
        "training_scale": vc.NOT_APPLICABLE,
        "model_system": vc.NOT_APPLICABLE,
        "seed_set": vc.NOT_APPLICABLE,
        "n_training_seeds": None,
        "point_estimate": None,
        "across_training_seed_sd_ddof1": None,
        # Per-seed reporting fields, added by the 2026-08-13 amendment. They
        # bind values that were already computed and already hashed before
        # VE-0 froze; nothing here is recomputed. A row that has none says so
        # in per_seed_status rather than carrying a silent null.
        "per_seed_values": None,
        "per_seed_seed_ids": None,
        "per_seed_range": None,
        "per_seed_effects": None,
        "per_seed_effect_range": None,
        "per_seed_status": vc.NOT_APPLICABLE,
        "per_seed_provenance": vc.NOT_APPLICABLE,
        "ci95": None,
        "ci_type": vc.NOT_APPLICABLE,
        "cluster_unit": vc.NOT_APPLICABLE,
        "n_resamples": None,
        "rng_seed": None,
        "n_questions": None,
        "n_unique_images": None,
        "answer_set_size": None,
        "metric_id": vc.NOT_APPLICABLE,
        "scoring_implementation": vc.NOT_APPLICABLE,
        "analysis_role": vc.NOT_APPLICABLE,
        "comparison_class": vc.NOT_APPLICABLE,
        "evidence_readiness": vc.NOT_APPLICABLE,
        "allowed_claim": vc.NOT_APPLICABLE,
        "forbidden_stronger_claim": vc.NOT_APPLICABLE,
        "mandatory_limitation": vc.NOT_APPLICABLE,
        "excludes_zero": None,
        "directional": None,
        # filled by run_ve0 from the authored specifications
        "proposed_figure_use": [],
        "proposed_table_use": [],
        "proposed_dissertation_section": [],
        "proposed_supervisor_slide": [],
        "appendix_destination": vc.NOT_APPLICABLE,
        "qualitative_pairing": vc.NOT_APPLICABLE,
    }


# --------------------------------------------------------------------------
# per-seed binding, added by the 2026-08-13 amendment
# --------------------------------------------------------------------------
# The independent VE-1 review found that VE0-FIG-A2, VE0-TAB-04, VE0-TAB-05
# and VE0-TAB-A2 all ask for per-seed values the inventory did not bind, even
# though those values already exist, fully computed and hash-pinned, inside
# two of VE-0's own declared closure inputs. This binds them. It reads; it
# does not compute. The mean, the standard deviation and the range stored
# beside each vector are re-derived only to REFUSE a vector that does not
# reproduce them, which is a consistency guard rather than a new statistic.

ROUNDING = 5
TOLERANCE = 1.1e-5


def _mean(values: list) -> float:
    return round(sum(values) / len(values), ROUNDING)


def _sd_ddof1(values: list) -> float:
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return round(variance ** 0.5, ROUNDING)


def _per_seed_vector(payload, seed_set: list) -> tuple:
    """Normalise the two stored shapes into (seed_ids, values).

    The closure stores per-seed accuracy as a seed-keyed mapping and per-seed
    effects either as a mapping or as a list ordered by the row's own
    training_seeds. Both are accepted; nothing else is.
    """
    if payload is None:
        return None, None
    if isinstance(payload, dict):
        ids = sorted(payload, key=int)
        return [int(i) for i in ids], [payload[i] for i in ids]
    if isinstance(payload, list):
        if seed_set is None or len(seed_set) != len(payload):
            raise AssertionError(
                "a list-shaped per-seed vector must align with the row's own "
                f"seed set: {payload} against {seed_set}")
        return list(seed_set), list(payload)
    raise AssertionError(f"unrecognised per-seed payload shape: {payload!r}")


def _per_seed_provenance(ctx, closure_key: str, row_id: str, field: str,
                         upstream_path: str) -> dict:
    """Exactly where a bound vector was read from, and what it came from."""
    closure_path = ctx["input_paths"][closure_key]
    return {
        "read_from": closure_path,
        "read_from_sha256": ctx["hashes"].get(closure_path, vc.NOT_AVAILABLE),
        "locator": f"rows[row_id={row_id}].{field}"
        if closure_key == "seed" else
        f"contrasts[contrast_id={row_id}].{field}",
        "upstream_canonical_source_path": upstream_path,
        "upstream_canonical_source_sha256": ctx["hashes"].get(
            upstream_path, vc.NOT_AVAILABLE),
        "computation": "READ_ONLY. The vector was read from a frozen closure "
                       "output that VE-0 already declares as an input and "
                       "already hashes. No value was recomputed, no model was "
                       "loaded and no evaluation was run.",
        "bound_by": "VE-0 amendment 2026-08-13",
    }


def _bind_seed_vector(ctx, row: dict, source: dict) -> None:
    """Attach the per-seed accuracies to one SEED row, or say why not."""
    ids, values = _per_seed_vector(source.get("per_seed_accuracy"),
                                   source.get("seed_set"))
    if values is None:
        row["per_seed_status"] = (
            "NOT_AVAILABLE_IN_FROZEN_SOURCE: the closure seed-variability "
            "artefact stores no per-seed accuracy for this row")
        return
    if ids != list(source["seed_set"]):
        raise AssertionError(
            f"{source['row_id']}: per-seed identifiers {ids} do not match the "
            f"row's seed set {source['seed_set']}")
    if len(values) != source["n_training_seeds"]:
        raise AssertionError(
            f"{source['row_id']}: {len(values)} per-seed values against "
            f"n_training_seeds {source['n_training_seeds']}")
    if abs(_mean(values) - source["mean"]) > TOLERANCE:
        raise AssertionError(
            f"{source['row_id']}: per-seed values mean to {_mean(values)}, "
            f"the stored mean is {source['mean']}")
    if len(values) > 1 and abs(_sd_ddof1(values)
                               - source["sd_across_training_seeds_ddof1"]
                               ) > TOLERANCE:
        raise AssertionError(
            f"{source['row_id']}: per-seed values give sd "
            f"{_sd_ddof1(values)}, the stored sd is "
            f"{source['sd_across_training_seeds_ddof1']}")
    stored_range = source.get("range")
    derived_range = round(max(values) - min(values), ROUNDING)
    if stored_range is not None and abs(derived_range
                                        - stored_range) > TOLERANCE:
        raise AssertionError(
            f"{source['row_id']}: per-seed range {derived_range} against the "
            f"stored range {stored_range}")
    row["per_seed_values"] = values
    row["per_seed_seed_ids"] = ids
    row["per_seed_range"] = stored_range if stored_range is not None \
        else derived_range
    row["per_seed_status"] = "BOUND"
    row["per_seed_provenance"] = _per_seed_provenance(
        ctx, "seed", source["row_id"], "per_seed_accuracy",
        source["source_artefact"])


def _bind_contrast_vector(ctx, row: dict, source: dict) -> None:
    """Attach the per-seed matched effects to one CON row, or say why not."""
    ids, values = _per_seed_vector(source.get("per_seed_effect"),
                                   source.get("training_seeds"))
    if values is None:
        row["per_seed_status"] = (
            "NOT_AVAILABLE_IN_FROZEN_SOURCE: the closure contrast artefact "
            "stores no per-seed effect vector for this contrast")
        return
    if ids != list(source["training_seeds"]):
        raise AssertionError(
            f"{source['contrast_id']}: per-seed identifiers {ids} do not "
            f"match the contrast's seed set {source['training_seeds']}")
    if abs(_mean(values) - source["effect"]) > TOLERANCE:
        raise AssertionError(
            f"{source['contrast_id']}: per-seed effects mean to "
            f"{_mean(values)}, the stored effect is {source['effect']}")
    sd = source.get("across_training_seed_sd_ddof1")
    if sd is not None and len(values) > 1 \
            and abs(_sd_ddof1(values) - sd) > TOLERANCE:
        raise AssertionError(
            f"{source['contrast_id']}: per-seed effects give sd "
            f"{_sd_ddof1(values)}, the stored sd is {sd}")
    row["per_seed_effects"] = values
    row["per_seed_seed_ids"] = ids
    row["per_seed_effect_range"] = round(max(values) - min(values), ROUNDING)
    row["per_seed_status"] = "BOUND"
    row["per_seed_provenance"] = _per_seed_provenance(
        ctx, "contrasts", source["contrast_id"], "per_seed_effect",
        source["source_artefact"])


# --------------------------------------------------------------------------
# per-class builders
# --------------------------------------------------------------------------

def _contrast_rows(ctx) -> list:
    rows = []
    derived = {c["contrast_id"]: c
               for c in ctx["reconstructed"]["contrasts"]}
    for c in ctx["contrasts"]["contrasts"]:
        family = c["family"]
        key = c["contrast_id"]
        row = _blank_row()
        split_id = _split_for(family, c["n_questions"], None)
        split = SPLITS[split_id]
        allowed, forbidden = _boundaries(c["comparison_status"])
        extra = derived.get(key, {})
        row.update({
            "evidence_id": vc.evidence_id("CON", key),
            "evidence_class": "CON",
            "experiment_family": family,
            "story_stage": _stage(family),
            "scientific_role": _role(family),
            "source_key": key,
            "quantity": "paired contrast, effect in accuracy points",
            "canonical_source_path": c["source_artefact"],
            "canonical_source_sha256": ctx["hashes"].get(
                c["source_artefact"], vc.NOT_AVAILABLE),
            "split": split_id,
            "dataset": split["dataset"],
            "training_scale": _scale_from_key(key),
            "model_system": _systems_from(extra, key),
            "seed_set": c["training_seeds"],
            "n_training_seeds": len(c["training_seeds"]),
            "point_estimate": c["point_estimate"],
            "across_training_seed_sd_ddof1":
                c["across_training_seed_sd_ddof1"],
            "ci95": c["ci95_image_clustered"],
            "ci_type": c["interval_kind"],
            "cluster_unit": c["cluster_unit"],
            "n_resamples": c["n_resamples"],
            "rng_seed": c["rng_seed"],
            "n_questions": c["n_questions"],
            "n_unique_images": c["n_unique_images"],
            "answer_set_size": split["answer_set_size"],
            "metric_id": _metric_id(family),
            "scoring_implementation": c["metric"],
            "analysis_role": c["analysis_role"],
            "comparison_class": c["comparison_status"],
            "evidence_readiness": ("READY" if c["ci95_image_clustered"]
                                   else "NO_INTERVAL"),
            "allowed_claim": allowed,
            "forbidden_stronger_claim": forbidden,
            "mandatory_limitation": _limitation(family, key),
            "excludes_zero": c["excludes_zero"],
            "directional": c["excludes_zero"],
        })
        _bind_contrast_vector(ctx, row, c)
        row["row_level_evidence_paths"], row["row_level_sha256"] = \
            _row_evidence_for(ctx, family, _scale_from_key(key))
        rows.append(row)
    return rows


def _scale_from_key(key: str) -> str:
    for scale in ("train_40k", "train_100k", "train_250k"):
        if scale in key:
            return scale
    if key.startswith("E10/scale_effect") or key.startswith("E8B/scale_effect"):
        return "train_40k vs train_250k"
    if key.endswith("difference_in_differences"):
        return "train_40k vs train_250k"
    return vc.NOT_APPLICABLE


def _systems_from(extra: dict, key: str) -> str:
    left, right = extra.get("left"), extra.get("right")
    if left and right:
        return f"{left} minus {right}"
    return key.split("/")[-2] if key.count("/") >= 2 else key


def _row_evidence_for(ctx, family: str, scale: str) -> tuple[list, list]:
    paths, hashes = [], []
    for entry in ctx["reconstructed_manifest"]["row_evidence"]:
        if entry["family"] != family:
            continue
        if scale not in (vc.NOT_APPLICABLE,) and entry.get("scale") != scale:
            continue
        paths.append(entry["row_evidence_path"])
        hashes.append(entry["row_evidence_sha256"])
    return sorted(paths), [h for _, h in sorted(zip(paths, hashes))]


def _arm_rows(ctx) -> list:
    rows = []
    for a in ctx["reconstructed"]["arm_accuracies"]:
        family = a["family"]
        key = a["arm_id"]
        row = _blank_row()
        split_id = _split_for(family, a["n_questions"], a["answer_set_size"])
        split = SPLITS[split_id]
        allowed, forbidden = _boundaries("DESCRIPTIVE_ONLY")
        row.update({
            "evidence_id": vc.evidence_id("ARM", key),
            "evidence_class": "ARM",
            "experiment_family": family,
            "story_stage": _stage(family),
            "scientific_role": _role(family),
            "source_key": key,
            "quantity": "system accuracy on the development split, with an "
                        "image-clustered evaluation interval",
            "canonical_source_path": "results/closure/"
                                     "reconstructed_statistics.json",
            "canonical_source_sha256": ctx["hashes"].get(
                "results/closure/reconstructed_statistics.json",
                vc.NOT_AVAILABLE),
            "split": split_id,
            "dataset": split["dataset"],
            "training_scale": a["scale"],
            "model_system": key.split("/")[1],
            "seed_set": a["training_seeds"],
            "n_training_seeds": a["n_training_seeds"],
            "point_estimate": a["point_estimate"],
            "across_training_seed_sd_ddof1":
                a["across_training_seed_sd_ddof1"],
            "ci95": a["ci95_image_clustered"],
            "ci_type": a["interval_kind"],
            "cluster_unit": a["cluster_unit"],
            "n_resamples": a["n_resamples"],
            "rng_seed": a["rng_seed"],
            "n_questions": a["n_questions"],
            "n_unique_images": a["n_unique_images"],
            "answer_set_size": a["answer_set_size"],
            "metric_id": _metric_id(family),
            "scoring_implementation":
                pol.METRIC_DEFINITIONS[_metric_id(family)],
            "analysis_role": "SECONDARY",
            "comparison_class": "DESCRIPTIVE_ONLY",
            "evidence_readiness": "READY",
            "allowed_claim": allowed,
            "forbidden_stronger_claim": forbidden,
            "mandatory_limitation": _limitation(family, key),
            "excludes_zero": a["excludes_zero"],
            "directional": None,
        })
        row["row_level_evidence_paths"], row["row_level_sha256"] = \
            _row_evidence_for(ctx, family, a["scale"])
        ckpt, ckpt_sha = _checkpoints_for(ctx, family, key.split("/")[1],
                                          a["scale"])
        row["checkpoint_identity"], row["checkpoint_sha256"] = ckpt, ckpt_sha
        rows.append(row)
    return rows


def _checkpoints_for(ctx, family, arm, scale) -> tuple[list, list]:
    """Checkpoint identity comes from the closure's own record, which carries
    the path and the SHA-256 together, so no second lookup can disagree."""
    found = {}
    for entry in ctx["reconstructed_manifest"]["row_evidence"]:
        if (entry["family"] == family and entry["arm"] == arm
                and entry.get("scale") == scale
                and entry.get("image_condition") in (None, "normal")):
            checkpoint = entry["source_checkpoint"]
            found[checkpoint["path"]] = checkpoint["sha256"]
    paths = sorted(found)
    return paths, [found[p] for p in paths]


def _seed_rows(ctx) -> list:
    rows = []
    for s in ctx["seed"]["rows"]:
        family = s["family"]
        key = s["row_id"]
        row = _blank_row()
        split_id = _split_for(family, None, None)
        split = SPLITS[split_id]
        allowed, forbidden = _boundaries("DESCRIPTIVE_ONLY")
        readiness = ("READY_SEED_ONLY_NO_INTERVAL" if family == "v2_07"
                     else "READY")
        row.update({
            "evidence_id": vc.evidence_id("SEED", key),
            "evidence_class": "SEED",
            "experiment_family": family,
            "story_stage": _stage(family),
            "scientific_role": _role(family),
            "source_key": key,
            "quantity": "across-training-seed mean and sample standard "
                        "deviation (ddof=1). NOT a confidence interval and "
                        "NOT evaluation-sampling uncertainty",
            "canonical_source_path": s["source_artefact"],
            "canonical_source_sha256": ctx["hashes"].get(
                s["source_artefact"], vc.NOT_AVAILABLE),
            "split": split_id,
            "dataset": split["dataset"],
            "training_scale": s["scale"],
            "model_system": s["arm"],
            "seed_set": s["seed_set"],
            "n_training_seeds": s["n_training_seeds"],
            "point_estimate": s["mean"],
            "across_training_seed_sd_ddof1":
                s["sd_across_training_seeds_ddof1"],
            "ci95": None,
            "ci_type": "NO_INTERVAL: this row carries training-seed "
                       "dispersion only",
            "cluster_unit": vc.NOT_APPLICABLE,
            "n_questions": split["n_questions"],
            "n_unique_images": split["n_unique_images"],
            "answer_set_size": split["answer_set_size"],
            "metric_id": _metric_id(family),
            "scoring_implementation":
                pol.METRIC_DEFINITIONS.get(_metric_id(family),
                                           vc.NOT_AVAILABLE),
            "analysis_role": "SECONDARY",
            "comparison_class": "DESCRIPTIVE_ONLY",
            "evidence_readiness": readiness,
            "allowed_claim": allowed,
            "forbidden_stronger_claim": forbidden,
            "mandatory_limitation": _limitation(family, key),
            "excludes_zero": None,
            "directional": None,
        })
        _bind_seed_vector(ctx, row, s)
        if s["high_seed_dispersion"]:
            row["mandatory_limitation"] += (
                f" HIGH SEED DISPERSION: sd "
                f"{s['sd_across_training_seeds_ddof1']} across "
                f"{s['n_training_seeds']} seeds, range {s['range']}. This row "
                f"must always be shown with its per-seed spread; its mean "
                f"alone is misleading.")
        rows.append(row)
    return rows


def _deficit_rows(ctx) -> list:
    rows = []
    for d in ctx["reconstructed"]["pooled_deficits"]:
        family = d["family"]
        key = d["deficit_id"]
        row = _blank_row()
        split_id = _split_for(family, d["n_questions"], None)
        split = SPLITS[split_id]
        allowed, forbidden = _boundaries(d["comparison_status"])
        row.update({
            "evidence_id": vc.evidence_id("DEF", key),
            "evidence_class": "DEF",
            "experiment_family": family,
            "story_stage": "S06_TYPE_AND_DEPTH",
            "scientific_role": "SUPPORTING",
            "source_key": key,
            "quantity": "pooled four-or-more-step deficit against the fixed "
                        "v2_05b per-bucket priors",
            "canonical_source_path": "results/closure/"
                                     "reconstructed_statistics.json",
            "canonical_source_sha256": ctx["hashes"].get(
                "results/closure/reconstructed_statistics.json",
                vc.NOT_AVAILABLE),
            "split": split_id,
            "dataset": split["dataset"],
            "training_scale": d["scale"],
            "model_system": d["arm"],
            "seed_set": d["training_seeds"],
            "n_training_seeds": d["n_training_seeds"],
            "point_estimate": d["point_estimate"],
            "across_training_seed_sd_ddof1":
                d["across_training_seed_sd_ddof1"],
            "ci95": d["ci95_image_clustered"],
            "ci_type": d["interval_kind"],
            "cluster_unit": d["cluster_unit"],
            "n_resamples": d["n_resamples"],
            "rng_seed": d["rng_seed"],
            "n_questions": d["n_questions"],
            "n_unique_images": d["n_unique_images"],
            "answer_set_size": split["answer_set_size"],
            "metric_id": _metric_id(family),
            "scoring_implementation":
                pol.METRIC_DEFINITIONS[_metric_id(family)],
            "analysis_role": d["analysis_role"],
            "comparison_class": d["comparison_status"],
            "evidence_readiness": "READY",
            "allowed_claim": allowed,
            "forbidden_stronger_claim": forbidden,
            "mandatory_limitation": _limitation(family, key) + " " +
                                    d["interpretation_bound"],
            "excludes_zero": d["excludes_zero"],
            "directional": d["excludes_zero"],
        })
        rows.append(row)
    return rows


def _slice_rows(ctx) -> list:
    rows = []
    for s in ctx["slices"]["slices"]:
        family = s["family"]
        key = s["slice_id"]
        row = _blank_row()
        split_id = _split_for(family, None, None)
        split = SPLITS[split_id]
        allowed, forbidden = _boundaries(s["comparison_status"])
        row.update({
            "evidence_id": vc.evidence_id("SLC", key),
            "evidence_class": "SLC",
            "experiment_family": family,
            "story_stage": "S06_TYPE_AND_DEPTH",
            "scientific_role": ("SUPPORTING" if s["analysis_role"] != "DIAGNOSTIC"
                                else "DIAGNOSTIC"),
            "source_key": key,
            "quantity": f"{s['quantity']} on the {s['slice_kind']} slice "
                        f"{s['slice']}",
            "canonical_source_path": s["source"],
            "canonical_source_sha256": ctx["hashes"].get(s["source"],
                                                         vc.NOT_AVAILABLE),
            "split": split_id,
            "dataset": split["dataset"],
            "training_scale": s["scale"],
            "model_system": s["arm"],
            "seed_set": s["training_seeds"],
            "n_training_seeds": len(s["training_seeds"] or []),
            "point_estimate": s["point_estimate"],
            "across_training_seed_sd_ddof1":
                s["across_training_seed_sd_ddof1"],
            "ci95": s["ci95_image_clustered"],
            "ci_type": ("evaluation-sampling, image-clustered"
                        if s["ci95_image_clustered"] else "NO_INTERVAL"),
            "cluster_unit": s["cluster_unit"],
            "n_resamples": s["n_resamples"],
            "rng_seed": s["rng_seed"],
            "n_questions": s["n_questions"],
            "n_unique_images": s["n_unique_images"],
            "answer_set_size": split["answer_set_size"],
            "metric_id": _metric_id(family),
            "scoring_implementation": s.get("metric", vc.NOT_AVAILABLE),
            "analysis_role": s["analysis_role"],
            "comparison_class": s["comparison_status"],
            "evidence_readiness": ("READY" if s["ci95_image_clustered"]
                                   else "NO_INTERVAL"),
            "allowed_claim": allowed,
            "forbidden_stronger_claim": forbidden,
            "mandatory_limitation": _limitation(family, key) + (
                " Slices are unequal and dependent; every interval is "
                "clustered on the slice's own represented development "
                "images, never on rows."),
            "excludes_zero": s["excludes_zero"],
            "directional": s["excludes_zero"],
            "appendix_destination": "APPENDIX_SLICE_TABLES",
        })
        rows.append(row)
    return rows


def _efficiency_rows(ctx) -> list:
    rows = []
    for e in ctx["efficiency"]["rows"]:
        family = e["experiment_family"]
        key = e["row_key"]
        row = _blank_row()
        superseded = e["evidence_status"].startswith("SUPERSEDED")
        role = ("NOT_FOR_REPORTING" if superseded else _role(family))
        allowed, forbidden = _boundaries("DESCRIPTIVE_ONLY")
        if e["timing_kind"] == "END_TO_END_SERIAL":
            allowed = ("a measured end-to-end serial latency for this system "
                       "on this named node, at batch 1, warm median")
            forbidden = ("comparing it against a latency measured on another "
                         "node, merging the two node frontiers, or restating "
                         "it as an energy or power figure")
        elif e["timing_kind"] in ("CACHED_FEATURE_HEAD_ONLY",
                                  "CACHED_IMAGE_QUESTION_SIDE", "COMPONENT"):
            allowed = ("a partial or component cost, always labelled as such")
            forbidden = ("presenting it as an end-to-end query latency")
        elif e["timing_kind"] == "ADDITIVE_COMPONENT_SUM_SUPERSEDED":
            allowed = ("nothing; this row exists to record that the additive "
                       "sum is superseded")
            forbidden = ("any end-to-end latency claim, any Pareto front and "
                         "any amortisation claim built on the additive sums")
        row.update({
            "evidence_id": vc.evidence_id("EFF", key),
            "evidence_class": "EFF",
            "experiment_family": family,
            "story_stage": "S14_EFFICIENCY_TRADEOFF",
            "scientific_role": role,
            "source_key": key,
            "quantity": f"{e['timing_kind']} latency",
            "canonical_source_path": e["source_artefact"],
            "canonical_source_sha256": ctx["hashes"].get(e["source_artefact"],
                                                         vc.NOT_AVAILABLE),
            "split": "timing sample, not an accuracy split",
            "dataset": "timing sample, not an accuracy split",
            "training_scale": vc.NOT_APPLICABLE,
            "model_system": e["system_id"],
            "seed_set": vc.NOT_APPLICABLE,
            "point_estimate": e["warm_serial_median_ms"],
            "ci95": None,
            "ci_type": "NO_INTERVAL: warm median over repeated passes; the "
                       "across-pass spread is recorded in the source "
                       "artefact",
            # an efficiency row measures time, never accuracy, whatever
            # family it belongs to. Carrying the family's accuracy metric here
            # would let an accuracy scorer and a latency measurement share a
            # metric identity and therefore share an axis.
            "metric_id": ("warm_median_serial_latency_ms"
                          if e["timing_kind"] == "END_TO_END_SERIAL"
                          else "latency_and_memory_components"),
            "scoring_implementation": e["timing_methodology"],
            "analysis_role": "SECONDARY",
            "comparison_class": "DESCRIPTIVE_ONLY",
            "evidence_readiness": e["evidence_status"],
            "allowed_claim": allowed,
            "forbidden_stronger_claim": forbidden,
            "mandatory_limitation": _limitation(family, key) + (
                f" timing_kind={e['timing_kind']}; node={e['node']}; "
                f"precision={e['precision']}; batch_size={e['batch_size']}; "
                f"comparable_with_end_to_end="
                f"{str(e['comparable_with_end_to_end']).lower()}."),
        })
        row["timing_kind"] = e["timing_kind"]
        row["node"] = e["node"]
        row["precision"] = e["precision"]
        row["batch_size"] = e["batch_size"]
        row["comparable_with_end_to_end"] = e["comparable_with_end_to_end"]
        row["context_only"] = e["context_only"]
        row["accuracy_pairing"] = _accuracy_pairing(key)
        rows.append(row)
    return rows


def _accuracy_pairing(row_key: str) -> dict:
    """Bind the accuracy that may be plotted against this latency.

    The efficiency table carries no accuracy, by design: its 64-row timing
    sample supports no accuracy claim. So the vertical coordinate of any
    accuracy-against-latency figure has to come from the system's own frozen
    artefact, on the common raw-distribution denominator the closure fixed.
    Resolving the pointer here, and reading the value, is what stops VE-1
    pairing a latency with an accuracy measured on a different denominator.
    """
    spec = pol.EFFICIENCY_ACCURACY_PAIRING.get(row_key)
    if spec is None:
        return {"status": "NOT_APPLICABLE",
                "reason": "not an end-to-end serial row, so no "
                          "accuracy-against-latency point exists for it"}
    if spec["status"] != "RESOLVED":
        return {"status": spec["status"],
                "denominator": pol.EFFICIENCY_ACCURACY_DENOMINATOR,
                "question_ve1_must_answer": spec["question"],
                "may_be_plotted_with_a_vertical_coordinate": False}
    payload = vc.read_json(vc.PROJECT_ROOT / spec["artefact"])
    value = vc.json_pointer(payload, spec["pointer"])
    return {
        "status": "RESOLVED",
        "accuracy": value,
        "denominator": pol.EFFICIENCY_ACCURACY_DENOMINATOR,
        "artefact": spec["artefact"],
        "artefact_sha256": vc.sha256_file(vc.PROJECT_ROOT / spec["artefact"]),
        "pointer": spec["pointer"],
        "may_be_plotted_with_a_vertical_coordinate": True,
    }


REQUIRED_SCALAR_IDENTITY = (
    "metric_id", "comparison_class", "analysis_role", "quantity_kind",
    "ci_type", "counts_provenance",
)


def _descriptive_rows(ctx) -> list:
    """One row per canonical scalar, each carrying its OWN identity.

    An earlier revision gave the whole class one structural metric identity.
    That mislabelled three real development accuracies as counted quantities,
    which in turn defeated the mixed-metric guard on the figure consuming
    them, because structural counts are legitimately exempt from it. Identity
    is therefore declared per scalar and the builder refuses an entry that
    omits any part of it.
    """
    rows = []
    for spec in pol.DESCRIPTIVE_SCALARS:
        missing = [f for f in REQUIRED_SCALAR_IDENTITY if not spec.get(f)]
        if missing:
            raise AssertionError(
                f"descriptive scalar {spec['key']} does not declare "
                f"{missing}; a scalar without a declared identity cannot be "
                f"guarded by the metric-compatibility rule")
        if spec["metric_id"] not in pol.METRIC_DEFINITIONS:
            raise AssertionError(
                f"descriptive scalar {spec['key']} declares an unknown "
                f"metric_id: {spec['metric_id']}")

        path = vc.PROJECT_ROOT / spec["artefact"]
        payload = vc.read_json(path)
        value = vc.json_pointer(payload, spec["pointer"])
        allowed, forbidden = _boundaries(spec["comparison_class"])
        split_id = spec.get("split")
        split = SPLITS.get(split_id, {})
        row = _blank_row()
        row.update({
            "evidence_id": vc.evidence_id("DSC", spec["key"]),
            "evidence_class": "DSC",
            "experiment_family": spec["family"],
            "story_stage": spec["story_stage"],
            "scientific_role": spec["scientific_role"],
            "source_key": spec["key"],
            "quantity": spec["quantity"],
            "canonical_source_path": spec["artefact"],
            "canonical_source_sha256": vc.sha256_file(path),
            "split": split_id or vc.NOT_APPLICABLE,
            "dataset": split.get("dataset", vc.NOT_APPLICABLE),
            "training_scale": spec.get("training_scale") or vc.NOT_APPLICABLE,
            "model_system": spec.get("model_system") or _system_from_key(spec),
            "seed_set": spec.get("seed_set") or vc.NOT_APPLICABLE,
            "n_training_seeds": (len(spec["seed_set"]) if spec.get("seed_set")
                                 else None),
            "n_questions": spec.get("n_questions"),
            "n_unique_images": spec.get("n_unique_images"),
            "answer_set_size": split.get("answer_set_size"),
            "point_estimate": value,
            "ci95": None,
            "ci_type": spec["ci_type"],
            "metric_id": spec["metric_id"],
            "scoring_implementation":
                f"{pol.METRIC_DEFINITIONS[spec['metric_id']]} Read from "
                f"{spec['artefact']} at pointer {spec['pointer']}.",
            "analysis_role": spec["analysis_role"],
            "comparison_class": spec["comparison_class"],
            "evidence_readiness": ("READY" if spec["metric_id"]
                                   == "structural_count_or_share"
                                   else "READY_SEED_ONLY_NO_INTERVAL"),
            "allowed_claim": allowed,
            "forbidden_stronger_claim": forbidden,
            "mandatory_limitation": _limitation(spec["family"], spec["key"]),
        })
        if spec["quantity_kind"] == "TRAINING_SEED_DISPERSION":
            row["across_training_seed_sd_ddof1"] = value
        row["source_pointer"] = spec["pointer"]
        row["quantity_kind"] = spec["quantity_kind"]
        row["counts_provenance"] = spec["counts_provenance"]
        if spec.get("comparison_class_reason"):
            row["comparison_class_reason"] = spec["comparison_class_reason"]
        rows.append(row)
    return rows


def _system_from_key(spec: dict) -> str:
    """The system a scalar describes, where the key names one."""
    key = spec["key"]
    if key.startswith("v3_01.reasoner_minus_fusion"):
        return "reasoner minus fusion"
    if key.startswith("v3_01.reasoner"):
        return "reasoner"
    return vc.NOT_APPLICABLE


def _supersession_rows(ctx) -> list:
    """Explicit rows for evidence that must NOT be reported.

    These exist so a later reader can look up a familiar old number and be
    told, by the same registry that holds the good numbers, that it is not
    usable and why.
    """
    rows = []
    for entry in pol.SUPERSESSION_ENTRIES:
        if entry["status"] in ("CURRENT_CANONICAL",):
            continue
        key = entry["family"] + "." + vc.slug(entry["artefact_group"])[:60]
        role = {"SUPERSEDED": "SUPERSEDED",
                "NOT_FOR_REPORTING": "NOT_FOR_REPORTING",
                "HISTORICAL_ONLY": "APPENDIX",
                "PARTLY_SUPERSEDED": "SUPPORTING"}[entry["status"]]
        row = _blank_row()
        row.update({
            "evidence_id": vc.evidence_id("SUP", key),
            "evidence_class": "SUP",
            "experiment_family": entry["family"],
            "story_stage": _stage(entry["family"]),
            "scientific_role": role,
            "source_key": entry["artefact_group"],
            "quantity": "supersession marker; carries no reportable value",
            "canonical_source_path": (entry["paths"][0] if entry["paths"]
                                      else vc.NOT_APPLICABLE),
            "canonical_source_sha256": (
                ctx["hashes"].get(entry["paths"][0], vc.NOT_AVAILABLE)
                if entry["paths"] else vc.NOT_APPLICABLE),
            "analysis_role": "DIAGNOSTIC",
            "comparison_class": "DESCRIPTIVE_ONLY",
            "evidence_readiness": entry["status"],
            "allowed_claim": ("describing the artefact's history and why it "
                              "is not used"),
            "forbidden_stronger_claim": ("quoting any invalid field of this "
                                         "artefact as a result"),
            "mandatory_limitation": entry["reason"],
        })
        row["supersession_status"] = entry["status"]
        row["valid_fields"] = entry["valid_fields"]
        row["invalid_fields"] = entry["invalid_fields"]
        row["superseded_by"] = entry["superseded_by"]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------

def build(ctx) -> dict:
    rows = []
    rows += _contrast_rows(ctx)
    rows += _arm_rows(ctx)
    rows += _seed_rows(ctx)
    rows += _deficit_rows(ctx)
    rows += _slice_rows(ctx)
    rows += _efficiency_rows(ctx)
    rows += _descriptive_rows(ctx)
    rows += _supersession_rows(ctx)
    rows.sort(key=lambda r: r["evidence_id"])

    for row in rows:
        family = row["experiment_family"] or ""
        entry = (pol.FAMILY_CHECKPOINT_SELECTION.get(family)
                 or pol.FAMILY_CHECKPOINT_SELECTION.get(family.upper()))
        if entry is None:
            row["checkpoint_selection"] = vc.NOT_APPLICABLE
            row["checkpoint_selection_class"] = vc.NOT_APPLICABLE
            continue
        row["checkpoint_selection"] = entry["detail"]
        row["checkpoint_selection_class"] = entry["class"]

    seen = set()
    for row in rows:
        if row["evidence_id"] in seen:
            raise AssertionError(
                f"duplicate evidence_id: {row['evidence_id']}")
        seen.add(row["evidence_id"])

    return {
        "title": "canonical evidence inventory",
        "ve0_output": "A",
        "row_count": len(rows),
        "evidence_classes": vc.EVIDENCE_CLASSES,
        "scientific_roles": pol.SCIENTIFIC_ROLES,
        "story_stages": pol.STORY_STAGES,
        "main_sequence": pol.MAIN_SEQUENCE,
        "comparison_classes": pol.COMPARISON_CLASSES,
        "analysis_roles": pol.ANALYSIS_ROLES,
        "metric_definitions": pol.METRIC_DEFINITIONS,
        "mixed_metric_rule": pol.MIXED_METRIC_RULE,
        "checkpoint_selection_by_family": pol.FAMILY_CHECKPOINT_SELECTION,
        "cross_family_selection_caveat": pol.CROSS_FAMILY_SELECTION_CAVEAT,
        "splits": SPLITS,
        "universal_limitation": pol.UNIVERSAL_LIMITATION,
        "rule": ("a number may appear in the dissertation, in a table, in a "
                 "figure or on a slide only if it resolves to a row here. A "
                 "row whose scientific_role is SUPERSEDED or "
                 "NOT_FOR_REPORTING may be described but never reported as a "
                 "result."),
        "clean_test_accessed": False,
        "rows": rows,
    }


INVENTORY_CSV_COLUMNS = [
    "evidence_id", "evidence_class", "experiment_family", "story_stage",
    "scientific_role", "source_key", "quantity", "canonical_source_path",
    "canonical_source_sha256", "split", "training_scale", "model_system",
    "seed_set", "n_training_seeds", "point_estimate",
    "across_training_seed_sd_ddof1", "ci95", "ci_type", "cluster_unit",
    "n_questions", "n_unique_images", "answer_set_size", "metric_id",
    "analysis_role", "comparison_class", "evidence_readiness",
    "checkpoint_selection", "excludes_zero",
    # per-seed reporting, added by the 2026-08-13 amendment so the flat view
    # carries the same bindings as the JSON
    "per_seed_status", "per_seed_seed_ids", "per_seed_values",
    "per_seed_range", "per_seed_effects", "per_seed_effect_range",
    "proposed_figure_use", "proposed_table_use",
    "proposed_dissertation_section", "proposed_supervisor_slide",
    "appendix_destination", "mandatory_limitation",
]
