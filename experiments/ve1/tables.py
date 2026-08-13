"""The VE-1 table builders, one function per VE-0 table specification.

Every cell is rendered at the precision the specification names, from a value
`Evidence` read out of the frozen inventory. A cell whose value VE-0 does not
bind is never blank and never a dash that could be read as zero: it carries an
explicit marker and the deficiency is published in the VE-1 manifest.

Two conventions the reporting style contract makes non-optional are enforced
here rather than left to the author. No value is ever bolded, because every
bolding rule in the specifications is "no bolding" or "no bolding by outcome".
And a missing interval always renders as the words "not available" with the
reason recorded in the row's own record, never as an empty cell.
"""

from __future__ import annotations

import re

from experiments.ve1 import output
from experiments.ve1 import ve1_common as vc
from experiments.ve1.ve1_common import Contract, Evidence, Recorder

BUILDER = "experiments/ve1/tables.py"

INTERFACE_SIDE = {
    "A0p": "question side (CLIP reference)",
    "A1": "question side", "A1r": "question side",
    "B1": "answer side (no language model)",
    "B2": "answer side", "B3": "answer side",
    "B4": "answer side", "B4r": "answer side",
}
LANGUAGE_MODEL = {
    "A0p": "none (frozen CLIP text tower)",
    "A1": "SmolLM2-135M, pretrained", "A1r": "SmolLM2-135M, random",
    "B1": "none (lightweight classifier readout)",
    "B2": "SmolLM2-135M, random", "B3": "SmolLM2-135M, pretrained",
    "B4": "SmolLM2-360M, pretrained", "B4r": "SmolLM2-360M, random",
}
PRETRAINED_OR_RANDOM = {
    "A0p": "not applicable", "A1": "pretrained", "A1r": "random",
    "B1": "not applicable", "B2": "random", "B3": "pretrained",
    "B4": "pretrained", "B4r": "random",
}
CONTRAST_SIDE = {
    "EV-CON-E8A.A1_minus_A1r.train_40k": ("question side", "135M"),
    "EV-CON-E8A.A1_minus_A1r.train_250k": ("question side", "135M"),
    "EV-CON-E8B.B3_-_B2.train_40k": ("answer side", "135M"),
    "EV-CON-E8B.B3_-_B2.train_250k": ("answer side", "135M"),
    "EV-CON-E10.pretraining_effect.train_40k": ("answer side", "360M"),
    "EV-CON-E10.pretraining_effect.train_250k": ("answer side", "360M"),
    "EV-CON-E10.difference_in_differences": ("answer side", "360M"),
}
REFERENCE_CONDITION = {
    "EV-CON-E8A.A1_minus_A1r.train_40k": "A1r, architecture-matched random "
                                         "SmolLM2-135M",
    "EV-CON-E8A.A1_minus_A1r.train_250k": "A1r, architecture-matched random "
                                          "SmolLM2-135M",
    "EV-CON-E8B.B3_-_B2.train_40k": "B2, architecture-matched random "
                                    "SmolLM2-135M",
    "EV-CON-E8B.B3_-_B2.train_250k": "B2, architecture-matched random "
                                     "SmolLM2-135M",
    "EV-CON-E10.pretraining_effect.train_40k": "B4r, architecture-matched "
                                               "random SmolLM2-360M",
    "EV-CON-E10.pretraining_effect.train_250k": "B4r, architecture-matched "
                                                "random SmolLM2-360M",
    "EV-CON-E10.difference_in_differences": "the same effect at train_40k "
                                            "(SECOND-ORDER)",
}


def _payload(contract: Contract, evidence: Evidence, rows: list,
             columns: list, caveat: str, notes: dict | None = None) -> dict:
    spec = evidence.spec
    payload = {
        "artefact_id": spec["table_id"].replace("VE0-", "VE1-"),
        "ve0_specification_id": spec["table_id"],
        "working_title": spec["working_title"],
        "scientific_question": spec["scientific_question"],
        "placement": spec["placement"],
        "dissertation_section": spec["dissertation_section"],
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "precision": spec["precision"],
        "uncertainty_notation": spec["uncertainty_notation"],
        "bolding_rule": spec["bolding_rule"],
        "highlight_rule": spec["highlight_rule"],
        "footnotes": spec["footnotes"],
        "mandatory_caveat": caveat,
        "metric_ids": spec["metric_ids"],
        "carries_v2_07_limitation": spec["carries_v2_07_limitation"],
        "mixes_checkpoint_selection": spec["mixes_checkpoint_selection"],
        "mixed_metric_justification": spec.get("mixed_metric_justification"),
        "consumed_evidence_ids": evidence.consumed(),
        "unbound_cells": evidence.unbound_cells,
        "ve0_source_artefacts": contract.source_inputs(),
        "development_set_only": True,
        "clean_test_accessed": False,
        "any_value_bolded": False,
    }
    if notes:
        payload.update(notes)
    payload["provenance"] = vc.provenance(BUILDER,
                                          inputs=evidence.source_paths())
    payload["content_sha256"] = vc.content_digest(payload)
    return payload


def _emit(contract, evidence, recorder, rows, columns, caveat, notes=None,
          out_dir=None):
    payload = _payload(contract, evidence, rows, columns, caveat, notes)
    outputs = output.save_table(payload["artefact_id"], rows, columns,
                                payload, out_dir)
    recorder.record(payload["artefact_id"], evidence.spec_id, "table",
                    evidence, outputs, BUILDER)
    return payload


def _limitation_caveat(contract: Contract, evidence: Evidence) -> str:
    """The mandatory limitation the consumed rows themselves carry.

    Built from the evidence rather than authored, so a table cannot state a
    weaker boundary than its own evidence requires. Deduplicated sentence by
    sentence, because most rows repeat the same development-set preamble and a
    caveat a reader stops reading is a caveat that does not work.
    """
    seen, sentences = set(), []
    for evidence_id in evidence.consumed():
        text = contract.rows[evidence_id].get("mandatory_limitation") or ""
        for sentence in re.split(r"(?<=\.)\s+", text.strip()):
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                seen.add(sentence)
                sentences.append(sentence)
    return " ".join(sentences)


# --------------------------------------------------------------------------
# VE0-TAB-01: main experimental progression
# --------------------------------------------------------------------------

TAB01_COLUMNS = ["system", "trainable parameters", "training scale", "seeds",
                 "mean accuracy", "sd (training seeds, ddof=1)",
                 "95% CI (image-clustered)", "interval available",
                 "evidence id"]

PARAMETER_KEY = {"concat": "trainable_parameters.concat",
                 "fusion": "trainable_parameters.fusion"}
ORDER_40K = ["question_only", "image_only", "concat", "concat_wide",
             "fusion_narrow", "fusion"]
ORDER_SCALE = ["question_only", "concat", "product_576k", "fusion"]


def build_tab_01(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-TAB-01")
    rows = []

    def parameters(system: str) -> str:
        key = PARAMETER_KEY.get(system)
        if key:
            return f"{recorder.use_scalar(key):,}"
        if system in ("concat_wide", "fusion_narrow", "product_576k"):
            return "matched budget (see VE1-TAB-02 note 3)"
        return evidence.note_unbound(
            "trainable parameters",
            f"VE-0 binds trainable parameter counts for concat and fusion "
            f"only, in the VE0-TAB-02 footnote and the VE0-FIG-02 caveat. It "
            f"binds no count for {system}, and VE-1 does not read one from a "
            f"historical experiment directory.")

    for system in ORDER_40K:
        family = "v2_03" if system in ("concat_wide", "fusion_narrow") \
            else "v2_02"
        evidence_id = f"EV-ARM-{family}.{system}.train_40k"
        row = evidence.row(evidence_id)
        rows.append({
            "system": vc.system_label(system),
            "trainable parameters": parameters(system),
            "training scale": vc.scale_label(row["training_scale"]),
            "seeds": vc.fmt_seeds(row["seed_set"]),
            "mean accuracy": vc.fmt(row["point_estimate"]),
            "sd (training seeds, ddof=1)":
                vc.fmt(row["across_training_seed_sd_ddof1"]),
            "95% CI (image-clustered)": vc.fmt_ci(row["ci95"]),
            "interval available": "yes",
            "evidence id": evidence_id,
        })

    for system in ORDER_SCALE:
        for scale in ("train_40k", "train_100k", "train_250k"):
            evidence_id = f"EV-SEED-v2_07.{system}.{scale}"
            row = evidence.row(evidence_id)
            rows.append({
                "system": vc.system_label(system) + " (v2_07 scaling)",
                "trainable parameters": parameters(system),
                "training scale": vc.scale_label(scale),
                "seeds": vc.fmt_seeds(row["seed_set"]),
                "mean accuracy": vc.fmt(row["point_estimate"]),
                "sd (training seeds, ddof=1)":
                    vc.fmt(row["across_training_seed_sd_ddof1"]),
                "95% CI (image-clustered)": vc.NOT_AVAILABLE,
                "interval available":
                    "no: v2_07 ACCEPTED DOCUMENTED LIMITATION",
                "evidence id": evidence_id,
            })

    evidence.assert_complete()
    return _emit(contract, evidence, recorder, rows, TAB01_COLUMNS,
                 _limitation_caveat(contract, evidence), out_dir=out_dir)


# --------------------------------------------------------------------------
# VE0-TAB-02: capacity-matched controls and interaction features
# --------------------------------------------------------------------------

CONTRAST_COLUMNS = ["contrast", "comparison class", "effect",
                    "95% CI (image-clustered)", "excludes zero",
                    "sd (training seeds, ddof=1)", "n questions", "n images",
                    "evidence id"]


def _contrast_row(evidence: Evidence, evidence_id: str) -> dict:
    row = evidence.row(evidence_id)
    key = str(row["source_key"]).split("/")
    name = key[1] if len(key) > 1 else key[0]
    return {
        "contrast": name.replace("_minus_", " minus ").replace("_", " "),
        "comparison class": row["comparison_class"],
        "effect": vc.fmt(row["point_estimate"]),
        "95% CI (image-clustered)": vc.fmt_ci(row["ci95"]),
        "excludes zero": vc.fmt(row["excludes_zero"]),
        "sd (training seeds, ddof=1)":
            vc.fmt(row["across_training_seed_sd_ddof1"]),
        "n questions": row["n_questions"],
        "n images": row["n_unique_images"],
        "evidence id": evidence_id,
    }


def build_tab_02(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-TAB-02")
    rows = [_contrast_row(evidence, e) for e in sorted(evidence.bound)]
    evidence.assert_complete()
    return _emit(contract, evidence, recorder, rows, CONTRAST_COLUMNS,
                 _limitation_caveat(contract, evidence), out_dir=out_dir)


# --------------------------------------------------------------------------
# VE0-TAB-03: training-scale effects, frozen-SLM families
# --------------------------------------------------------------------------

TAB03_COLUMNS = ["system", "language model", "effect (250k minus 40k)",
                 "95% CI (image-clustered)", "excludes zero",
                 "sd (training seeds, ddof=1)", "evidence id"]


def build_tab_03(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-TAB-03")
    rows = []
    for evidence_id in sorted(evidence.bound):
        row = evidence.row(evidence_id)
        arm = str(row["source_key"]).split("/")[-1].replace("scale_effect_",
                                                            "")
        rows.append({
            "system": f"{arm} ({INTERFACE_SIDE.get(arm, 'not applicable')})",
            "language model": LANGUAGE_MODEL.get(arm, "not recorded"),
            "effect (250k minus 40k)": vc.fmt(row["point_estimate"]),
            "95% CI (image-clustered)": vc.fmt_ci(row["ci95"]),
            "excludes zero": vc.fmt(row["excludes_zero"]),
            "sd (training seeds, ddof=1)":
                vc.fmt(row["across_training_seed_sd_ddof1"]),
            "evidence id": evidence_id,
        })
    evidence.assert_complete()
    return _emit(contract, evidence, recorder, rows, TAB03_COLUMNS,
                 _limitation_caveat(contract, evidence), out_dir=out_dir)


# --------------------------------------------------------------------------
# VE0-TAB-04: frozen-SLM interface comparison
# --------------------------------------------------------------------------

TAB04_COLUMNS = ["arm", "interface side", "language model",
                 "pretrained or random", "training scale", "seeds",
                 "mean accuracy", "sd (training seeds, ddof=1)",
                 "per-seed values", "high dispersion", "evidence id"]


def build_tab_04(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-TAB-04")
    rows = []
    for evidence_id in sorted(evidence.bound):
        row = evidence.row(evidence_id)
        arm = row["model_system"]
        dispersion = "HIGH SEED DISPERSION" in str(row["mandatory_limitation"])
        rows.append({
            "arm": arm,
            "interface side": INTERFACE_SIDE.get(arm, "not recorded"),
            "language model": LANGUAGE_MODEL.get(arm, "not recorded"),
            "pretrained or random": PRETRAINED_OR_RANDOM.get(arm,
                                                             "not recorded"),
            "training scale": vc.scale_label(row["training_scale"]),
            "seeds": vc.fmt_seeds(row["seed_set"]),
            "mean accuracy": vc.fmt(row["point_estimate"]),
            "sd (training seeds, ddof=1)":
                vc.fmt(row["across_training_seed_sd_ddof1"]),
            "per-seed values": evidence.note_unbound(
                "per-seed values",
                "the VE0-TAB-04 specification requires a per-seed column and "
                "calls it mandatory, but no VE-0 inventory row binds per-seed "
                "accuracies: a SEED row carries the across-seed mean, the "
                "sample standard deviation and the seed set only. VE-1 does "
                "not recover them from a historical experiment directory.",
                [evidence_id]),
            "high dispersion": "yes (dagger)" if dispersion else "no",
            "evidence id": evidence_id,
        })
    evidence.assert_complete()
    return _emit(contract, evidence, recorder, rows, TAB04_COLUMNS,
                 _limitation_caveat(contract, evidence), out_dir=out_dir)


# --------------------------------------------------------------------------
# VE0-TAB-05: primary pretraining contrasts and the difference in differences
# --------------------------------------------------------------------------

TAB05_COLUMNS = ["contrast", "order", "interface side",
                 "language model size", "reference condition",
                 "training scale", "effect", "95% CI (image-clustered)",
                 "excludes zero", "per-seed effects", "comparison class",
                 "evidence id"]

TAB05_ORDER = [
    "EV-CON-E8A.A1_minus_A1r.train_40k",
    "EV-CON-E8A.A1_minus_A1r.train_250k",
    "EV-CON-E8B.B3_-_B2.train_40k",
    "EV-CON-E8B.B3_-_B2.train_250k",
    "EV-CON-E10.pretraining_effect.train_40k",
    "EV-CON-E10.pretraining_effect.train_250k",
    "EV-CON-E10.difference_in_differences",
]


def build_tab_05(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-TAB-05")
    rows = []
    for evidence_id in TAB05_ORDER:
        row = evidence.row(evidence_id)
        side, size = CONTRAST_SIDE[evidence_id]
        second_order = evidence_id.endswith("difference_in_differences")
        rows.append({
            "contrast": ("(pretrained minus random) at 250k minus the same "
                         "at 40k" if second_order else
                         str(row["source_key"]).split("/")[1]
                         .replace("_", " ")),
            "order": "SECOND_ORDER interaction" if second_order
                     else "FIRST_ORDER",
            "interface side": side,
            "language model size": size,
            "reference condition": REFERENCE_CONDITION[evidence_id],
            "training scale": ("250k against 40k" if second_order
                               else vc.scale_label(row["training_scale"])),
            "effect": vc.fmt(row["point_estimate"]),
            "95% CI (image-clustered)": vc.fmt_ci(row["ci95"]),
            "excludes zero": vc.fmt(row["excludes_zero"]),
            "per-seed effects": evidence.note_unbound(
                "per-seed effects",
                "the VE0-TAB-05 specification asks for per-seed effects so a "
                "reader can see where seeds disagree in sign. No VE-0 "
                "inventory row binds per-seed contrast values; the seed "
                "disagreement is recorded in prose in the row's mandatory "
                "limitation and is reproduced in this table's caveat.",
                [evidence_id]),
            "comparison class": row["comparison_class"],
            "evidence id": evidence_id,
        })
    evidence.assert_complete()
    return _emit(contract, evidence, recorder, rows, TAB05_COLUMNS,
                 _limitation_caveat(contract, evidence), out_dir=out_dir)


# --------------------------------------------------------------------------
# VE0-TAB-06: measured efficiency, by node and timing class
# --------------------------------------------------------------------------

TAB06_COLUMNS = ["system", "experiment family", "timing class", "node",
                 "precision", "batch size",
                 "warm median serial latency (ms)",
                 "comparable with end-to-end", "evidence status",
                 "accuracy pairing", "evidence id"]


def build_tab_06(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-TAB-06")
    rows = []
    for evidence_id in sorted(
            evidence.bound,
            key=lambda e: (str(contract.rows[e]["node"]),
                           str(contract.rows[e]["timing_kind"]), e)):
        row = evidence.row(evidence_id)
        superseded = row["timing_kind"] == "ADDITIVE_COMPONENT_SUM_SUPERSEDED"
        value = row["point_estimate"]
        if value is None:
            latency = evidence.note_unbound(
                "warm median serial latency (ms)",
                "VE-0 binds no latency value for this row: its point estimate "
                "is null in the canonical evidence inventory. For the E7a "
                "additive row this is deliberate, so the row cannot be read "
                "as a measurement; for the cached and component rows it is a "
                "gap in the evidence contract, recorded here and not filled "
                "from a historical experiment directory."
                if not superseded else
                "deliberately carries no latency value, so a superseded "
                "additive sum cannot be read as a measurement or ranked "
                "against one.",
                [evidence_id])
        else:
            latency = vc.fmt(value, 4)
        pairing = row["accuracy_pairing"]
        rows.append({
            "system": row["model_system"],
            "experiment family": row["experiment_family"],
            "timing class": row["timing_kind"],
            "node": row["node"],
            "precision": row["precision"],
            "batch size": row["batch_size"],
            "warm median serial latency (ms)": latency,
            "comparable with end-to-end":
                vc.fmt(row["comparable_with_end_to_end"]),
            "evidence status": ("SUPERSEDED_FOR_END_TO_END_CLAIMS"
                                if superseded else row["evidence_readiness"]),
            "accuracy pairing": pairing.get("status"),
            "evidence id": evidence_id,
        })
    evidence.assert_complete()
    return _emit(contract, evidence, recorder, rows, TAB06_COLUMNS,
                 _limitation_caveat(contract, evidence), out_dir=out_dir,
                 notes={"nodes_merged": False,
                        "energy_or_power_measured": False})


# --------------------------------------------------------------------------
# VE0-TAB-07: the claim, evidence and limitation ledger
# --------------------------------------------------------------------------

TAB07_COLUMNS = ["claim id", "claim as stated", "status",
                 "positive results claim", "evidence ids", "mandatory caveat",
                 "forbidden stronger version"]


def build_tab_07(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-TAB-07")
    rows = []
    for claim_id in sorted(contract.claims):
        claim = contract.claims[claim_id]
        rows.append({
            "claim id": claim_id,
            "claim as stated": claim["claim_text"],
            "status": claim["status"],
            "positive results claim": vc.fmt(claim["is_positive_results_"
                                                   "claim"]),
            "evidence ids": (" ".join(claim["evidence_ids"])
                             if claim["evidence_ids"] else
                             "none bound: this row records a prohibition "
                             "rather than a positive finding"
                             if claim["status"] == "NOT_SUPPORTED" else
                             "none bound in the VE-0 claim ledger"),
            "mandatory caveat": claim["mandatory_caveat"],
            "forbidden stronger version": claim["forbidden_stronger_version"],
        })
    evidence.assert_complete()
    caveat = ("Generated from the frozen VE-0 claim ledger. The Abstract, "
              "Discussion and Conclusion may not state a stronger version of "
              "any row here. NOT_SUPPORTED rows are prohibitions and are kept "
              "in the table rather than deleted from it. Every row is a "
              "development-set result: the clean test is embargoed and unread "
              "and no held-out generalisation is claimed.")
    return _emit(contract, evidence, recorder, rows, TAB07_COLUMNS, caveat,
                 out_dir=out_dir,
                 notes={"consumes_claim_ledger": True,
                        "claim_count": len(rows),
                        "not_supported_rows_retained": sorted(
                            c for c in contract.claims
                            if contract.claims[c]["status"]
                            == "NOT_SUPPORTED")})


# --------------------------------------------------------------------------
# VE0-TAB-A1: full slice statistics
# --------------------------------------------------------------------------

TABA1_COLUMNS = ["slice id", "family", "system", "scale", "slice kind",
                 "slice", "quantity", "metric", "point estimate",
                 "95% CI (image-clustered)", "n questions", "n images",
                 "counts provenance", "evidence id"]


def build_tab_a1(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-TAB-A1")
    rows = []
    for evidence_id in sorted(evidence.bound):
        row = evidence.row(evidence_id)
        parts = str(row["source_key"]).split("/")
        slice_name = parts[-2] if len(parts) >= 2 else "not recorded"
        kind = ("program length" if slice_name.startswith("steps")
                else "structural" if slice_name.startswith("structural")
                else "semantic" if slice_name.startswith("semantic")
                else "other")
        rows.append({
            "slice id": row["source_key"],
            "family": row["experiment_family"],
            "system": row["model_system"],
            "scale": vc.scale_label(row["training_scale"]),
            "slice kind": kind,
            "slice": slice_name,
            "quantity": row["quantity"],
            "metric": row["metric_id"],
            "point estimate": vc.fmt(row["point_estimate"]),
            "95% CI (image-clustered)": vc.fmt_ci(row["ci95"]),
            "n questions": row["n_questions"],
            "n images": (row["n_unique_images"] if row["n_unique_images"]
                         is not None else vc.NOT_AVAILABLE),
            "counts provenance": row.get("counts_provenance",
                                         "NOT_APPLICABLE"),
            "evidence id": evidence_id,
        })
    evidence.assert_complete()
    return _emit(contract, evidence, recorder, rows, TABA1_COLUMNS,
                 _limitation_caveat(contract, evidence), out_dir=out_dir)


# --------------------------------------------------------------------------
# VE0-TAB-A2: full training-seed variability
# --------------------------------------------------------------------------

TABA2_COLUMNS = ["row id", "family", "system", "scale", "metric", "seed set",
                 "per-seed accuracies", "mean",
                 "sd (training seeds, ddof=1)", "range", "high dispersion",
                 "checkpoint selection", "evidence id"]


def build_tab_a2(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-TAB-A2")
    rows = []
    for evidence_id in sorted(evidence.bound):
        row = evidence.row(evidence_id)
        dispersion = "HIGH SEED DISPERSION" in str(row["mandatory_limitation"])
        rows.append({
            "row id": row["source_key"],
            "family": row["experiment_family"],
            "system": row["model_system"],
            "scale": vc.scale_label(row["training_scale"]),
            "metric": row["metric_id"],
            "seed set": vc.fmt_seeds(row["seed_set"]),
            "per-seed accuracies": evidence.note_unbound(
                "per-seed accuracies",
                "no VE-0 inventory row binds per-seed accuracies. A SEED row "
                "carries the across-seed mean, the sample standard deviation "
                "and the seed set only, so the question this table asks "
                "cannot be answered from the frozen evidence contract as it "
                "stands.", [evidence_id]),
            "mean": vc.fmt(row["point_estimate"]),
            "sd (training seeds, ddof=1)":
                vc.fmt(row["across_training_seed_sd_ddof1"]),
            "range": evidence.note_unbound(
                "range",
                "the range is derivable only from per-seed values, which VE-0 "
                "does not bind. Two rows state their range inside their "
                "mandatory limitation and those statements are preserved in "
                "the caveat.", [evidence_id]),
            "high dispersion": "yes" if dispersion else "no",
            "checkpoint selection": row["checkpoint_selection_class"],
            "evidence id": evidence_id,
        })
    evidence.assert_complete()
    return _emit(contract, evidence, recorder, rows, TABA2_COLUMNS,
                 _limitation_caveat(contract, evidence), out_dir=out_dir)


# --------------------------------------------------------------------------
# VE0-TAB-A3: SigLIP against CLIP at both scales
# --------------------------------------------------------------------------

TABA3_COLUMNS = ["encoder", "system", "scale", "seeds", "mean accuracy",
                 "sd (training seeds, ddof=1)",
                 "95% CI (image-clustered)", "interval available",
                 "evidence id"]


def build_tab_a3(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-TAB-A3")
    rows = []
    for evidence_id in sorted(evidence.bound):
        row = evidence.row(evidence_id)
        family = row["experiment_family"]
        encoder = ("SigLIP-B/16" if family in ("e2", "E2")
                   else "CLIP ViT-B/32")
        has_ci = row["ci95"] is not None
        rows.append({
            "encoder": encoder,
            "system": vc.system_label(row["model_system"]),
            "scale": vc.scale_label(row["training_scale"]),
            "seeds": vc.fmt_seeds(row["seed_set"]),
            "mean accuracy": vc.fmt(row["point_estimate"]),
            "sd (training seeds, ddof=1)":
                vc.fmt(row["across_training_seed_sd_ddof1"]),
            "95% CI (image-clustered)": vc.fmt_ci(row["ci95"]),
            "interval available": ("yes" if has_ci else
                                   "no: v2_07 ACCEPTED DOCUMENTED LIMITATION"
                                   if family == "v2_07" else
                                   "no: this row carries seed dispersion "
                                   "only"),
            "evidence id": evidence_id,
        })
    evidence.assert_complete()
    return _emit(contract, evidence, recorder, rows, TABA3_COLUMNS,
                 _limitation_caveat(contract, evidence), out_dir=out_dir)


# --------------------------------------------------------------------------
# VE0-TAB-A4: the complete contrast registry
# --------------------------------------------------------------------------

TABA4_COLUMNS = ["contrast id", "family", "metric", "analysis role",
                 "comparison class", "effect", "95% CI (image-clustered)",
                 "excludes zero", "n questions", "n images",
                 "training seeds", "checkpoint selection", "evidence id"]


def build_tab_a4(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-TAB-A4")
    rows = []
    for evidence_id in sorted(evidence.bound):
        row = evidence.row(evidence_id)
        rows.append({
            "contrast id": row["source_key"],
            "family": row["experiment_family"],
            "metric": row["metric_id"],
            "analysis role": row["analysis_role"],
            "comparison class": row["comparison_class"],
            "effect": vc.fmt(row["point_estimate"]),
            "95% CI (image-clustered)": vc.fmt_ci(row["ci95"]),
            "excludes zero": vc.fmt(row["excludes_zero"]),
            "n questions": row["n_questions"],
            "n images": (row["n_unique_images"] if row["n_unique_images"]
                         is not None else vc.NOT_AVAILABLE),
            "training seeds": row["n_training_seeds"],
            "checkpoint selection": row["checkpoint_selection_class"],
            "evidence id": evidence_id,
        })
    evidence.assert_complete()
    return _emit(contract, evidence, recorder, rows, TABA4_COLUMNS,
                 _limitation_caveat(contract, evidence), out_dir=out_dir,
                 notes={"intervals_containing_zero": sum(
                     1 for e in evidence.consumed()
                     if contract.rows[e]["excludes_zero"] is False)})


BUILDERS = [
    ("VE0-TAB-01", build_tab_01),
    ("VE0-TAB-02", build_tab_02),
    ("VE0-TAB-03", build_tab_03),
    ("VE0-TAB-04", build_tab_04),
    ("VE0-TAB-05", build_tab_05),
    ("VE0-TAB-06", build_tab_06),
    ("VE0-TAB-07", build_tab_07),
    ("VE0-TAB-A1", build_tab_a1),
    ("VE0-TAB-A2", build_tab_a2),
    ("VE0-TAB-A3", build_tab_a3),
    ("VE0-TAB-A4", build_tab_a4),
]
