"""The remaining VE-0 outputs: supersession map, claim ledger, research
question matrix, qualitative protocol and record schema, and the four frozen
contracts.

Each builder validates what it can prove locally and refuses rather than
emitting something a later reader would have to trust. A claim whose evidence
does not resolve, a research question naming an unknown claim, a qualitative
category whose prediction source is undeclared and a contract naming an
unknown section are all hard failures here, not test findings later.
"""

from __future__ import annotations

from . import policy_claims as pcl
from . import policy_contracts as con
from . import policy_evidence as pol
from . import ve0_common as vc


# --------------------------------------------------------------------------
# Output C: supersession map
# --------------------------------------------------------------------------

def build_supersession(ctx) -> dict:
    entries = []
    for entry in pol.SUPERSESSION_ENTRIES:
        record = dict(entry)
        if record["status"] not in pol.SUPERSESSION_STATUSES:
            raise AssertionError(
                f"unknown supersession status: {record['status']}")
        if record["status"] == "PARTLY_SUPERSEDED" and not (
                record["valid_fields"] and record["invalid_fields"]):
            raise AssertionError(
                f"{record['artefact_group']} is PARTLY_SUPERSEDED but does "
                f"not name both the fields that remain valid and the fields "
                f"that do not")
        resolved = []
        for path in record["paths"]:
            full = vc.PROJECT_ROOT / path
            resolved.append({
                "path": path,
                "exists": full.exists(),
                "sha256": (vc.sha256_file(full) if full.exists()
                           else vc.NOT_AVAILABLE),
                "pinned_in_pre_closure_export_manifest":
                    path in ctx["hashes"],
                "pre_closure_export_sha256": ctx["hashes"].get(
                    path, vc.NOT_APPLICABLE),
            })
        record["path_records"] = resolved
        record["evidence_id"] = vc.evidence_id(
            "SUP", record["family"] + "."
            + vc.slug(record["artefact_group"])[:60])
        entries.append(record)

    entries.sort(key=lambda e: (e["family"], e["artefact_group"]))
    by_status = {}
    for entry in entries:
        by_status.setdefault(entry["status"], []).append(
            entry["artefact_group"])
    return {
        "title": "supersession map",
        "ve0_output": "C",
        "statuses": pol.SUPERSESSION_STATUSES,
        "rule": "an artefact not listed here has not been assessed and must "
                "not supply a dissertation number. A PARTLY_SUPERSEDED "
                "artefact names both the fields that survive and the fields "
                "that do not; there is no unqualified partial status.",
        "entry_count": len(entries),
        "by_status": {k: sorted(v) for k, v in sorted(by_status.items())},
        "clean_test_accessed": False,
        "entries": entries,
    }


# --------------------------------------------------------------------------
# Output D: claim ledger
# --------------------------------------------------------------------------

def build_claim_ledger(rows: list, index: dict) -> dict:
    claims = []
    for entry in pcl.CLAIMS:
        claim = dict(entry)
        claim_id = claim["claim_id"]
        if claim["status"] not in pcl.CLAIM_STATUSES:
            raise AssertionError(f"{claim_id} has an unknown status")
        if not claim["mandatory_caveat"]:
            raise AssertionError(f"{claim_id} carries no mandatory caveat")
        if not claim["forbidden_stronger_version"]:
            raise AssertionError(
                f"{claim_id} does not name its forbidden stronger version")
        for section in claim["dissertation_sections"]:
            if section not in con.SECTION_IDS:
                raise AssertionError(
                    f"{claim_id} names an unknown section: {section}")

        resolved = vc.resolve_evidence(entry, rows, index)
        if not resolved and not claim.get("evidence_free_scope_statement"):
            raise AssertionError(
                f"{claim_id} is a positive claim with no resolvable evidence")
        if resolved and claim.get("evidence_free_scope_statement"):
            raise AssertionError(
                f"{claim_id} is declared a scope statement but resolves "
                f"evidence")

        superseded = sorted(
            e for e in resolved
            if index[e]["scientific_role"] in ("SUPERSEDED",
                                               "NOT_FOR_REPORTING"))
        if superseded:
            raise AssertionError(
                f"{claim_id} rests on superseded evidence: {superseded}")

        claim["evidence_ids"] = resolved
        claim["evidence_count"] = len(resolved)
        claim["directional_evidence_count"] = sum(
            1 for e in resolved if index[e].get("excludes_zero") is True)
        claim["evidence_with_interval"] = sum(
            1 for e in resolved if index[e].get("ci95"))
        claim["families"] = sorted({index[e]["experiment_family"]
                                    for e in resolved})
        claim["is_positive_results_claim"] = claim["status"] in (
            "ESTABLISHED", "SUPPORTED", "TENTATIVE")
        claims.append(claim)

    claims.sort(key=lambda c: c["claim_id"])
    by_status = {}
    for claim in claims:
        by_status.setdefault(claim["status"], []).append(claim["claim_id"])
    return {
        "title": "final claim ledger",
        "ve0_output": "D",
        "statuses": pcl.CLAIM_STATUSES,
        "governing_rule": "this ledger controls the Results, the Discussion, "
                          "the Abstract, the Conclusion, the supervisor deck "
                          "and the viva preparation. None of them may state a "
                          "stronger version of a row here, and the "
                          "forbidden_stronger_version field says what "
                          "stronger would look like so the boundary is "
                          "checkable rather than a matter of taste.",
        "claim_count": len(claims),
        "by_status": {k: sorted(v) for k, v in sorted(by_status.items())},
        "preserved_from_closure": sorted(
            c["claim_id"] for c in claims
            if c["source"].startswith("closure")),
        "added_by_ve0": sorted(c["claim_id"] for c in claims
                               if c["source"].startswith("VE-0")),
        "clean_test_accessed": False,
        "claims": claims,
    }


# --------------------------------------------------------------------------
# Output E: research-question evidence matrix
# --------------------------------------------------------------------------

def build_rq_matrix(ledger: dict) -> dict:
    claims = {c["claim_id"]: c for c in ledger["claims"]}
    questions = []
    for entry in pcl.RESEARCH_QUESTIONS:
        question = dict(entry)
        rq_id = question["rq_id"]
        if question["answer_status"] not in pcl.ANSWER_STATUSES:
            raise AssertionError(f"{rq_id} has an unknown answer status")
        sections = (question["dissertation_section"]
                    if isinstance(question["dissertation_section"], list)
                    else [question["dissertation_section"]])
        for section in sections:
            if section not in con.SECTION_IDS:
                raise AssertionError(
                    f"{rq_id} names an unknown section: {section}")
        question["dissertation_section"] = sections

        evidence = []
        for claim_id in question["claim_ids"]:
            if claim_id not in claims:
                raise AssertionError(
                    f"{rq_id} names an unknown claim: {claim_id}")
            evidence.extend(claims[claim_id]["evidence_ids"])
        question["evidence_ids"] = sorted(set(evidence))
        question["evidence_count"] = len(question["evidence_ids"])

        answered = question["answer_status"] in ("ANSWERED",
                                                 "PARTIALLY_ANSWERED")
        if answered and not question["evidence_ids"]:
            supporting = [claims[c]["status"] for c in question["claim_ids"]]
            if not any(s in ("ESTABLISHED", "SUPPORTED", "TENTATIVE")
                       for s in supporting):
                raise AssertionError(
                    f"{rq_id} is marked {question['answer_status']} but rests "
                    f"on no supported claim")
        if question["answer_status"] == "NOT_ANSWERED":
            unsupported = [claims[c]["status"] for c in question["claim_ids"]]
            if "NOT_SUPPORTED" not in unsupported:
                raise AssertionError(
                    f"{rq_id} is NOT_ANSWERED but names no NOT_SUPPORTED "
                    f"claim to record why")
        if not question["limitation"]:
            raise AssertionError(f"{rq_id} carries no limitation")
        questions.append(question)

    questions.sort(key=lambda q: q["rq_id"])
    by_status = {}
    for question in questions:
        by_status.setdefault(question["answer_status"], []).append(
            question["rq_id"])
    return {
        "title": "research-question evidence matrix",
        "ve0_output": "E",
        "provenance_of_the_questions": pcl.RQ_PROVENANCE,
        "answer_statuses": pcl.ANSWER_STATUSES,
        "concept_separation": "held-out GQA evaluation (F2, not run), "
            "reasoning-depth analysis (the in-distribution step-bucket "
            "evidence) and true out-of-distribution compositional "
            "generalisation (never tested) are three distinct concepts. No "
            "answer here substitutes one for another.",
        "question_count": len(questions),
        "by_status": {k: sorted(v) for k, v in sorted(by_status.items())},
        "clean_test_accessed": False,
        "questions": questions,
    }


# --------------------------------------------------------------------------
# Outputs H and I: qualitative protocol and record schema
# --------------------------------------------------------------------------

def build_qualitative_protocol() -> dict:
    categories = []
    for entry in con.QUALITATIVE_CATEGORIES:
        category = dict(entry)
        source = category["evidence_source"]
        if source not in con.PREDICTION_EVIDENCE_SOURCES:
            raise AssertionError(
                f"{category['category_id']} names an undeclared prediction "
                f"evidence source: {source}")
        declared = con.PREDICTION_EVIDENCE_SOURCES[source]
        category["evidence_source_status"] = declared["status"]
        category["evidence_gives_prediction_text"] = \
            declared["gives_prediction_text"]
        if not category["interpretation_bound"]:
            raise AssertionError(
                f"{category['category_id']} carries no interpretation bound")
        if declared["status"] == "REQUIRES_VE2_VERIFICATION" and \
                not category.get("drop_if_unverified"):
            raise AssertionError(
                f"{category['category_id']} depends on unverified evidence "
                f"but does not declare drop_if_unverified")
        categories.append(category)

    categories.sort(key=lambda c: c["category_id"])
    available = [c["category_id"] for c in categories
                 if c["status"] == "AVAILABLE"]
    conditional = [c["category_id"] for c in categories
                   if c["status"] != "AVAILABLE"]
    return {
        "title": "deterministic qualitative-evidence protocol",
        "ve0_output": "H",
        "populated_by": "VE-2",
        "populated_here": False,
        "development_only": True,
        "clean_test_accessed": False,
        "selection_rule": con.QUALITATIVE_SELECTION_RULE,
        "prediction_evidence_sources": con.PREDICTION_EVIDENCE_SOURCES,
        "category_count": len(categories),
        "available_categories": sorted(available),
        "conditional_categories": sorted(conditional),
        "main_text_categories": sorted(
            c["category_id"] for c in categories
            if c["placement"] == "MAIN_TEXT"),
        "appendix_categories": sorted(
            c["category_id"] for c in categories
            if c["placement"] == "APPENDIX"),
        "completeness_rule": "not every category must be populated. A "
                             "category whose canonical prediction evidence "
                             "does not support it is recorded UNAVAILABLE and "
                             "dropped, and the drop is reported. Fabricating "
                             "or approximating the missing evidence is "
                             "forbidden.",
        "categories": categories,
    }


def build_qualitative_schema() -> dict:
    schema = dict(con.QUALITATIVE_RECORD_SCHEMA)
    names = [f["name"] for f in schema["fields"]]
    for required in ("qualitative_id", "question_id", "image_id",
                     "gold_answer", "systems", "limitation", "source_hashes",
                     "selection_rank", "selection_rank_key"):
        if required not in names:
            raise AssertionError(f"record schema is missing {required}")
    return {
        "title": "qualitative record schema",
        "ve0_output": "I",
        "populated_by": "VE-2",
        "populated_here": False,
        "clean_test_accessed": False,
        "field_count": len(schema["fields"]),
        **schema,
    }


# --------------------------------------------------------------------------
# Outputs J, K, L, M: the frozen contracts
# --------------------------------------------------------------------------

def build_style_contract() -> dict:
    return {
        "title": "visual and reporting style contract",
        "ve0_output": "J",
        "applies_to": ["dissertation main text", "dissertation appendices",
                       "supervisor presentation", "viva preparation"],
        "clean_test_accessed": False,
        **con.REPORTING_STYLE_CONTRACT,
    }


def build_provenance_contract(outputs: dict) -> dict:
    contract = dict(con.PROVENANCE_CONTRACT)
    contract["ve0_outputs_that_must_be_cited_by_ve1_and_ve2"] = sorted(
        outputs)
    return {
        "title": "provenance contract",
        "ve0_output": "K",
        "clean_test_accessed": False,
        **contract,
    }


def build_efficiency_contract(rows: list, index: dict) -> dict:
    contract = dict(con.EFFICIENCY_VISUALISATION_CONTRACT)
    groups = {}
    for row in rows:
        if row["evidence_class"] != "EFF":
            continue
        node = row.get("node") or vc.NOT_APPLICABLE
        groups.setdefault(str(node), []).append(row["evidence_id"])
    contract["measurement_groups"] = {
        node: sorted(ids) for node, ids in sorted(groups.items())}
    contract["group_count"] = len(groups)
    plottable = sorted(
        row["evidence_id"] for row in rows
        if row["evidence_class"] == "EFF"
        and row.get("timing_kind") == "END_TO_END_SERIAL")
    contract["end_to_end_plottable_evidence"] = plottable
    contract["never_plottable_on_an_end_to_end_axis"] = sorted(
        row["evidence_id"] for row in rows
        if row["evidence_class"] == "EFF"
        and row.get("timing_kind") != "END_TO_END_SERIAL")

    resolved, unresolved = [], []
    for row in rows:
        if row["evidence_class"] != "EFF":
            continue
        pairing = row.get("accuracy_pairing") or {}
        if pairing.get("status") == "RESOLVED":
            resolved.append(row["evidence_id"])
        elif pairing.get("status") == "PAIRING_UNRESOLVED":
            unresolved.append({
                "evidence_id": row["evidence_id"],
                "question_ve1_must_answer":
                    pairing["question_ve1_must_answer"]})
    contract["accuracy_pairing"] = {
        "denominator": pol.EFFICIENCY_ACCURACY_DENOMINATOR,
        "why_this_exists": "the efficiency table carries no accuracy: its "
            "64-row timing sample supports no accuracy claim. An "
            "accuracy-against-latency figure therefore needs a vertical "
            "coordinate from each system's own frozen artefact, on one stated "
            "denominator. Binding the pointer in VE-0 is what prevents VE-1 "
            "pairing a latency with an accuracy measured on a different "
            "denominator.",
        "resolved": sorted(resolved),
        "resolved_count": len(resolved),
        "unresolved": sorted(unresolved, key=lambda u: u["evidence_id"]),
        "unresolved_count": len(unresolved),
        "ve1_precondition": "an unresolved row is NOT paired speculatively. "
            "VE-1 must either resolve the recorded question from the frozen "
            "artefacts and record the pointer it used, or plot the row "
            "latency-only, or omit it and state the omission. Guessing a "
            "denominator is forbidden.",
    }
    return {
        "title": "efficiency visualisation contract",
        "ve0_output": "L",
        "clean_test_accessed": False,
        **contract,
    }


def build_results_mapping(figures: dict, tables: dict, ledger: dict,
                          matrix: dict) -> dict:
    sections = []
    for entry in con.RESULTS_SECTIONS:
        section = dict(entry)
        section_id = section["section_id"]
        section["figures"] = sorted(
            f["figure_id"] for f in figures["figures"]
            if section_id in f["dissertation_section"])
        section["tables"] = sorted(
            t["table_id"] for t in tables["tables"]
            if section_id in t["dissertation_section"])
        section["claims"] = sorted(
            c["claim_id"] for c in ledger["claims"]
            if section_id in c["dissertation_sections"])
        section["research_questions"] = sorted(
            q["rq_id"] for q in matrix["questions"]
            if section_id in q["dissertation_section"])
        if section.get("placeholder"):
            if section["figures"] or section["tables"]:
                raise AssertionError(
                    f"{section_id} is a placeholder but has been assigned "
                    f"content")
        sections.append(section)

    unassigned_figures = sorted(
        f["figure_id"] for f in figures["figures"]
        if not any(s in con.SECTION_IDS for s in f["dissertation_section"]))
    return {
        "title": "results-section mapping",
        "ve0_output": "M",
        "rule": "every figure, table and claim belongs to a named section, "
                "and R10 stays empty until F2.",
        "placeholder_sections": sorted(
            s["section_id"] for s in sections if s.get("placeholder")),
        "unassigned_figures": unassigned_figures,
        "section_count": len(sections),
        "clean_test_accessed": False,
        "sections": sections,
    }
