"""E9 bounded preflight: freeze the protocol, or halt.

This is the ONLY stage permitted to establish frozen values. It runs the
checks the user's amendment 14 enumerates and nothing else; it is not an
open-ended audit. Its single scientific output is
`results/experiments/e9_compact_vlm/frozen_protocol.json`, after which no
prompt, generation parameter, condition, support or scorer may change.

It observes no GQA score. It generates at most a handful of tokens on a
technical smoke batch, purely to prove the processor accepts each image
condition, and it discards that text without scoring it.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

import e9_common as e9
import generate as e9gen


def check_dependency_gate() -> dict:
    """`accelerate` is absent. If loading needs it, that is a stop-and-ask."""
    try:
        import accelerate  # noqa: F401
        present = True
        version = accelerate.__version__
    except ImportError:
        present = False
        version = None
    import tokenizers
    import transformers
    return {"accelerate_installed": present, "accelerate_version": version,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "torch": torch.__version__,
            "policy": "if the idefics3 path required accelerate this would "
                      "be a stop-and-ask, not an install"}


def check_exposure(model_key: str) -> dict:
    """G-EXPOSE: the documented mixture at the pinned revision.

    Amendment 1: the absence of GQA from a published list is NOT proof of
    zero image or content overlap, and this record says so in the artefact
    rather than only in the report."""
    from huggingface_hub import hf_hub_download
    pin = e9.MODELS[model_key]
    path = hf_hub_download(pin["repo"], "README.md",
                           revision=pin["revision"])
    card = Path(path).read_text(encoding="utf-8")
    lowered = card.lower()
    names_cauldron = "the_cauldron" in lowered or "the cauldron" in lowered
    names_docmatix = "docmatix" in lowered
    gqa_hits = [m.start() for m in re.finditer(r"\bgqa\b", lowered)]
    e9.require(names_cauldron and names_docmatix, "G-EXPOSE",
               f"{model_key} card no longer documents The Cauldron and "
               f"Docmatix; the exposure table may not be amended silently")
    e9.require(not gqa_hits, "G-EXPOSE",
               f"{model_key} card now mentions GQA at offsets {gqa_hits}; "
               f"stop and return to the user")
    return {
        "card_sha256": e9.sha256_file(Path(path)),
        "documented_datasets": ["HuggingFaceM4/the_cauldron",
                                "HuggingFaceM4/Docmatix"],
        "gqa_named_in_card": False,
        "supervised_gqa_claim": "NOT MADE. No direct GQA supervised-training "
                                "claim is made in either direction.",
        "overlap_caveat": "Absence of GQA from a published training-dataset "
                          "list is NOT proof of zero image or content "
                          "overlap. GQA imagery derives from Visual Genome, "
                          "which Cauldron subsets draw on, so image-level "
                          "exposure is likely and E9 is never described as "
                          "zero-shot at the image level.",
        "framing": "CONTEXTUAL POSITIONING, not a matched causal comparison. "
                   "SmolVLM is externally pretrained and instruction-tuned.",
    }


def check_conditions(processor, model, frame, derangement) -> dict:
    """G-COND: every image condition is technically accepted, proven by
    actually running two rows of each through the processor and the model."""
    blank = e9.blank_image()
    accepted = {}
    device = e9gen.config_device()
    for condition in e9.CONDITIONS:
        images = e9.resolve_images(frame.iloc[:2], condition, derangement)
        inputs = e9gen.build_batch(processor, list(frame["question"][:2]),
                                   images, blank).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=2,
                                 do_sample=False, num_beams=1,
                                 pad_token_id=processor.tokenizer.pad_token_id)
        accepted[condition] = {
            "accepted": True,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "image_tokens": int((inputs["input_ids"]
                                 == model.config.image_token_id).sum()) // 2,
            "pixel_values_shape": list(inputs["pixel_values"].shape),
            "generated_shape": list(out.shape),
            "role": e9.CONDITION_ROLE[condition],
        }
    return accepted


def check_scorer() -> dict:
    """The reviewed G21 scorer is imported unmodified and re-verified."""
    sys.path.insert(0, str(e9.E8A_DIR))
    import g21_scorer as g21
    provenance = g21.scorer_provenance()
    # Independent literal expectations, not outputs of the implementation.
    cases = [("Yes.", "yes", True), (" yes ", "yes", True),
             ("The answer is yes", "yes", False), ("2", "two", True),
             ("A man", "man", True), ("no", "yes", False)]
    for prediction, gold, expected in cases:
        got = g21.normalized_exact(prediction, gold)
        e9.require(got == expected, "G21",
                   f"normalised_exact({prediction!r}, {gold!r}) is {got}, "
                   f"expected {expected}")
    e9.require(not g21.raw_exact("Yes.", "yes"), "G21",
               "strict raw exact must not match 'Yes.' against 'yes'")
    return {"provenance": provenance, "independent_cases": len(cases),
            "verified": True}


def check_images(frame) -> dict:
    missing = [i for i in sorted(set(frame["imageId"]))
               if not e9.image_path(i).exists()]
    e9.require(not missing, "G-IMAGES",
               f"{len(missing)} development images do not resolve")
    return {"distinct_images": frame["imageId"].nunique(), "missing": 0}


def check_output_paths() -> dict:
    """Amendment 14: output paths must be empty or validly resumable, and no
    stale scientific result may be mistaken for a fresh one."""
    e9.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in e9.RESULTS_DIR.iterdir())
    scientific = [name for name in existing
                  if name.endswith((".json", ".npz", ".csv", ".png"))
                  and name not in ("frozen_protocol.json",
                                   "resource_ledger.json", "preflight.json")]
    stale = []
    for name in scientific:
        path = e9.RESULTS_DIR / name
        if name.endswith(".json"):
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                stale.append(name)
                continue
            if not any(key.startswith("e9_") for key in payload):
                stale.append(name)
    e9.require(not stale, "G-STALE",
               f"unrecognised artefacts in the output directory: {stale}")
    return {"existing_files": existing, "resumable_records": scientific,
            "stale": stale}


def check_vocabularies() -> dict:
    """G18: neither vocabulary may drift, and V1 indices can never leak."""
    hundred = e9.sha256_file(e9.VOCAB_100)
    thousand = e9.sha256_file(e9.VOCAB_1000)
    e9.require(hundred == e9.VOCAB_100_SHA256, "G18",
               f"top-100 vocabulary sha256 {hundred} does not match the pin")
    e9.require(thousand == e9.VOCAB_1000_SHA256, "G18",
               f"top-1000 vocabulary sha256 {thousand} does not match the pin")
    answers_100 = e9.load_vocabulary(e9.VOCAB_100)
    answers_1000 = e9.load_vocabulary(e9.VOCAB_1000)
    e9.require(answers_1000[:100] == answers_100 or
               set(answers_100).issubset(set(answers_1000)), "G18",
               "the top-100 answers are not a subset of the top-1000 answers")
    return {"top_100_sha256": hundred, "top_1000_sha256": thousand,
            "top_100_size": len(answers_100),
            "top_1000_size": len(answers_1000),
            "top_100_is_subset_of_top_1000": True}


def build_trie_record(processor) -> dict:
    """G-TRIE: freeze the answer support and the derived constrained cap."""
    answers = e9.load_vocabulary(e9.VOCAB_1000)
    cache = e9.build_answer_sequences(processor.tokenizer, answers)
    root = e9.build_trie(cache)

    # The accepted string set, after stripping, must be exactly the support.
    accepted = set()

    def walk(node, prefix):
        if node["leaf"] is not None:
            accepted.add(processor.tokenizer.decode(
                prefix, skip_special_tokens=True,
                clean_up_tokenization_spaces=False).strip())
        for token in sorted(node["children"]):
            walk(node["children"][token], prefix + [token])
    walk(root, [])
    e9.require(accepted == set(answers), "G-TRIE",
               f"the trie accepts {len(accepted)} distinct stripped strings, "
               f"expected exactly the {len(answers)} frozen answers")
    cap = cache["max_length"] + e9.CONSTRAINED_CAP_MARGIN
    return {
        "support": e9.TRIE_SUPPORT,
        "n_answers": len(answers),
        "n_token_sequences": len(cache["sequences"]),
        "surface_forms": "bare and one leading space, both mapping to the "
                         "same answer index",
        "max_sequence_length": cache["max_length"],
        "length_histogram": cache["length_histogram"],
        "constrained_max_new_tokens": cap,
        "sequences_sha256": cache["sha256"],
        "accepted_set_equals_support": True,
        "support_disclosure": "The constrained readout offers 1000 "
                              "candidates. E8B's classifiers choose among "
                              "100. A top-1000 constrained task is NOT "
                              "equated with E8B's top-100 classifier task; "
                              "the difference is disclosed wherever the two "
                              "appear together.",
    }


def main() -> int:
    started = time.time()
    e9.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if e9.FROZEN_PROTOCOL_PATH.exists():
        print("E9 REFUSED: frozen_protocol.json already exists. The protocol "
              "is frozen and preflight may not re-freeze it.",
              file=sys.stderr)
        return 4

    report: dict = {"e9_preflight": {}}
    section = report["e9_preflight"]
    section["dated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    section["embargo_scan_before"] = e9.embargo_source_scan()
    section["dependency_gate"] = check_dependency_gate()
    section["output_paths"] = check_output_paths()
    section["vocabularies"] = check_vocabularies()

    frame = e9.load_dev_raw()
    section["denominators"] = {
        "raw_rows": len(frame), "raw_clusters": frame["imageId"].nunique(),
        "in_vocabulary_rows": int(frame["in_vocabulary"].sum()),
        "in_vocabulary_clusters": int(
            frame[frame["in_vocabulary"]]["imageId"].nunique()),
        "top_1000_rows": int(frame["answer"].isin(
            set(e9.load_vocabulary(e9.VOCAB_1000))).sum()),
    }
    e9.require(section["denominators"]["in_vocabulary_clusters"]
               == e9.IN_VOCAB_CLUSTERS, "G-ROWS",
               "the in-vocabulary cluster count is not 768")
    e9.require(section["denominators"]["top_1000_rows"] == e9.TOP1000_ROWS,
               "G-ROWS", "the top-1000 row count is not 9,823")
    section["images"] = check_images(frame)
    section["scorer"] = check_scorer()

    mapping, provenance = e9.build_derangement(frame)
    section["derangement"] = provenance
    section["blank_image"] = {"size": list(e9.BLANK_SIZE),
                              "value": list(e9.BLANK_VALUE),
                              "sha256": e9.blank_image_sha256(),
                              "role": "AUXILIARY / UNMATCHED ABLATION",
                              "note": "not matched to E8B fixed_image, which "
                                      "is a feature-space mean"}

    models = {}
    exposure = {}
    conditions = {}
    trie = None
    image_token_id = None
    for model_key in e9.MODELS:
        exposure[model_key] = check_exposure(model_key)
        processor, model = e9gen.load_model(model_key)
        runtime = e9.describe_runtime(model_key, processor, model)
        e9.require(runtime["trainable_parameters"] == 0, "G-FROZEN",
                   f"{model_key} is not fully frozen")
        torch.cuda.reset_peak_memory_stats()
        conditions[model_key] = check_conditions(processor, model, frame,
                                                 mapping)
        runtime["smoke_peak_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved())
        runtime["checkpoint_bytes"] = int(
            runtime["parameters"] * 2)   # bf16 on disk
        models[model_key] = runtime
        image_token_id = int(model.config.image_token_id)
        if trie is None:
            trie = build_trie_record(processor)
        del model
        torch.cuda.empty_cache()

    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    worst_peak = max(m["smoke_peak_reserved_bytes"] for m in models.values())
    section["memory"] = {
        "device_total_bytes": total_memory,
        "device_total_mib": round(total_memory / 2**20),
        "smoke_worst_peak_bytes": worst_peak,
        "ceiling_bytes": int(total_memory * e9.MEMORY_FRACTION_CEILING),
        "sufficient": worst_peak <= total_memory * e9.MEMORY_FRACTION_CEILING,
    }
    e9.require(section["memory"]["sufficient"], "G19",
               "the smoke peak already exceeds the 80 per cent ceiling")
    section["exposure"] = exposure
    section["conditions"] = conditions
    section["embargo_scan_after"] = e9.embargo_source_scan()
    e9.require(section["embargo_scan_before"]
               == section["embargo_scan_after"], "G17",
               "the E9 embargo scan changed during preflight")
    section["elapsed_seconds"] = round(time.time() - started, 3)
    report["metadata"] = e9.run_metadata()

    frozen = {
        "e9_frozen_protocol": {
            "frozen_utc": section["dated_utc"],
            "framing": "E9 is a CONTEXTUAL BASELINE, not a clean causal "
                       "architecture experiment. Training history and "
                       "multimodal pretraining differ, so no E9-vs-E8 "
                       "difference is attributed to integrated architecture "
                       "alone.",
            "primary_readout": "open generation",
            "secondary_readout": "trie-constrained generation, DIAGNOSTIC "
                                 "only, never promoted to primary",
            "auxiliary_condition": "blank image, UNMATCHED ablation",
        },
        "prompt": e9.PROMPT_TEMPLATE,
        "prompt_sha256": e9.sha256_bytes(e9.PROMPT_TEMPLATE.encode("utf-8")),
        "models": models,
        "exposure": exposure,
        "trie": trie,
        "image_token_id": image_token_id,
        "derangement": provenance,
        "blank_image": section["blank_image"],
        "conditions": {c: e9.CONDITION_ROLE[c] for c in e9.CONDITIONS},
        "denominators": section["denominators"],
        "statistics": {"draws": e9.BOOTSTRAP_DRAWS,
                       "rng": f"default_rng({e9.BOOTSTRAP_SEED})",
                       "alpha": e9.ALPHA,
                       "raw_clusters": e9.RAW_CLUSTERS,
                       "in_vocabulary_clusters": e9.IN_VOCAB_CLUSTERS},
        "batch_size": e9.BATCH_SIZE,
        "branch_halt_hours": e9.BRANCH_HALT_HOURS,
        "execution_priority": {"mandatory": [list(x) for x
                                             in e9.MANDATORY_ORDER],
                               "optional": [list(x) for x
                                            in e9.OPTIONAL_ORDER]},
        "vocabularies": section["vocabularies"],
        "scorer": section["scorer"]["provenance"],
    }
    frozen_sha = e9.atomic_write_json(e9.FROZEN_PROTOCOL_PATH, frozen)
    section["frozen_protocol_sha256"] = frozen_sha
    e9.atomic_write_json(e9.RESULTS_DIR / "preflight.json", report)
    e9.charge_ledger("preflight", time.time() - started,
                     "bounded preflight and protocol freeze")

    print(json.dumps({
        "frozen_protocol_sha256": frozen_sha,
        "constrained_max_new_tokens": trie["constrained_max_new_tokens"],
        "trie_sequences": trie["n_token_sequences"],
        "derangement_sha256": provenance["sha256"],
        "image_tokens_normal": conditions["smolvlm_256m"]["normal"]
                                        ["image_tokens"],
        "memory_sufficient": section["memory"]["sufficient"],
        "elapsed_seconds": section["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
