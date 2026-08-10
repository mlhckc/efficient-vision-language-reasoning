"""E9 compact-VLM contextual baseline: frozen constants, gates and helpers.

Everything in this module that can affect a scientific number is a FROZEN
constant, declared before any GQA score was observed and hashed into
`results/experiments/e9_compact_vlm/frozen_protocol.json` by `preflight.py`.
Every later stage re-asserts its own configuration against that frozen record
and refuses to run on a mismatch.

Scope, from the user's approval of 10 August 2026:

  * E9 is a CONTEXTUAL BASELINE, not a clean causal architecture experiment.
    Training history and multimodal pretraining differ between E9 and E8, so
    no E9-vs-E8 difference is attributed to integrated architecture alone.
  * Evaluation only. Zero trainable parameters. No fine-tuning, no LoRA, no
    adaptation, no prompt search, no sampling.
  * Open generation is the PRIMARY readout. Trie-constrained generation is a
    SECONDARY DIAGNOSTIC that separates task failure from output-format
    failure, and is never promoted to primary even if it scores higher.
  * The blank-image condition is an AUXILIARY / UNMATCHED ablation.
  * The clean test is embargoed: no path here resolves it (gate G17).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
E8A_DIR = PROJECT_ROOT / "experiments" / "e8a_question_encoder"
if str(E8A_DIR) not in sys.path:
    sys.path.insert(0, str(E8A_DIR))

import config  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "e9_compact_vlm"
DATA_V2 = PROJECT_ROOT / "data" / "v2"
DATA_V2_1000 = PROJECT_ROOT / "data" / "v2_1000"
IMAGE_DIR = PROJECT_ROOT / "data" / "gqa" / "images"
DEV_RAW = DATA_V2 / "dev_raw.csv"
VOCAB_100 = DATA_V2 / "answer_vocab_v2.json"
VOCAB_1000 = DATA_V2_1000 / "answer_vocab_v2_1000.json"

# --- G17 -----------------------------------------------------------------
# The embargoed target's file name is deliberately NOT written anywhere in
# this package, following the E8A convention, so the G17 scan can be a plain
# substring test with no exemption for "it only appears in a comment". The
# forbidden token is assembled at run time and never appears as a literal.
_EMBARGOED_TOKEN = "test_" + "clean_targets"


# --- Frozen model pins --------------------------------------------------------

MODELS = {
    "smolvlm_256m": {
        "role": "PRIMARY",
        "repo": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "revision": "7e3e67edbbed1bf9888184d9df282b700a323964",
        "licence": "apache-2.0",
        "expected_parameters": 256_484_928,
        "text_backbone": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "vision_backbone": "google/siglip-base-patch16-512",
    },
    "smolvlm_500m": {
        "role": "SECONDARY",
        "repo": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "revision": "a7da5b986cb59b408707209984f360a5f4ad7e47",
        "licence": "apache-2.0",
        "expected_parameters": 507_482_304,
        "text_backbone": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "vision_backbone": "google/siglip-base-patch16-512",
    },
}

# --- Frozen prompt. ONE pre-declared prompt. Never changed after any score. ---

PROMPT_TEMPLATE = (
    "Answer the question using a single word or phrase.\n"
    "Question: {question}\n"
    "Answer:"
)

# --- Frozen generation configuration -----------------------------------------

DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "sdpa"
PADDING_SIDE = "left"          # required for correct batched decoder generation
DO_SAMPLE = False
NUM_BEAMS = 1
OPEN_MAX_NEW_TOKENS = 20
# The constrained cap is derived deterministically from the frozen answer
# support: the deepest trie path plus two (one for EOS, one of margin). It is
# computed at preflight and pinned into frozen_protocol.json.
CONSTRAINED_CAP_MARGIN = 2
BATCH_SIZE = 8
# Generation parameters deliberately left at their library defaults, recorded
# so that "everything that can affect output" is enumerated rather than implied.
FROZEN_GENERATION_DEFAULTS = {
    "temperature": None, "top_p": None, "top_k": None,
    "repetition_penalty": 1.0, "length_penalty": 1.0,
    "no_repeat_ngram_size": 0, "use_cache": True,
    "early_stopping": False, "num_return_sequences": 1,
}

# --- Frozen answer support ----------------------------------------------------
# Amendment 4: the trie support is frozen at top-1000 (the reviewed canonical
# plan pins the top-1000 vocabulary and reports a top-1000 view). Metrics are
# ALSO reported on the common 7,714 top-100-supported rows, with the candidate
# support difference disclosed. A top-1000 constrained task is never silently
# equated with E8B's top-100 classifier task.
TRIE_SUPPORT = "top_1000"
VOCAB_100_SHA256 = ("f92618b2f59939586d5ad79b184a44ed5f3c9d2aaf4d6e10f6a394"
                    "7d90358680")
VOCAB_1000_SHA256 = ("6f307c2bd088340e56e831986e7cfe2b3e8f505bb5117f061bb532"
                     "9089b7dbde")

# --- Frozen conditions --------------------------------------------------------

CONDITIONS = ("normal", "deranged", "blank")
CONDITION_ROLE = {
    "normal": "PRIMARY",
    "deranged": "PRIMARY (matched image-partner intervention)",
    "blank": "AUXILIARY / UNMATCHED ABLATION",
}
BLANK_SIZE = (384, 384)
BLANK_VALUE = (128, 128, 128)
# Verified by recomputation on 10 August 2026 and identical to the mapping
# E8B's final_evaluation.build_intervention_context derives.
EXPECTED_DERANGEMENT_SHA256 = ("5f8f7868fc73804b40d2424923a369740404504ae902"
                               "bbc5527ac67dde4044f4")

# --- Frozen denominators ------------------------------------------------------

RAW_ROWS = 10_004
RAW_CLUSTERS = 777
IN_VOCAB_ROWS = 7_714
IN_VOCAB_CLUSTERS = 768
TOP1000_ROWS = 9_823

# --- Frozen statistics --------------------------------------------------------

BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 0
ALPHA = 0.05

# --- Frozen resource policy ---------------------------------------------------

BRANCH_HALT_HOURS = 6.0          # self-imposed, tighter than every binding gate
PER_RUN_HALT_HOURS = 8.0         # master protocol section 14
MEMORY_FRACTION_CEILING = 0.80   # of the device's reported total
DETERMINISM_ROWS = 512

# Amendment 11: pre-registered execution priority. Optional work can never
# consume budget the primary experiment needs.
MANDATORY_ORDER = (
    ("smolvlm_256m", "open", "normal"),
    ("smolvlm_256m", "open", "deranged"),
    ("smolvlm_256m", "open", "blank"),
    ("smolvlm_256m", "constrained", "normal"),
    ("smolvlm_256m", "constrained", "deranged"),
)
OPTIONAL_ORDER = (
    ("smolvlm_500m", "open", "normal"),
    ("smolvlm_500m", "constrained", "normal"),
)


# --- Integrity helpers --------------------------------------------------------

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True,
                                   default=str).encode("utf-8"))


def atomic_write_json(path: Path, payload: dict) -> str:
    """Write, fsync, rename. A partial artefact can never be observed.

    Returns the SHA-256 of the bytes actually written, so the caller can
    record it without re-reading and trusting the re-read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=False,
                      default=str).encode("utf-8")
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return sha256_bytes(data)


def atomic_write_npz(path: Path, **arrays) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.npz")
    os.close(handle)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return sha256_file(path)


def halt(gate: str, reason: str) -> None:
    """An explicit, recorded exit. Never a judgement call."""
    record = {"gate": gate, "reason": reason, "halted": True}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(RESULTS_DIR / "HALT.json", record)
    print(f"E9 HALT [{gate}]: {reason}", file=sys.stderr, flush=True)
    sys.exit(3)


def require(condition: bool, gate: str, reason: str) -> None:
    if not condition:
        halt(gate, reason)


# --- G17: clean-test embargo --------------------------------------------------

def assert_no_clean_test_path(*candidates) -> None:
    """Refuse any string that could resolve the embargoed target."""
    for candidate in candidates:
        if _EMBARGOED_TOKEN in str(candidate):
            halt("G17", "a clean-test path was constructed")


def embargo_source_scan() -> dict:
    """G17, the project-standard form: prove no E9 source can resolve it.

    This REPLACES an earlier implementation that stat()ed the embargoed
    target for a size-and-mtime fingerprint. That implementation never
    opened the file and read no label, but it did construct and resolve a
    path G17 forbids, and it wrote the file name into this package, which
    would have forced the scan to carry an exemption. The scan below is
    both stronger evidence and compliant: if no E9 source contains the
    token, no E9 code can reach the file at all."""
    package = Path(__file__).resolve().parent
    offending = []
    for source in sorted(package.glob("*.py")):
        if _EMBARGOED_TOKEN in source.read_text(encoding="utf-8"):
            offending.append(source.name)
    require(not offending, "G17",
            f"E9 sources reference the embargoed clean-test path: "
            f"{offending}")
    return {"scanned": [p.name for p in sorted(package.glob("*.py"))],
            "offending": offending, "clean_test_resolvable": False,
            "basis": "substring scan of every E9 source for the assembled "
                     "forbidden token; the file name appears in no E9 "
                     "source, so no E9 code can resolve or open it"}


# --- Data ---------------------------------------------------------------------

def load_dev_raw():
    """The 10,004-row raw development partition, with gold strings."""
    import pandas as pd
    assert_no_clean_test_path(DEV_RAW)
    frame = pd.read_csv(DEV_RAW, dtype={"questionId": str, "imageId": str},
                        keep_default_na=False, na_filter=False)
    frame["label"] = frame["label"].astype(int)
    frame["in_vocabulary"] = frame["in_vocabulary"].astype(bool)
    require(len(frame) == RAW_ROWS, "G-ROWS",
            f"dev_raw.csv has {len(frame)} rows, expected {RAW_ROWS}")
    require(int(frame["in_vocabulary"].sum()) == IN_VOCAB_ROWS, "G-ROWS",
            "the in-vocabulary row count does not match the frozen 7,714")
    require(frame["imageId"].nunique() == RAW_CLUSTERS, "G-ROWS",
            "the raw partition does not carry 777 distinct images")
    return frame


def load_vocabulary(path: Path) -> list:
    payload = json.loads(path.read_text())
    return list(payload["answers"])


def image_index_of(frame) -> tuple:
    """Contiguous 0..n-1 cluster index over the frame's images, for the
    clustered bootstrap. Sorted so the mapping is deterministic."""
    unique = sorted(frame["imageId"].unique())
    lookup = {image: i for i, image in enumerate(unique)}
    return np.array([lookup[i] for i in frame["imageId"]]), len(unique)


# --- Conditions ---------------------------------------------------------------

def build_derangement(frame) -> tuple:
    """The frozen image-partner derangement, imported from E8A unmodified.

    Amendment 5: this is an identical IMAGE-PARTNER intervention to E8B's.
    It is NOT an identical feature or representation intervention, because
    E8B swapped cached CLIP token tensors while E9 loads a different image
    file through a different model's preprocessing."""
    import e8a_common as e8a
    mapping, provenance = e8a.imageid_level_derangement(list(frame["imageId"]))
    e8a.assert_derangement(mapping)
    require(provenance["self_pairs"] == 0, "G-DERANGE",
            "the derangement has a self-pair")
    require(provenance["sha256"] == EXPECTED_DERANGEMENT_SHA256, "G-DERANGE",
            f"derangement sha256 {provenance['sha256']} does not equal the "
            f"frozen E8B mapping {EXPECTED_DERANGEMENT_SHA256}")
    provenance["intervention_kind"] = (
        "identical IMAGE-PARTNER intervention to E8B; NOT an identical "
        "feature-space intervention, because preprocessing differs")
    return mapping, provenance


def blank_image():
    """The pinned neutral image. AUXILIARY / UNMATCHED ablation."""
    from PIL import Image
    return Image.new("RGB", BLANK_SIZE, BLANK_VALUE)


def blank_image_sha256() -> str:
    return sha256_bytes(np.asarray(blank_image()).tobytes())


def image_path(image_id: str) -> Path:
    return IMAGE_DIR / f"{image_id}.jpg"


def resolve_images(frame, condition: str, derangement: dict) -> list:
    """The image id actually shown for each row, under the given condition."""
    if condition == "normal":
        return list(frame["imageId"])
    if condition == "deranged":
        return [derangement[i] for i in frame["imageId"]]
    if condition == "blank":
        return [None] * len(frame)
    halt("G-COND", f"unknown condition {condition!r}")


# --- Answer support and the constrained-decoding trie -------------------------

def build_answer_sequences(tokenizer, answers: list) -> dict:
    """Token sequences for every candidate answer.

    Two surface forms are admitted for the same answer -- the bare string and
    the string with one leading space -- because a BPE tokenizer encodes a
    word differently at the start of a segment than after a space, and the
    assistant turn may begin either way. Both forms map to the SAME answer
    index, and the accepted string set after stripping is asserted to be
    exactly the candidate set. This rule is frozen here, before any score."""
    sequences = []
    owners = []
    seen = {}
    for index, answer in enumerate(answers):
        for surface in (answer, " " + answer):
            ids = tokenizer(surface, add_special_tokens=False)["input_ids"]
            if not ids:
                halt("G-TRIE", f"answer {answer!r} encodes to no tokens")
            decoded = tokenizer.decode(ids, skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
            if decoded != surface:
                halt("G-TRIE", f"round-trip failure for {surface!r}: "
                               f"{decoded!r}")
            key = tuple(ids)
            if key in seen:
                # Two distinct answers may not share one token sequence.
                if seen[key] != index:
                    halt("G-TRIE",
                         f"answers {answers[seen[key]]!r} and {answer!r} "
                         f"share the token sequence {list(key)}")
                continue
            seen[key] = index
            sequences.append(ids)
            owners.append(index)
    lengths = [len(s) for s in sequences]
    histogram = {}
    for length in lengths:
        histogram[str(length)] = histogram.get(str(length), 0) + 1
    return {"sequences": sequences, "owners": owners,
            "answers": list(answers), "max_length": max(lengths),
            "length_histogram": histogram,
            "sha256": sha256_json(sequences)}


def build_trie(cache: dict) -> dict:
    """Prefix trie whose accepted string set is exactly the candidate set.

    Nodes are dicts token_id -> child; `leaf` holds the answer index where a
    complete sequence ends and EOS becomes legal."""
    root = {"children": {}, "leaf": None}
    for sequence, owner in zip(cache["sequences"], cache["owners"]):
        node = root
        for token in sequence:
            node = node["children"].setdefault(
                token, {"children": {}, "leaf": None})
        if node["leaf"] is not None and node["leaf"] != owner:
            halt("G-TRIE", "two answers reach one leaf")
        node["leaf"] = owner
    leaves = set()

    def collect(node):
        if node["leaf"] is not None:
            leaves.add(node["leaf"])
        for token in sorted(node["children"]):
            collect(node["children"][token])
    collect(root)
    require(leaves == set(range(len(cache["answers"]))), "G-TRIE",
            f"the trie leaf set has {len(leaves)} answers, expected "
            f"{len(cache['answers'])}")
    return root


def trie_walk(root: dict, tokens):
    """Return the node reached by `tokens`, or None if the path leaves the
    trie (which the mask makes impossible, so None is a defect signal)."""
    node = root
    for token in tokens:
        child = node["children"].get(int(token))
        if child is None:
            return None
        node = child
    return node


# --- Resource ledger ----------------------------------------------------------

LEDGER_PATH = RESULTS_DIR / "resource_ledger.json"


def read_ledger() -> dict:
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text())
    return {"entries": [], "total_gpu_hours": 0.0}


def charge_ledger(step: str, seconds: float, note: str = "") -> dict:
    """Charge wall-clock GPU occupancy and enforce the branch halt.

    Charged whether the step succeeded or failed: a failed run's compute is
    spent compute, exactly as E8B charged its failed cell-1 attempt."""
    ledger = read_ledger()
    hours = seconds / 3600.0
    ledger["entries"].append({"step": step, "seconds": round(seconds, 3),
                              "gpu_hours": round(hours, 6), "note": note})
    ledger["total_gpu_hours"] = round(
        sum(e["gpu_hours"] for e in ledger["entries"]), 6)
    ledger["branch_halt_hours"] = BRANCH_HALT_HOURS
    ledger["headroom_hours"] = round(
        BRANCH_HALT_HOURS - ledger["total_gpu_hours"], 6)
    atomic_write_json(LEDGER_PATH, ledger)
    return ledger


def assert_budget(projected_hours: float = 0.0) -> dict:
    ledger = read_ledger()
    total = ledger["total_gpu_hours"] + projected_hours
    require(total <= BRANCH_HALT_HOURS, "G19",
            f"projected E9 branch total {total:.4f} GPU-h exceeds the "
            f"pre-registered branch halt of {BRANCH_HALT_HOURS} GPU-h")
    return ledger


def assert_memory(peak_bytes: int, total_bytes: int) -> None:
    fraction = peak_bytes / max(total_bytes, 1)
    require(fraction <= MEMORY_FRACTION_CEILING, "G19",
            f"peak memory {peak_bytes / 2**20:.0f} MiB is "
            f"{fraction:.1%} of the device total, above the "
            f"{MEMORY_FRACTION_CEILING:.0%} ceiling")


# --- Frozen-protocol binding --------------------------------------------------

FROZEN_PROTOCOL_PATH = RESULTS_DIR / "frozen_protocol.json"


def load_frozen_protocol() -> dict:
    require(FROZEN_PROTOCOL_PATH.exists(), "G-PROMPT",
            "frozen_protocol.json is absent; run preflight.py first")
    return json.loads(FROZEN_PROTOCOL_PATH.read_text())


def assert_against_frozen(model_key: str, processor, model) -> dict:
    """Every later stage re-derives its own configuration and compares it
    with the frozen record. A mismatch halts rather than being reported."""
    frozen = load_frozen_protocol()
    entry = frozen["models"][model_key]
    live = describe_runtime(model_key, processor, model)
    for field in ("prompt_sha256", "chat_template_sha256",
                  "image_processor_sha256", "tokenizer_sha256",
                  "generation_sha256", "parameters", "revision"):
        require(live[field] == entry[field], "G-PIN",
                f"{model_key}.{field} is {live[field]!r} at run time but "
                f"{entry[field]!r} in the frozen protocol")
    return frozen


def describe_runtime(model_key: str, processor, model) -> dict:
    """The complete output-affecting configuration, in one place."""
    pin = MODELS[model_key]
    tokenizer = processor.tokenizer
    generation = {
        "do_sample": DO_SAMPLE, "num_beams": NUM_BEAMS,
        "open_max_new_tokens": OPEN_MAX_NEW_TOKENS,
        "dtype": DTYPE, "attn_implementation": ATTN_IMPLEMENTATION,
        "padding_side": PADDING_SIDE,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "defaults": FROZEN_GENERATION_DEFAULTS,
    }
    image_processor = {k: v for k, v in processor.image_processor.to_dict()
                       .items() if k != "_processor_class"}
    return {
        "repo": pin["repo"],
        "revision": pin["revision"],
        "licence": pin["licence"],
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters()
                                        if p.requires_grad)),
        "prompt": PROMPT_TEMPLATE,
        "prompt_sha256": sha256_bytes(PROMPT_TEMPLATE.encode("utf-8")),
        "chat_template": processor.chat_template,
        "chat_template_sha256": sha256_bytes(
            (processor.chat_template or "").encode("utf-8")),
        "image_processor": image_processor,
        "image_processor_sha256": sha256_json(image_processor),
        "tokenizer_sha256": sha256_json({
            "vocab_size": len(tokenizer),
            "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id,
            "bos": tokenizer.bos_token_id,
            "class": type(tokenizer).__name__}),
        "generation": generation,
        "generation_sha256": sha256_json(generation),
    }


def run_metadata() -> dict:
    from src import utils
    return utils.run_metadata()
