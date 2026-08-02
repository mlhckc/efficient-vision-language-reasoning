"""E8A shared implementation: the frozen-SLM question-encoder interface.

The canonical primary interface (master protocol section 8.0, E8A canonical
plan section 3) is sequence / token-level:

    question text
      -> SmolLM2-135M tokenisation
      -> frozen LM hidden states [B, L, H]
      -> valid-token selection and key-padding mask
      -> one trainable token-wise Linear(H, 512)
      -> projected sequence [B, L, 512]
      -> the unmodified latent-query reasoner
      -> answer classifier.

src/reasoner.py is imported unmodified and is never edited. The frozen
language model is never trained, never unfrozen and is held in its native
bfloat16 at all times (master protocol section 20).

Two arms are implemented here:

  A1   pretrained frozen SmolLM2-135M
  A1r  architecture-matched random SmolLM2-135M, built with from_config from
       the pretrained model's own config under a pinned seed, every parameter
       frozen, no pretrained weights loaded.

Within the pair, the reasoner trunk and the projection have identical initial
weights for a given training seed, because the construction order of gate G13
is followed exactly: load and freeze the LM, then utils.set_seed(seed), then
build the trunk from unmodified src/reasoner.py, then build the projection.

Every constant that is specific to this experiment is named here and recorded
in the result metadata, as CLAUDE.md requires. Constants inherited from the
v3_01 recipe are read from experiments/v3_01_reasoner/run.py at run time, not
copied, so gate G0 compares live values rather than a transcription.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from src.reasoner import LatentQueryReasoner  # noqa: E402

# --- Experiment-specific constants (E8A) ------------------------------------

MODEL_REPO = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_HIDDEN_SIZE = 576
MODEL_N_LAYERS = 30
MODEL_PARAMETERS = 134_515_008

# Master protocol section 2: the random control is constructed, not downloaded.
RANDOM_INIT_SEED = 20260802

# Master protocol section 4: the common question-token budget.
TOKEN_BUDGET_L = 32
D_MODEL = 512

# Master protocol section 18.1, gate G20. Pre-registered, not invented here.
G20_RELATIVE_L2_MAX = 1e-3

# v3_01 gate 2 / master protocol gate G5.
G5_MASK_TOLERANCE = 1e-4

# Master protocol section 14: operative for E8 execution. The historical
# v3_01 WALL_CLOCK_GATE_HOURS = 12.0 is non-operative and is asserted to be
# unused by gate G0.
WALL_CLOCK_HALT_HOURS = 8.0
MEMORY_CEILING_FRACTION = 0.80

# Master protocol section 11: the majority reference and the collapse
# criterion. All three thresholds are fixed before execution.
MAJORITY_REFERENCE = 0.22466
COLLAPSE_ENTROPY_MAX_NATS = 0.30
COLLAPSE_CLASS_SHARE_MIN = 0.60
QUESTION_INSENSITIVITY_MAX = 0.01

# Master protocol section 11.1: the pinned neutral question string.
NEUTRAL_QUESTION_STRING = "question"

# Paths. Canonical plan section 11 confines new code and outputs.
V2_DIR = config.DATA_DIR / "v2"
SLM_TOKEN_DIR = config.DATA_DIR / "v3_slm_tokens"
CLIP_TOKEN_DIR = config.DATA_DIR / "v3" / "tokens"
OUT_DIR = config.RESULTS_DIR / "experiments" / "e8a_question_encoder"

ARMS = {
    "A1": {"pretrained": True,
           "description": "pretrained frozen SmolLM2-135M question encoder"},
    "A1r": {"pretrained": False,
            "description": "architecture-matched random SmolLM2-135M, frozen"},
}

# A0p is the interface-matched CLIP question-token control of E8A plan section
# 2 and master protocol section 8.0: the same token-level CLIP question
# sequence the stored A0 reasoner consumes, plus one trainable Linear(512, 512)
# so that A1 - A0p differs in the question representation source and nothing
# else. A0p is NOT the stored A0 artefact and never reuses its checkpoint.
A0P_ARM = "A0p"
CLIP_QUESTION_WIDTH = 512
A0P_PROJECTION_PARAMETERS = 262_656

ARM_SPECS = {
    "A0p": {"encoder": "clip_question_tokens", "d_question": 512,
            "principal": True,
            "description": "interface-matched CLIP question-token control"},
    "A1": {"encoder": "slm_135m", "d_question": MODEL_HIDDEN_SIZE,
           "pretrained": True, "principal": True,
           "description": "pretrained frozen SmolLM2-135M question encoder"},
    "A1r": {"encoder": "slm_135m", "d_question": MODEL_HIDDEN_SIZE,
            "pretrained": False, "principal": False,
            "description": "architecture-matched random SmolLM2-135M, frozen"},
}


def arm_d_question(arm: str) -> int:
    return ARM_SPECS[arm]["d_question"]


def arm_is_principal(arm: str) -> bool:
    """G8 and G10 are halting for principal arms and recorded diagnostics for
    the random controls (master protocol section 11, E8A plan section 7)."""
    return ARM_SPECS[arm]["principal"]


def prepare_encoder_for_build(arm: str):
    """Discharge the G13 construction step 'load and freeze the encoder' for
    any arm, before `utils.set_seed(seed)` and the trunk are built.

    For the SLM arms this loads and freezes the language model. For A0p the
    frozen encoder is CLIP, which was run once by `v3_00` and whose states are
    read from the cached store, so there is nothing to load; the step is
    discharged by construction and recorded as such.
    """
    if ARM_SPECS[arm]["encoder"] == "slm_135m":
        model, provenance = load_frozen_lm(ARM_SPECS[arm]["pretrained"],
                                           verbose=False)
        del model
        return provenance
    return {
        "arm_model": "frozen CLIP ViT-B/32 question tokens, cached",
        "encoder": "clip_question_tokens",
        "clip_model": config.CLIP_MODEL_NAME,
        "clip_pretrained": config.CLIP_PRETRAINED,
        "store": str((CLIP_TOKEN_DIR / "question_tokens.h5")
                     .relative_to(PROJECT_ROOT)),
        "construction": (
            "no encoder is loaded at training time: the frozen CLIP question "
            "states were produced once by experiments/v3_00_tokens and are "
            "read from the cached store, exactly as arm A0 consumes them"),
        "all_parameters_frozen": True,
        "recipe_identifier": "7.1 fixed inherited v3_01 reasoner recipe",
        "projection_scale_applied": "none",
    }


def bind_store_to_encoder(arm: str, store, provenance: dict) -> str:
    """Assert that the question store was produced by this arm's encoder.

    Without this a store built from a different checkpoint, a different random
    seed, or with the arms swapped would be consumed silently.
    """
    if ARM_SPECS[arm]["encoder"] == "slm_135m":
        assert store.attrs["lm_state_dict_sha256"] == \
            provenance["state_dict_sha256"], (
                f"{arm}: the hidden-state store was not produced by the "
                f"language model loaded here")
        return provenance["state_dict_sha256"]
    assert store.attrs["ln_final_applied"] and \
        store.attrs["text_projection_applied"] and \
        not store.attrs["normalized"], store.attrs
    assert store.states.shape[1] == CLIP_QUESTION_WIDTH, store.states.shape
    return str(store.attrs["length_convention"])


def rows_used_by(store, manifests) -> np.ndarray:
    """The packed state rows this run actually reads, deduplicated and sorted.

    A store may cover more rows than a run consumes — the CLIP question store
    spans the union including the clean-test INPUT rows, which are excluded
    from every development quantity. Diagnostics must therefore be computed
    over the rows the run uses, not over the whole file.
    """
    wanted = set()
    for manifest in manifests:
        frame = pd.read_csv(manifest, dtype={"questionId": str},
                            keep_default_na=False)
        for qid in frame["questionId"]:
            offset, length = store.span(qid)
            wanted.update(range(offset, offset + length))
    return np.array(sorted(wanted), dtype="int64")


def rms_over_rows(states: np.ndarray, rows: np.ndarray,
                  chunk: int = 65536) -> float:
    """Root mean square over selected rows, in chunks and in float64."""
    total, count = 0.0, 0
    for start in range(0, len(rows), chunk):
        block = states[rows[start:start + chunk]].astype(np.float64)
        total += float((block ** 2).sum())
        count += block.size
    return round(float(np.sqrt(total / count)), 5)


def frozen_encoder_parameters(arm: str) -> dict:
    """The frozen encoder's parameter count for this arm, measured.

    Hard-coding the SmolLM2 count for every arm would attribute 134.5M frozen
    parameters to A0p, which loads no language model at all.
    """
    if ARM_SPECS[arm]["encoder"] == "slm_135m":
        return {"encoder": "SmolLM2-135M, frozen",
                "parameters": MODEL_PARAMETERS,
                "basis": "asserted against the canonical table at load time"}
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED)
    text_path = [model.token_embedding.weight, model.positional_embedding,
                 model.text_projection, *model.ln_final.parameters(),
                 *model.transformer.parameters()]
    if getattr(model, "attn_mask", None) is not None:
        pass  # a buffer, not a parameter
    count = int(sum(p.numel() for p in text_path))
    trainable = int(sum(p.numel() for p in model.parameters()
                        if p.requires_grad))
    del model
    return {"encoder": f"CLIP {config.CLIP_MODEL_NAME} text tower, frozen",
            "parameters": count,
            "trainable_in_the_loaded_clip_model": trainable,
            "basis": "measured over token_embedding, positional_embedding, "
                     "transformer, ln_final and text_projection, which is the "
                     "path v3_00 ran to build the question-token store"}


def open_clip_question_store() -> "SLMQuestionStore":
    """The cached frozen-CLIP question tokens, in the same packed layout.

    `data/v3/tokens/question_tokens.h5` stores `ids` / `offsets` / `lengths` /
    `tokens`, which is structurally identical to the SLM stores' `ids` /
    `offsets` / `lengths` / `states`. Adapting it here rather than writing a
    second dataset class means A0p runs the same dataset, collate,
    intervention and evaluation code as A1, which is the strongest available
    form of interface matching. The store is opened read-only (gate G12).
    """
    path = CLIP_TOKEN_DIR / "question_tokens.h5"
    with h5py.File(path, "r") as store:
        ids = [i.decode("utf-8") if isinstance(i, bytes) else str(i)
               for i in store["ids"][:]]
        states = store["tokens"][:]
        offsets = store["offsets"][:]
        lengths = store["lengths"][:]
        attrs = {k: (v.item() if hasattr(v, "item") else v)
                 for k, v in store.attrs.items()}
    attrs["arm"] = A0P_ARM
    attrs["layer_rule"] = (
        "CLIP ln_final then text_projection, per-token; the v3_00 approach-A "
        "token states, unnormalised")
    attrs["source_path"] = str(path.relative_to(PROJECT_ROOT))
    attrs["source_sha256"] = "not hashed here; pinned by G15 at run time"
    return SLMQuestionStore(states=states, offsets=offsets, lengths=lengths,
                            row_of={q: i for i, q in enumerate(ids)},
                            attrs=attrs)


def build_neutral_question_states_clip(device) -> tuple:
    """The pinned neutral question sequence for A0p, shape (L, 512), fp16.

    Master protocol section 11.1 defines it as the CLIP or SLM token states of
    the fixed string "question". For A0p that is the CLIP states, reproduced
    through exactly the frozen text path `v3_00` used to build the store:
    token embedding, positional embedding, the transformer under its causal
    attention mask, `ln_final`, then `text_projection`, stored unnormalised,
    with the true length taken as the EOT position plus one and therefore
    including SOT and EOT.
    """
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert model.text_pool_type == "argmax"
    assert tuple(model.text_projection.shape) == (512, 512)

    token_ids = tokenizer([NEUTRAL_QUESTION_STRING])
    eot = int(token_ids.argmax(dim=-1))
    valid = eot + 1
    assert 0 < valid <= TOKEN_BUDGET_L, valid
    with torch.no_grad():
        cast_dtype = model.transformer.get_cast_dtype()
        x = model.token_embedding(token_ids.to(device)).to(cast_dtype)
        x = x + model.positional_embedding.to(cast_dtype)
        x = model.transformer(x, attn_mask=model.attn_mask)
        x = model.ln_final(x)
        states = (x @ model.text_projection)[0, :valid]

    padded = np.zeros((TOKEN_BUDGET_L, CLIP_QUESTION_WIDTH), dtype=np.float16)
    padded[:valid] = states.float().cpu().numpy().astype(np.float16)
    mask = np.ones(TOKEN_BUDGET_L, dtype=bool)
    mask[:valid] = False
    provenance = {
        "definition": 'frozen-CLIP per-token states of the fixed string '
                      f'"{NEUTRAL_QUESTION_STRING}", padded to L',
        "string": NEUTRAL_QUESTION_STRING,
        "token_ids": token_ids[0, :valid].tolist(),
        "valid_positions": valid,
        "length_convention": "EOT position + 1, including SOT and EOT",
        "L": TOKEN_BUDGET_L,
        "shape": list(padded.shape),
        "dtype": "float16",
        "sha256": sha256_array(padded),
        "mask_sha256": sha256_array(mask),
        "no_fully_masked_row": bool((~mask).any()),
        "path": "token embedding + positional, transformer under the causal "
                "attn_mask, ln_final, text_projection; unnormalised; "
                "identical to experiments/v3_00_tokens/extract_tokens.py "
                "text_tokens_batch",
    }
    del model
    torch.cuda.empty_cache()
    return padded, mask, provenance

STRING_DTYPE = h5py.special_dtype(vlen=str)


# --- Recipe (7.1), read live from v3_01 -------------------------------------

EXPECTED_FIXED_RECIPE = {"batch_size": 128, "weight_decay": 1e-2,
                         "grad_clip": 1.0, "patience": 10,
                         "final_max_epochs": 100}
HISTORICAL_NON_OPERATIVE_WALL_CLOCK = 12.0


def load_v3_01():
    """Import experiments/v3_01_reasoner/run.py without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "v3_01_run", PROJECT_ROOT / "experiments" / "v3_01_reasoner" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate_g0_recipe(verbose: bool = True) -> dict:
    """G0: recipe identity against the live v3_01 constants and selected config.

    Returns the complete recipe actually used, including the three values that
    are NOT module constants (lr, warmup_frac, dropout) but the outcome of
    v3_01's own dev-accuracy search, recorded as selected_config.
    """
    v301 = load_v3_01()
    live = {"batch_size": v301.BATCH_SIZE, "weight_decay": v301.WEIGHT_DECAY,
            "grad_clip": v301.GRAD_CLIP, "patience": v301.PATIENCE,
            "final_max_epochs": v301.FINAL_MAX_EPOCHS}
    assert live == EXPECTED_FIXED_RECIPE, (live, EXPECTED_FIXED_RECIPE)
    assert v301.WALL_CLOCK_GATE_HOURS == HISTORICAL_NON_OPERATIVE_WALL_CLOCK, \
        v301.WALL_CLOCK_GATE_HOURS

    stored_path = (config.RESULTS_DIR / "experiments" / "v3_01_reasoner"
                   / "results.json")
    stored = json.loads(stored_path.read_text())["v3_01_reasoner"]
    selected = stored["selected_config"]
    assert set(selected) == {"lr", "warmup_frac", "dropout"}, selected
    assert selected == {"lr": 3e-4, "warmup_frac": 0.0, "dropout": 0.1}, selected

    recipe = {
        "recipe_identifier": "7.1 fixed inherited v3_01 reasoner recipe",
        "optimizer": "AdamW, two parameter groups",
        "weight_decay": v301.WEIGHT_DECAY,
        "weight_decay_applies_to": "ndim >= 2 only; 0.0 on ndim < 2",
        "batch_size": v301.BATCH_SIZE,
        "learning_rate": selected["lr"],
        "warmup_frac": selected["warmup_frac"],
        "dropout": selected["dropout"],
        "grad_clip": v301.GRAD_CLIP,
        "patience": v301.PATIENCE,
        "final_max_epochs": v301.FINAL_MAX_EPOCHS,
        "schedule": "cosine after warmup, LambdaLR",
        "schedule_horizon": "final_max_epochs x steps_per_epoch",
        "schedule_step_frequency": "per optimizer step",
        "precision": "bf16 autocast on the training path only; fp32 evaluation",
        "checkpoint_selection": "best development accuracy",
        "tie_break": "earliest epoch attaining the best value",
        "wall_clock_halt_hours_operative": WALL_CLOCK_HALT_HOURS,
        "wall_clock_gate_hours_v3_01_historical_non_operative":
            v301.WALL_CLOCK_GATE_HOURS,
        "source": "experiments/v3_01_reasoner/run.py (live constants) and its "
                  "stored results.json selected_config",
        "stored_results_sha256": sha256_file(stored_path),
    }
    if verbose:
        print("=== GATE G0: recipe identity (7.1) ===")
        print(f"[PASS] live v3_01 constants match {EXPECTED_FIXED_RECIPE}")
        print(f"[PASS] selected_config {selected}")
        print(f"[PASS] v3_01 WALL_CLOCK_GATE_HOURS is "
              f"{v301.WALL_CLOCK_GATE_HOURS} and is NOT used; the operative "
              f"halt is {WALL_CLOCK_HALT_HOURS} GPU-hours")
    return recipe


# --- Small helpers -----------------------------------------------------------

def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sha256_state_dict(state: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        tensor = state[key].detach().cpu()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(np.ascontiguousarray(
            tensor.view(torch.uint8).numpy()
            if tensor.dtype == torch.bfloat16
            else tensor.numpy()).tobytes())
    return digest.hexdigest()


def model_snapshot_dir() -> Path:
    """The resolved local snapshot of the pinned checkpoint."""
    import os
    root = Path(os.environ.get("HF_HOME", config.CACHE_DIR / "huggingface"))
    return (root / "hub" / ("models--" + MODEL_REPO.replace("/", "--"))
            / "snapshots" / MODEL_REVISION)


def assert_no_clean_test_path(text: str, where: str) -> None:
    """G17 helper: no clean-test target path may be resolvable in E8A code."""
    forbidden = "test_" + "clean_targets"
    if forbidden in text:
        raise AssertionError(f"{where} references the embargoed clean-test path")


# --- Frozen language model ---------------------------------------------------

def load_frozen_lm(pretrained: bool, device=None, verbose: bool = True):
    """Load the frozen SmolLM2-135M encoder for A1 (pretrained) or A1r (random).

    A1  : from_pretrained at the pinned revision, native bfloat16.
    A1r : AutoConfig from the same pinned revision, utils.set_seed(20260802)
          immediately before construction, AutoModel.from_config in float32 so
          the initialisation is the PyTorch-defined one and depends only on the
          seed, then cast to bfloat16 so both members of the pair are held in
          the same dtype. No pretrained weight is loaded.

    Every parameter is frozen and the module is put in eval(). Returns
    (model, provenance dict).
    """
    import transformers
    import tokenizers
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    snapshot = model_snapshot_dir()
    if pretrained:
        model = AutoModel.from_pretrained(MODEL_REPO, revision=MODEL_REVISION,
                                          dtype=torch.bfloat16)
        provenance = {
            "arm_model": "pretrained",
            "repo_id": MODEL_REPO,
            "pinned_revision": MODEL_REVISION,
            "construction": "AutoModel.from_pretrained at the pinned revision",
        }
    else:
        utils.set_seed(RANDOM_INIT_SEED)
        model = AutoModel.from_config(cfg, dtype=torch.float32)
        model = model.to(torch.bfloat16)
        provenance = {
            "arm_model": "random",
            "repo_id": MODEL_REPO,
            "config_from_revision": MODEL_REVISION,
            "construction": (
                "utils.set_seed(20260802); AutoModel.from_config(cfg, "
                "dtype=torch.float32); .to(torch.bfloat16). No pretrained "
                "weight is loaded. Pinned by this seed plus the config "
                "SHA-256."),
            "random_init_seed": RANDOM_INIT_SEED,
        }

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if device is not None:
        model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    assert total == MODEL_PARAMETERS, total
    assert cfg.hidden_size == MODEL_HIDDEN_SIZE
    assert cfg.num_hidden_layers == MODEL_N_LAYERS
    assert not any(p.requires_grad for p in model.parameters())
    assert not model.training

    provenance.update({
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "vocab_size": cfg.vocab_size,
        "parameter_count": int(total),
        "loaded_dtype": str(next(model.parameters()).dtype),
        "all_parameters_frozen": True,
        "module_in_eval": True,
        "state_dict_sha256": sha256_state_dict(model.state_dict()),
        "extraction_layer_rule": (
            "last_hidden_state, the post-final-norm output; never "
            "hidden_states[-1] used interchangeably"),
        # Canonical section 20 provenance, and section 2's requirement that
        # the random control be pinned by its seed PLUS the SHA-256 of the
        # config it was built from.
        "resolved_revision": getattr(cfg, "_commit_hash", None) or snapshot.name,
        "resolved_snapshot_path": str(snapshot),
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "config_sha256": sha256_file(snapshot / "config.json"),
        "tokenizer_file_sha256": sha256_file(snapshot / "tokenizer.json"),
        "tokenizer_config_sha256": sha256_file(snapshot / "tokenizer_config.json"),
        "config_fields": {
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "intermediate_size": cfg.intermediate_size,
            "vocab_size": cfg.vocab_size,
            "rms_norm_eps": cfg.rms_norm_eps,
            "tie_word_embeddings": cfg.tie_word_embeddings,
            "initializer_range": cfg.initializer_range,
        },
        "projection_scale_applied": (
            "none. Canonical section 9.1: no model-specific RMS rescaling is "
            "applied in E8A, because ReasonerBlock LayerNorms the question "
            "source and a constant rescale would be a no-op."),
        "recipe_identifier": "7.1 fixed inherited v3_01 reasoner recipe",
    })
    if verbose:
        print(f"[G2/G3] {provenance['arm_model']} SmolLM2-135M: "
              f"{total:,} parameters, dtype "
              f"{provenance['loaded_dtype']}, all frozen, eval()")
    return model, provenance


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)


def tokenize_questions(tokenizer, strings):
    """Tokenise question strings.

    The SmolLM2 tokenizer has add_bos_token and add_eos_token both False, so
    add_special_tokens=True and add_special_tokens=False are byte-identical for
    these strings and the question sequence contains no special token at all.
    That is asserted here rather than assumed, so a tokenizer change cannot
    silently alter the interface.
    """
    with_special = tokenizer(list(strings), add_special_tokens=True)["input_ids"]
    without = tokenizer(list(strings), add_special_tokens=False)["input_ids"]
    assert with_special == without, (
        "the tokenizer adds special tokens; the canonical E8A interface "
        "retains all valid question-token positions and the EOS treatment "
        "must then be stated explicitly rather than measured as vacuous")
    return with_special


# --- Hidden-state store ------------------------------------------------------

@dataclass
class SLMQuestionStore:
    """Packed frozen-LM question states, keyed by questionId.

    Layout mirrors data/v3/tokens/question_tokens.h5 exactly: identical
    question strings share one packed block, and every questionId carries an
    (offset, length) index into it.
    """

    states: np.ndarray          # (T, H) float16
    offsets: np.ndarray         # (n_ids,) int64
    lengths: np.ndarray         # (n_ids,) int32
    row_of: dict                # questionId -> index
    attrs: dict

    @classmethod
    def open(cls, path):
        with h5py.File(path, "r") as store:
            ids = [i.decode("utf-8") if isinstance(i, bytes) else str(i)
                   for i in store["ids"][:]]
            states = store["states"][:]
            offsets = store["offsets"][:]
            lengths = store["lengths"][:]
            attrs = {k: (v.item() if hasattr(v, "item") else v)
                     for k, v in store.attrs.items()}
        return cls(states=states, offsets=offsets, lengths=lengths,
                   row_of={qid: i for i, qid in enumerate(ids)}, attrs=attrs)

    def span(self, question_id: str):
        index = self.row_of[question_id]
        return int(self.offsets[index]), int(self.lengths[index])


class ImageTokenStore:
    """The cached frozen-CLIP image tokens, opened read-only (gate G12).

    restrict_to loads only the named imageIds, which lets the test module work
    on a small deterministic sample without materialising the 3.26 GB store.
    """

    def __init__(self, path=None, restrict_to=None):
        path = Path(path or CLIP_TOKEN_DIR / "image_tokens.h5")
        with h5py.File(path, "r") as store:
            ids = [i.decode("utf-8") if isinstance(i, bytes) else str(i)
                   for i in store["ids"][:]]
            if restrict_to is None:
                self.tokens = store["tokens"][:]
                kept = ids
            else:
                index_of = {image_id: i for i, image_id in enumerate(ids)}
                kept = sorted(set(restrict_to))
                rows = np.array([index_of[i] for i in kept])
                order = np.argsort(rows)
                selected = store["tokens"][rows[order]]
                self.tokens = np.empty_like(selected)
                self.tokens[order] = selected
        self.row_of = {image_id: i for i, image_id in enumerate(kept)}
        self.path = path


class E8ATokenDataset(Dataset):
    """Manifest rows served as (image tokens, frozen-LM question states, ...).

    Labels come from the V2 manifest only. The embargoed clean-test target
    file is never opened by this module. Its name is deliberately not written
    anywhere in this package, so that the gate-G17 scan can be a plain
    substring test with no exemption for "mentions in comments".
    """

    def __init__(self, manifest_path, images: ImageTokenStore,
                 questions: SLMQuestionStore):
        frame = pd.read_csv(manifest_path,
                            dtype={"questionId": str, "imageId": str},
                            keep_default_na=False)
        if "label" not in frame.columns:
            raise ValueError(f"{manifest_path} has no label column")
        self.images = images
        self.questions = questions
        self.image_rows = np.array([images.row_of[i] for i in frame["imageId"]])
        spans = [questions.span(q) for q in frame["questionId"]]
        self.question_offsets = np.array([s[0] for s in spans], dtype="int64")
        self.question_lengths = np.array([s[1] for s in spans], dtype="int64")
        self.labels = frame["label"].to_numpy("int64")
        self.question_ids = frame["questionId"].tolist()
        self.image_ids = frame["imageId"].tolist()
        self.original_image_rows = self.image_rows.copy()
        # Interventions replace INPUTS only; no module is modified, no mask
        # argument is added and src/reasoner.py is untouched.
        self.fixed_image = None            # (50, 512) float32 or None
        self.fixed_question = None         # (L, H) float32 or None
        self.fixed_question_mask = None    # (L,) bool, True marks padding

    def __len__(self):
        return len(self.image_rows)

    def __getitem__(self, index):
        if self.fixed_image is None:
            image = torch.from_numpy(
                self.images.tokens[self.image_rows[index]].astype(np.float32))
        else:
            image = self.fixed_image
        if self.fixed_question is None:
            offset = int(self.question_offsets[index])
            length = int(self.question_lengths[index])
            question = torch.from_numpy(
                self.questions.states[offset:offset + length]
                .astype(np.float32))
            mask = torch.zeros(length, dtype=torch.bool)
        else:
            question = self.fixed_question
            mask = self.fixed_question_mask
        return image, question, mask, int(self.labels[index])

    # --- intervention setters, all pinned inputs -------------------------

    def set_normal(self):
        self.fixed_image = None
        self.fixed_question = None
        self.fixed_question_mask = None
        self.image_rows = self.original_image_rows.copy()

    def set_fixed_image(self, neutral_image: np.ndarray):
        self.fixed_image = torch.from_numpy(
            neutral_image.astype(np.float32).copy())

    def set_fixed_question(self, neutral_states: np.ndarray,
                           neutral_mask: np.ndarray):
        self.fixed_question = torch.from_numpy(
            neutral_states.astype(np.float32).copy())
        self.fixed_question_mask = torch.from_numpy(neutral_mask.copy())

    def set_shuffled_images_by_imageid(self, mapping: dict):
        self.image_rows = np.array(
            [self.images.row_of[mapping[i]] for i in self.image_ids])

    def set_shuffled_images_by_row(self, permutation: np.ndarray):
        self.image_rows = self.original_image_rows[permutation]

    def self_pair_count(self) -> int:
        return int((self.image_rows == self.original_image_rows).sum())


def collate_e8a(batch):
    """Pad question states to the batch maximum.

    Mask convention, identical to src/tokens_data.collate_tokens and to the
    contract in src/reasoner.py: True marks a PADDED position that attention
    must ignore; False marks a valid token. Each item carries its own mask, so
    the pinned neutral question sequence keeps the padding pattern the
    canonical protocol fixes for it instead of having one inferred from a
    length.
    """
    images = torch.stack([item[0] for item in batch])
    item_lengths = [item[1].shape[0] for item in batch]
    lengths = torch.tensor([int((~item[2]).sum()) for item in batch],
                           dtype=torch.long)
    max_length = max(item_lengths)
    hidden = batch[0][1].shape[-1]
    questions = torch.zeros(len(batch), max_length, hidden)
    key_padding_mask = torch.ones(len(batch), max_length, dtype=torch.bool)
    for row, item in enumerate(batch):
        n = item_lengths[row]
        questions[row, :n] = item[1]
        key_padding_mask[row, :n] = item[2]
    assert bool((~key_padding_mask).any(dim=1).all()), \
        "an attention row would be fully masked"
    labels = torch.tensor([item[3] for item in batch], dtype=torch.long)
    return images, questions, lengths, key_padding_mask, labels


def make_loaders(train_manifest, dev_manifest, images, questions,
                 batch_size: int, seed: int):
    train_dataset = E8ATokenDataset(train_manifest, images, questions)
    dev_dataset = E8ATokenDataset(dev_manifest, images, questions)
    pin_memory = config.DEVICE == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=utils.make_generator(seed), worker_init_fn=utils.seed_worker,
        num_workers=0, pin_memory=pin_memory, collate_fn=collate_e8a)
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=pin_memory, collate_fn=collate_e8a)
    return train_loader, dev_loader


# --- The E8A model -----------------------------------------------------------

class E8AQuestionEncoderReasoner(nn.Module):
    """One trainable token-wise Linear(H, 512) in front of the unmodified trunk.

    The frozen language model is deliberately NOT a submodule: training reads
    cached hidden states, and keeping the LM out of this module guarantees that
    no frozen parameter can reach the optimizer or gradient clipping.
    """

    def __init__(self, projection: nn.Linear, trunk: LatentQueryReasoner):
        super().__init__()
        self.projection = projection
        self.trunk = trunk

    def forward(self, image_tokens, question_states, question_mask):
        projected = self.projection(question_states)
        return self.trunk(image_tokens, projected, question_mask)


def build_e8a_model(d_lm: int, dropout: float, seed: int):
    """Build the trainable part in the canonical G13 construction order.

    The caller must already have loaded and frozen the language model. This
    function then performs, in order: utils.set_seed(seed); build the trunk
    from the unmodified src/reasoner.py; build the projection. Nothing between
    those steps consumes randomness, so the trunk state is bitwise identical to
    a freshly seeded unmodified LatentQueryReasoner and both trunk and
    projection are bitwise identical across the A1/A1r pair at a given seed.
    """
    utils.set_seed(seed)
    trunk = LatentQueryReasoner(dropout=dropout)
    projection = nn.Linear(d_lm, D_MODEL)
    return E8AQuestionEncoderReasoner(projection, trunk)


def parameter_report(model: nn.Module, frozen_lm: nn.Module | None = None):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters()
                        if not p.requires_grad)
    report = {
        "trainable_projection": sum(p.numel()
                                    for p in model.projection.parameters()),
        "trainable_reasoner_trunk": sum(p.numel()
                                        for p in model.trunk.parameters()),
        "trainable_total": int(trainable),
        "non_trainable_inside_trained_module": int(non_trainable),
    }
    if frozen_lm is not None:
        frozen = sum(p.numel() for p in frozen_lm.parameters())
        report["frozen_language_model"] = int(frozen)
        report["total_loaded_parameters"] = int(trainable + non_trainable
                                                + frozen)
    return report


# --- Optimiser and schedule, taken from v3_01 unchanged ----------------------

def make_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for _, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def selection_step(accuracy, best_accuracy, best_epoch, epoch,
                   without_improvement, patience):
    """One epoch of the checkpoint-selection and early-stopping decision.

    Extracted as a pure function so it can be driven by a test over a stubbed
    accuracy sequence. The behaviour is that of
    experiments/v3_01_reasoner/run.py:152-162: a strict `>` comparison, so the
    EARLIEST epoch attaining the best value wins the tie-break, and patience
    counts consecutive epochs without an improvement.

    Returns (best_accuracy, best_epoch, without_improvement, improved, stop).
    """
    if accuracy > best_accuracy:
        return accuracy, epoch, 0, True, False
    without_improvement += 1
    return (best_accuracy, best_epoch, without_improvement, False,
            without_improvement >= patience)


def make_scheduler(optimizer, total_steps, warmup_frac):
    import math
    warmup_steps = int(round(warmup_frac * total_steps))

    def factor(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# --- Evaluation --------------------------------------------------------------

@torch.no_grad()
def predict_logits(model, loader, device):
    """fp32 evaluation: the trainable modules run outside autocast."""
    model.eval()
    logits, labels = [], []
    for images, questions, _lengths, mask, batch_labels in loader:
        out = model(images.to(device), questions.to(device), mask.to(device))
        logits.append(out.float().cpu())
        labels.append(batch_labels)
    return torch.cat(logits), torch.cat(labels)


def classification_diagnostics(logits: torch.Tensor, labels: torch.Tensor,
                               n_classes: int = config.TOP_K_ANSWERS) -> dict:
    """Accuracy and the section-11 degeneracy quantities, all in fp32.

    prediction entropy  H = mean_n( - sum_k p_n[k] ln p_n[k] ), in nats.
    maximum class share = the largest fraction of rows sharing one argmax.
    constant-answer rate: the canonical set requires this quantity but defines
    it nowhere (round-10 finding SL10-1). The round-10 scientific reviewer's
    own suggested clause is adopted and disclosed: the fraction of rows on
    which the model emits its single most frequent answer. Under that reading
    it equals maximum class share by construction, so both are reported and
    the redundancy is visible rather than hidden.
    """
    probabilities = torch.softmax(logits.double(), dim=-1)
    entropy = float((-(probabilities
                       * torch.log(probabilities.clamp_min(1e-300)))
                     ).sum(dim=-1).mean())
    predictions = logits.argmax(dim=-1)
    counts = torch.bincount(predictions, minlength=n_classes)
    max_share = float(counts.max()) / len(predictions)
    return {
        "n_rows": int(len(predictions)),
        "accuracy": round(float((predictions == labels).double().mean()), 5),
        "prediction_entropy_nats": round(entropy, 5),
        "maximum_class_share": round(max_share, 5),
        "constant_answer_rate": round(max_share, 5),
        "constant_answer_rate_definition": (
            "the fraction of development rows on which the model emits its "
            "single most frequent answer. The canonical set requires this "
            "quantity but defines it nowhere (round-10 finding SL10-1); the "
            "round-10 scientific reviewer's own suggested clause is adopted "
            "and disclosed. Under it the value equals maximum_class_share by "
            "construction, so both are reported and the redundancy is "
            "visible. No decision rule consumes it."),
        "distinct_answers_predicted": int((counts > 0).sum()),
        "majority_reference": MAJORITY_REFERENCE,
    }


def collapse_fires(diagnostics: dict) -> bool:
    return (diagnostics["prediction_entropy_nats"]
            <= COLLAPSE_ENTROPY_MAX_NATS
            or diagnostics["maximum_class_share"]
            >= COLLAPSE_CLASS_SHARE_MIN)


# --- Pinned intervention tensors --------------------------------------------

def build_neutral_image_tokens(images: ImageTokenStore,
                               training_manifest=None) -> tuple:
    """The pinned neutral image-token sequence, shape (50, 512), fp16.

    Master protocol section 11.1 defines it as the element-wise mean of the
    cached CLIP image-token store over all training images. The store also
    contains development and clean-test *input* images (63,599 rows against
    61,859 training images), so the mean must be restricted to the training
    image set rather than taken over the whole store. The training image set is
    read as the unique imageIds of data/v2/train_250k.csv, the complete
    training-image set of the programme and a superset of train_40k's, so the
    pinned tensor is one programme-wide constant that does not change between
    scales. Only the imageId column is read; no label, and no clean-test file.
    """
    manifest = Path(training_manifest or V2_DIR / "train_250k.csv")
    frame = pd.read_csv(manifest, usecols=["imageId"], dtype={"imageId": str},
                        keep_default_na=False)
    unique_images = sorted(set(frame["imageId"]))
    rows = np.array([images.row_of[i] for i in unique_images])
    rows.sort()
    accumulator = np.zeros((50, 512), dtype=np.float64)
    chunk = 4096
    for start in range(0, len(rows), chunk):
        block = images.tokens[rows[start:start + chunk]].astype(np.float64)
        accumulator += block.sum(axis=0)
    mean = (accumulator / len(rows)).astype(np.float16)
    provenance = {
        "definition": "element-wise mean of the cached CLIP image-token store "
                      "over the unique training imageIds of "
                      "data/v2/train_250k.csv",
        "manifest": str(manifest.relative_to(PROJECT_ROOT)),
        "manifest_sha256": sha256_file(manifest),
        "n_training_images": len(unique_images),
        "n_rows_in_store": len(images.row_of),
        "shape": list(mean.shape),
        "dtype": "float16",
        "sha256": sha256_array(mean),
    }
    return mean, provenance


def build_neutral_question_states(lm, tokenizer, device) -> tuple:
    """The pinned neutral question sequence for one arm, shape (L, H), fp16.

    Master protocol section 11.1: the SLM token states of the fixed string
    "question", padded to L with a valid key-padding mask marking exactly the
    padded positions. At least one position is non-padding, so no attention row
    is fully masked. The tensor is arm-specific because the states come from
    that arm's own frozen encoder.
    """
    ids = tokenize_questions(tokenizer, [NEUTRAL_QUESTION_STRING])[0]
    valid = len(ids)
    assert 0 < valid <= TOKEN_BUDGET_L, valid
    input_ids = torch.tensor([ids], device=device)
    attention = torch.ones_like(input_ids)
    with torch.no_grad():
        states = lm(input_ids=input_ids,
                    attention_mask=attention).last_hidden_state[0]
    padded = np.zeros((TOKEN_BUDGET_L, states.shape[-1]), dtype=np.float16)
    padded[:valid] = states.float().cpu().numpy().astype(np.float16)
    mask = np.ones(TOKEN_BUDGET_L, dtype=bool)
    mask[:valid] = False
    provenance = {
        "definition": 'frozen-LM last_hidden_state of the fixed string '
                      f'"{NEUTRAL_QUESTION_STRING}", padded to L',
        "string": NEUTRAL_QUESTION_STRING,
        "token_ids": ids,
        "valid_positions": valid,
        "L": TOKEN_BUDGET_L,
        "shape": list(padded.shape),
        "dtype": "float16",
        "sha256": sha256_array(padded),
        "mask_sha256": sha256_array(mask),
        "no_fully_masked_row": bool((~mask).any()),
    }
    return padded, mask, provenance


def assert_derangement(mapping: dict) -> int:
    """Validate an imageId map: a bijection of the image set with no fixed
    point. Returns the self-pair count, which is zero when it passes."""
    if sorted(mapping.values()) != sorted(mapping.keys()):
        raise AssertionError("the image map is not a bijection")
    self_pairs = sum(1 for key, value in mapping.items() if key == value)
    if self_pairs:
        raise AssertionError(f"the image map has {self_pairs} self-pairs")
    return self_pairs


def imageid_level_derangement(image_ids, seed: int = config.RANDOM_SEED):
    """A pinned imageId-level derangement with zero self-pairs.

    Master protocol section 11.1 names the shuffled-image condition "the fixed
    imageId-level permutation"; the Phase-1A instruction requires zero
    self-pairs. A derangement satisfies both, and section 19's requirement that
    the observed self-pair count be reported is satisfied by reporting zero.
    Permutations are drawn from a pinned RNG and rejected until one has no
    fixed point, so the result depends only on the seed.
    """
    unique = sorted(set(image_ids))
    rng = np.random.default_rng(seed)
    draws = 0
    while True:
        draws += 1
        permuted = rng.permutation(len(unique))
        if not (permuted == np.arange(len(unique))).any():
            break
        if draws > 10_000:
            raise RuntimeError("no derangement found")
    mapping = {unique[i]: unique[int(permuted[i])] for i in range(len(unique))}
    self_pairs = sum(1 for k, v in mapping.items() if k == v)
    provenance = {
        "construction": "rejection-sampled permutation of the sorted unique "
                        "imageIds until it has no fixed point",
        "rng": f"np.random.default_rng({seed})",
        "seed": seed,
        "draws_until_derangement": draws,
        "n_images": len(unique),
        "self_pairs": self_pairs,
        "sha256": hashlib.sha256(
            json.dumps(mapping, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    assert self_pairs == 0
    return mapping, provenance


def row_level_permutation(n_rows: int, seed: int = config.RANDOM_SEED):
    """The v3_01-comparable row-level permutation, self-pairs reported.

    Reproduces experiments/v3_01_reasoner/run.py:487 exactly, so the pilot's
    shuffled-image drop is comparable with the stored v3_01 reference
    (0.54641 normal, 0.43635 shuffled, drop 0.11006, permutation seed 42),
    which is a single seed-0 run and is labelled as such wherever it is cited.
    """
    permutation = np.random.default_rng(seed).permutation(n_rows)
    return permutation, {
        "construction": "np.random.default_rng(config.RANDOM_SEED)"
                        ".permutation(n_rows), identical to "
                        "experiments/v3_01_reasoner/run.py:487",
        "seed": seed,
        "n_rows": int(n_rows),
        "sha256": sha256_array(permutation),
    }
