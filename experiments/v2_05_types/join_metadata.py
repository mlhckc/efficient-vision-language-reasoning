"""v2_05 part 1: re-join GQA question metadata onto the V2 dev and train_40k
manifests, and build the per-type prior baselines.

From the raw GQA train_balanced and val_balanced JSONs this extracts, per
questionId: the structural type (types.structural), the semantic type
(types.semantic) and the number of semantic program steps (len(semantic)).
The metadata is joined onto dev.csv and train_40k.csv by string questionId;
any join miss is a hard failure. Metadata files are written for dev and
train_40k only; no metadata file is built for the clean test at this stage,
and no clean-test statistic of any kind is computed (question-type statistics
of the clean test are embargoed).

Per-type priors (the VQA/GQA-paper prior methodology): from the joined
train_40k rows, the most frequent answer per structural type and per semantic
type; each prior predictor is evaluated on dev, overall and per type, and
reported alongside the global train_40k majority.

Analysis only; no training.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
META_DIR = V2_DIR / "metadata"
OUT_DIR = config.RESULTS_DIR / "experiments" / "v2_05_types"


def load_type_lookup() -> dict:
    """questionId -> (structural, semantic, n_steps) from both raw files."""
    lookup = {}
    for filename in (config.GQA_TRAIN_QUESTIONS, config.GQA_VAL_QUESTIONS):
        path = config.GQA_RAW_DIR / filename
        print(f"loading {filename}")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        for question_id, entry in data.items():
            lookup[str(question_id)] = (entry["types"]["structural"],
                                        entry["types"]["semantic"],
                                        len(entry["semantic"]))
        del data
    return lookup


def read_manifest(name: str) -> pd.DataFrame:
    return pd.read_csv(V2_DIR / f"{name}.csv",
                       dtype={"questionId": str, "imageId": str},
                       keep_default_na=False)


def join_types(frame: pd.DataFrame, lookup: dict, name: str) -> pd.DataFrame:
    missing = [q for q in frame["questionId"] if q not in lookup]
    matched = len(frame) - len(missing)
    print(f"[{'PASS' if not missing else 'FAIL'}] {name} join: "
          f"{matched}/{len(frame)} matched")
    if missing:
        sys.exit(f"FAIL: {len(missing)} unmatched questionIds in {name}, "
                 f"examples {missing[:10]}")
    rows = [lookup[q] for q in frame["questionId"]]
    return pd.DataFrame({
        "questionId": frame["questionId"],
        "structural": [r[0] for r in rows],
        "semantic": [r[1] for r in rows],
        "n_steps": [r[2] for r in rows],
    })


def distribution(types: pd.DataFrame) -> dict:
    return {
        "structural_counts": dict(sorted(Counter(types["structural"]).items())),
        "semantic_counts": dict(sorted(Counter(types["semantic"]).items())),
        "n_steps_histogram": {str(k): int(v) for k, v in
                              sorted(Counter(types["n_steps"]).items())},
    }


def per_type_prior(train_frame, train_types, dev_frame, dev_types, column):
    """Most frequent train_40k answer per type; evaluated on dev."""
    priors = {}
    for type_value, group in train_frame.groupby(train_types[column]):
        counts = Counter(group["answer"])
        priors[type_value] = sorted(counts.items(),
                                    key=lambda kv: (-kv[1], kv[0]))[0][0]
    dev_prediction = [priors.get(t) for t in dev_types[column]]
    correct = (pd.Series(dev_prediction).to_numpy()
               == dev_frame["answer"].to_numpy())
    per_type = {}
    for type_value in sorted(set(dev_types[column])):
        mask = (dev_types[column] == type_value).to_numpy()
        per_type[type_value] = {
            "prior_answer": priors.get(type_value),
            "n_dev": int(mask.sum()),
            "dev_accuracy": round(float(correct[mask].mean()), 5),
        }
    return {"overall_dev_accuracy": round(float(correct.mean()), 5),
            "per_type": per_type}


def main() -> None:
    utils.set_seed()
    lookup = load_type_lookup()
    print(f"type lookup entries: {len(lookup)}")

    dev = read_manifest("dev")
    train_40k = read_manifest("train_40k")
    dev_types = join_types(dev, lookup, "dev")
    train_types = join_types(train_40k, lookup, "train_40k")

    META_DIR.mkdir(parents=True, exist_ok=True)
    dev_types.to_csv(META_DIR / "dev_types.csv", index=False,
                     encoding="utf-8", lineterminator="\n")
    train_types.to_csv(META_DIR / "train_40k_types.csv", index=False,
                       encoding="utf-8", lineterminator="\n")
    print(f"wrote {META_DIR / 'dev_types.csv'} and train_40k_types.csv "
          f"(rows aligned to the manifest row order)")

    global_majority = sorted(Counter(train_40k["answer"]).items(),
                             key=lambda kv: (-kv[1], kv[0]))[0][0]
    global_accuracy = round(
        float((dev["answer"] == global_majority).mean()), 5)

    summary = {
        "join": {"dev_matched": int(len(dev)),
                 "train_40k_matched": int(len(train_40k)),
                 "dev_match_rate": 1.0, "train_40k_match_rate": 1.0},
        "distributions": {"dev": distribution(dev_types),
                          "train_40k": distribution(train_types)},
        "per_type_priors": {
            "global_majority": {"answer": global_majority,
                                "dev_accuracy": global_accuracy},
            "structural_prior": per_type_prior(train_40k, train_types,
                                               dev, dev_types, "structural"),
            "semantic_prior": per_type_prior(train_40k, train_types,
                                             dev, dev_types, "semantic"),
        },
        "note": ("Dev and train_40k only; no clean-test metadata was built "
                 "and no clean-test statistic was computed."),
    }
    metadata = utils.run_metadata()
    metadata["v2_05_metadata"] = summary
    utils.save_json(metadata, OUT_DIR / "metadata_summary.json")

    print(f"global majority '{global_majority}': dev {global_accuracy:.4f}")
    for kind in ("structural_prior", "semantic_prior"):
        print(f"{kind}: overall dev "
              f"{summary['per_type_priors'][kind]['overall_dev_accuracy']:.4f}")
    print("dev structural counts:",
          summary["distributions"]["dev"]["structural_counts"])
    print("dev semantic counts:",
          summary["distributions"]["dev"]["semantic_counts"])
    print("dev n_steps histogram:",
          summary["distributions"]["dev"]["n_steps_histogram"])
    print("v2_05 metadata join complete.")


if __name__ == "__main__":
    main()
