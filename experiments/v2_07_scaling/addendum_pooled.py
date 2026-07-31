"""Recompute the v2_07 seed-42 >=4-step deficit by question count."""

import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils


SCALING_RESULTS_PATH = (
    PROJECT_ROOT / "results/experiments/v2_07_scaling/results.json"
)
TYPE_ADDENDUM_PATH = (
    PROJECT_ROOT / "results/experiments/v2_05_types/addendum.json"
)
STEP_STATISTICS_PATH = (
    PROJECT_ROOT / "results/experiments/v3_02a_refs/step_statistics.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results/experiments/v2_07_scaling/addendum_pooled.json"
)

MODELS = ("question_only", "concat", "fusion", "product_576k")
EXPECTED_STEP_ROWS = {
    "<=2": (2400, 0.20917),
    "3": (3330, 0.14625),
    "4": (870, 0.27356),
    ">=5": (1114, 0.51616),
}
REFERENCE_VALUES = {
    "question_only": (0.15121, 0.07489),
    "concat": (0.22631, 0.08443),
    "fusion": (0.23085, 0.08075),
    "product_576k": (0.23488, 0.07780),
}
REFERENCE_TOLERANCE = 5e-5
PRIOR_TOLERANCE = 1e-5


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_step_rows(addendum: dict) -> dict:
    rows = {}
    for row in addendum["lift"]:
        if row["kind"] != "steps":
            continue
        bucket = row["slice"]
        if bucket in rows:
            raise ValueError(f"duplicate steps row for bucket {bucket}")
        rows[bucket] = row

    if set(rows) != set(EXPECTED_STEP_ROWS):
        raise ValueError(
            f"unexpected steps buckets: {sorted(rows)}"
        )

    priors_by_slice = addendum["prior_accuracy_by_slice"]
    for bucket, (expected_n, expected_prior) in EXPECTED_STEP_ROWS.items():
        row = rows[bucket]
        if row["n"] != expected_n:
            raise ValueError(
                f"steps {bucket} count {row['n']} != {expected_n}"
            )
        prior = float(row["prior_accuracy"])
        if not math.isclose(
            prior, expected_prior, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"steps {bucket} prior {prior} != {expected_prior}"
            )
        indexed_prior = float(priors_by_slice[f"steps:{bucket}"])
        if not math.isclose(
            prior, indexed_prior, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"steps {bucket} row/index prior mismatch: "
                f"{prior} != {indexed_prior}"
            )
    return rows


def load_combined_row(step_statistics: dict) -> dict:
    rows = [
        row
        for row in step_statistics["bucket_table"]
        if row["bucket"] == ">=4_combined"
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one >=4_combined row, found {len(rows)}"
        )
    return rows[0]


def calculate_models(lift_results: dict, n4: int, n5: int) -> dict:
    models = {}
    for model in MODELS:
        source = lift_results[model]
        lift4 = float(source["per_bucket_lift"]["4"])
        lift5 = float(source["per_bucket_lift"][">=5"])
        mean_step_lift = float(source["mean_step_lift"])
        unweighted_lift = float(
            source["steps_ge4_lift_mean_of_buckets"]
        )
        pooled_lift = (n4 * lift4 + n5 * lift5) / (n4 + n5)
        pooled_deficit = mean_step_lift - pooled_lift
        unweighted_deficit = mean_step_lift - unweighted_lift

        reference_lift, reference_deficit = REFERENCE_VALUES[model]
        if abs(pooled_lift - reference_lift) > REFERENCE_TOLERANCE:
            raise ValueError(
                f"{model} pooled lift {pooled_lift} does not match "
                f"reference {reference_lift}"
            )
        if abs(pooled_deficit - reference_deficit) > REFERENCE_TOLERANCE:
            raise ValueError(
                f"{model} pooled deficit {pooled_deficit} does not match "
                f"reference {reference_deficit}"
            )

        models[model] = {
            "bucket_lift_4": lift4,
            "bucket_lift_ge5": lift5,
            "mean_step_lift": mean_step_lift,
            "pooled_ge4_lift": pooled_lift,
            "pooled_ge4_deficit": pooled_deficit,
            "unweighted_ge4_lift": unweighted_lift,
            "unweighted_deficit": unweighted_deficit,
        }
    return models


def print_tables(step_rows: dict, pooled_prior: float, models: dict) -> None:
    print("step_bucket n prior")
    for bucket in ("<=2", "3", "4", ">=5"):
        row = step_rows[bucket]
        print(
            f"{bucket:>11} {row['n']:4d} "
            f"{float(row['prior_accuracy']):.5f}"
        )
    print(f"pooled >=4 {step_rows['4']['n'] + step_rows['>=5']['n']:4d} "
          f"{pooled_prior:.5f}")
    print()
    print(
        "model bucket_4 bucket_ge5 mean_step pooled_ge4_lift "
        "pooled_ge4_deficit unweighted_ge4_lift unweighted_deficit"
    )
    for model in MODELS:
        row = models[model]
        print(
            f"{model} "
            f"{row['bucket_lift_4']:.5f} "
            f"{row['bucket_lift_ge5']:.5f} "
            f"{row['mean_step_lift']:.5f} "
            f"{row['pooled_ge4_lift']:.5f} "
            f"{row['pooled_ge4_deficit']:.5f} "
            f"{row['unweighted_ge4_lift']:.5f} "
            f"{row['unweighted_deficit']:.5f}"
        )


def main() -> None:
    utils.set_seed()

    scaling = load_json(SCALING_RESULTS_PATH)["v2_07_scaling"]
    type_addendum = load_json(TYPE_ADDENDUM_PATH)["v2_05b_addendum"]
    step_statistics = load_json(STEP_STATISTICS_PATH)[
        "v3_02a_step_statistics"
    ]

    step_rows = load_step_rows(type_addendum)
    row4 = step_rows["4"]
    row5 = step_rows[">=5"]
    n4 = int(row4["n"])
    n5 = int(row5["n"])
    prior4 = float(row4["prior_accuracy"])
    prior5 = float(row5["prior_accuracy"])
    pooled_prior = (n4 * prior4 + n5 * prior5) / (n4 + n5)

    combined_row = load_combined_row(step_statistics)
    if combined_row["n_questions"] != n4 + n5:
        raise ValueError(
            "combined >=4 question count "
            f"{combined_row['n_questions']} != {n4 + n5}"
        )
    stored_pooled_prior = float(combined_row["prior_accuracy"])
    if abs(pooled_prior - stored_pooled_prior) > PRIOR_TOLERANCE:
        raise ValueError(
            f"pooled prior {pooled_prior} does not match stored "
            f"{stored_pooled_prior}"
        )

    models = calculate_models(
        scaling["lift_250k_seed42"], n4=n4, n5=n5
    )
    definitions = step_statistics["definitions"]
    output = {
        "metadata": utils.run_metadata(),
        "v2_07_pooled_addendum": {
            "definitions": {
                "pooled_ge4_lift": definitions["combined_ge4"],
                "pooled_ge4_deficit": definitions["ge4_deficit"],
                "superseded_definition": definitions["v3_01_note"],
                "scope": (
                    "V2 development results at 250k and seed 42 only; "
                    "other seeds and scales require checkpoint re-evaluation."
                ),
            },
            "sources": [
                "results/experiments/v2_07_scaling/results.json",
                "results/experiments/v2_05_types/addendum.json",
                "results/experiments/v3_02a_refs/step_statistics.json",
            ],
            "n4": n4,
            "n5": n5,
            "priors": {
                bucket: float(step_rows[bucket]["prior_accuracy"])
                for bucket in ("<=2", "3", "4", ">=5")
            },
            "pooled_prior": pooled_prior,
            "models": models,
        },
    }
    utils.save_json(output, OUTPUT_PATH)
    print_tables(step_rows, pooled_prior, models)


if __name__ == "__main__":
    main()
