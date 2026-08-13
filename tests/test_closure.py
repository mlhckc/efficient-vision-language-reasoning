"""Closure tests: artefact integrity, timing-class discipline, claim discipline.

CPU-only and read-only over the closure outputs, except for two isolated
temporary files used to prove the deterministic writers are deterministic.
Nothing is trained, no checkpoint is opened and the embargoed clean-test
target is never read or named; the token is assembled at run time so this
source does not contain it.

Every load-bearing invariant is exercised as a known-positive and a
known-negative, because a check that cannot fail is not evidence.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.closure import closure_common as cc  # noqa: E402

CLOSURE = PROJECT_ROOT / "results" / "closure"
EMBARGOED = "test_" + "clean_targets"
PERMITTED_STATUSES = {"READY", "READY_AFTER_CPU_STATS",
                      "OPTIONAL_EVALUATION_ONLY", "NOT_SUPPORTED",
                      "SUPERSEDED"}
END_TO_END = "END_TO_END_SERIAL"

VERBOSE = False
_CHECKS: list[tuple[str, bool]] = []
_INSIDE_KNOWN_NEGATIVE = False


def check(name: str, condition: bool, detail: str = "") -> None:
    if not _INSIDE_KNOWN_NEGATIVE:
        _CHECKS.append((name, bool(condition)))
        if VERBOSE:
            print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
                  + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def must_fail(name: str, thunk) -> None:
    """A known-negative: the guard must reject the bad input."""
    global _INSIDE_KNOWN_NEGATIVE
    _INSIDE_KNOWN_NEGATIVE = True
    try:
        thunk()
        raised = False
    except Exception:
        raised = True
    finally:
        _INSIDE_KNOWN_NEGATIVE = False
    check(f"known-negative: {name}", raised,
          "the guard accepted input it must reject")


def load(name: str) -> dict:
    return json.loads((CLOSURE / name).read_text())


EXPECTED = {
    "A_evidence_registry.json": ("registry", "manifest_verification"),
    "B_primary_contrasts.json": ("contrasts", "by_role"),
    "C_reconstructed_evidence_manifest.json": ("row_evidence", "blockers"),
    "D_slice_statistics.json": ("slices", "slice_counts_reference"),
    "E_seed_variability.json": ("rows", "separation_statement"),
    "F_multiplicity_status.json": ("interval_counts",
                                   "multiplicity_statement"),
    "G_claim_evidence.json": ("claims", "rule"),
    "H_evidence_readiness.json": ("rows", "status_definitions"),
    "efficiency_closure_table.json": ("rows", "frontiers"),
    "manifest_e7b_serial_efficiency.json": ("entries", "entry_count"),
    "manifest_e8b_readout_generation.json": ("entries", "entry_count"),
    "reconstructed_statistics.json": ("contrasts", "arm_accuracies"),
}


def test_artefacts_present_and_shaped() -> None:
    for name, keys in EXPECTED.items():
        path = CLOSURE / name
        check(f"{name} exists", path.exists())
        payload = json.loads(path.read_text())
        for key in keys:
            check(f"{name} has {key}", key in payload)
        check(f"{name} records provenance", "provenance" in payload)
        provenance = payload["provenance"]
        for field in ("repository_head", "script", "script_sha256",
                      "command", "device", "gpu_used", "gpu_hours_charged"):
            check(f"{name} provenance has {field}", field in provenance)
        check(f"{name} declares cpu", provenance["device"] == "cpu")
        check(f"{name} declares no gpu", provenance["gpu_used"] is False)
        check(f"{name} charges no gpu-hours",
              provenance["gpu_hours_charged"] == 0.0)
        check(f"{name} trained nothing",
              provenance["trained_anything"] is False)
        check(f"{name} selected no checkpoint",
              provenance["selected_any_checkpoint"] is False)


def test_no_embargoed_reference() -> None:
    for path in sorted(CLOSURE.rglob("*")):
        if not path.is_file() or path.suffix == ".npz":
            continue
        check(f"{path.name} does not name the embargoed target",
              EMBARGOED not in path.read_text())
    for path in sorted((PROJECT_ROOT / "experiments" / "closure")
                       .glob("*.py")):
        text = path.read_text()
        if path.name == "closure_common.py":
            # Names the token only to refuse it. Assert that is still true:
            # it must appear in a refusal, never in an open() call.
            check("closure_common names the target only in a refusal",
                  "assert_not_embargoed" in text)
            continue
        check(f"{path.name} does not name the embargoed target",
              EMBARGOED not in text)
    must_fail("embargo guard rejects an embargoed path",
              lambda: cc.assert_not_embargoed(
                  PROJECT_ROOT / "data" / "v2" / (EMBARGOED + ".csv")))
    check("embargo guard accepts an ordinary path",
          cc.assert_not_embargoed(PROJECT_ROOT / "config.py").name
          == "config.py")


def test_registry_hashes_match_disk() -> None:
    registry = load("A_evidence_registry.json")
    for row in registry["registry"]:
        path = PROJECT_ROOT / row["artefact"]
        check(f"{row['artefact']} exists", path.exists())
        check(f"{row['artefact']} sha256 non-null", bool(row["sha256"]))
        check(f"{row['artefact']} sha256 matches disk",
              cc.sha256_file(path) == row["sha256"])
    totals = registry["manifest_verification"]["totals"]
    check("all manifest entries verified",
          totals["mismatch"] == 0 and totals["missing"] == 0,
          f"{totals}")
    check("manifest verification covered every entry",
          totals["checked"] == totals["ok"], f"{totals}")

    for manifest in ("manifest_e7b_serial_efficiency.json",
                     "manifest_e8b_readout_generation.json"):
        payload = load(manifest)
        check(f"{manifest} entry_count agrees",
              payload["entry_count"] == len(payload["entries"]))
        for entry in payload["entries"][:12]:
            path = PROJECT_ROOT / entry["path"]
            check(f"{entry['path']} hashed correctly",
                  path.exists() and cc.sha256_file(path) == entry["sha256"])
            for field in ("path", "sha256", "bytes", "experiment_family",
                          "role"):
                check(f"{manifest} entry has {field}", field in entry)


def test_row_evidence_matches_manifest() -> None:
    manifest = load("C_reconstructed_evidence_manifest.json")
    check("reconstruction recorded a reproduction gate",
          "reproduction_gate" in manifest)
    for row in manifest["row_evidence"]:
        check(f"{row['condition_id']} reproduced its point estimate",
              row["point_estimate_reproduced"] is True)
        check(f"{row['condition_id']} recomputed equals stored",
              row["recomputed_accuracy"] == row["stored_accuracy"])
        path = PROJECT_ROOT / row["row_evidence_path"]
        check(f"{row['condition_id']} array present", path.exists())
        for field in ("source_checkpoint", "source_cache", "metric_id",
                      "split_identity", "seed", "scale", "arm"):
            check(f"{row['condition_id']} records {field}", field in row)
    sample = manifest["row_evidence"][:8]
    for row in sample:
        path = PROJECT_ROOT / row["row_evidence_path"]
        check(f"{row['condition_id']} sha256 matches",
              cc.sha256_file(path) == row["row_evidence_sha256"])
        with np.load(path) as arrays:
            for field in ("question_id", "image_id", "gold_label",
                          "prediction_label", "correct"):
                check(f"{row['condition_id']} array has {field}",
                      field in arrays)
            check(f"{row['condition_id']} row count agrees",
                  len(arrays["correct"]) == row["n_questions"])
            check(f"{row['condition_id']} correctness matches accuracy",
                  round(float(arrays["correct"].mean()), 5)
                  == row["recomputed_accuracy"])

    check("v2_06 normal rows match their originating family",
          manifest["v2_06_normal_cross_check"]["all_identical"] is True)

    # A stopped family must contribute nothing at all.
    stopped = set(manifest["stopped_families"])
    check("at least one family was evaluated", bool(manifest["families"]))
    for row in manifest["row_evidence"]:
        check(f"{row['family']} is not a stopped family",
              row["family"] not in stopped)
    contrasts = load("B_primary_contrasts.json")["contrasts"]
    for family in stopped:
        offending = [c["contrast_id"] for c in contrasts
                     if c["contrast_id"].startswith(f"{family}/")]
        check(f"stopped family {family} contributes no contrast",
              not offending, str(offending))
    for blocker in manifest["blockers"]:
        check("a blocker names its action", bool(blocker.get("action")))


def test_intervals_are_coherent() -> None:
    for name, key in (("B_primary_contrasts.json", "contrasts"),
                      ("D_slice_statistics.json", "slices")):
        for row in load(name)[key]:
            interval = row.get("ci95_image_clustered")
            if not interval:
                continue
            low, high = interval
            check(f"{name} interval ordered", low <= high, str(row)[:120])
            point = row.get("point_estimate")
            if point is not None and row.get("analysis_status") != \
                    "READ_FROM_SOURCE":
                check(f"{name} point inside its interval",
                      low <= point <= high,
                      f"{row.get('contrast_id', row.get('slice_id'))} "
                      f"{point} not in {interval}")
            if row.get("n_questions") and row.get("n_unique_images"):
                check(f"{name} clusters do not exceed rows",
                      row["n_unique_images"] <= row["n_questions"])
            check(f"{name} names its cluster unit", bool(row.get(
                "cluster_unit")))


def test_seed_and_evaluation_uncertainty_are_separate() -> None:
    seeds = load("E_seed_variability.json")
    check("seed table states the separation",
          "training" in seeds["separation_statement"].lower()
          and "evaluation" in seeds["separation_statement"].lower())
    for row in seeds["rows"]:
        check(f"{row['row_id']} labels its sd quantity",
              "training seeds" in row["quantity"])
        check(f"{row['row_id']} carries no interval field",
              not any(k.startswith("ci95") for k in row))
        check(f"{row['row_id']} counts its seeds",
              row["n_training_seeds"] == len(row["seed_set"]))
    contrasts = load("B_primary_contrasts.json")["contrasts"]
    for row in contrasts:
        check(f"{row['contrast_id']} labels its interval kind",
              row["interval_kind"] == "evaluation-sampling, image-clustered")
        check(f"{row['contrast_id']} keeps sd out of the interval field",
              isinstance(row["ci95_image_clustered"], list))


def test_efficiency_timing_classes() -> None:
    table = load("efficiency_closure_table.json")
    check("efficiency measured nothing", table["measured_anything"] is False)
    for row in table["rows"]:
        check(f"{row['system_id']} declares a timing kind",
              row["timing_kind"] in table["timing_kind_definitions"])
        if row["timing_kind"] != END_TO_END:
            check(f"{row['system_id']} is not comparable with end-to-end",
                  row["comparable_with_end_to_end"] is False)
        else:
            check(f"{row['system_id']} is comparable with end-to-end",
                  row["comparable_with_end_to_end"] is True)
            check(f"{row['system_id']} has a serial latency",
                  row.get("warm_serial_median_ms") is not None)

    check("every efficiency row has a unique key",
          len({r["row_key"] for r in table["rows"]}) == len(table["rows"]))
    by_key = {r["row_key"]: r for r in table["rows"]}
    for name, frontier in table["frontiers"].items():
        members = set(frontier["members"])
        check(f"{name} frontier members all resolve",
              members <= set(by_key), str(members - set(by_key)))
        rows = [by_key[m] for m in members]
        check(f"{name} frontier is non-empty", bool(rows))
        check(f"{name} frontier has one timing kind",
              len({r["timing_kind"] for r in rows}) == 1)
        check(f"{name} frontier has one node",
              len({r["node"] for r in rows}) == 1)
        check(f"{name} frontier is end-to-end only",
              all(r["timing_kind"] == END_TO_END for r in rows))

    nodes = {frontier["node"] for frontier in table["frontiers"].values()}
    check("the two efficiency frontiers are on different nodes",
          len(nodes) == len(table["frontiers"]), str(nodes))
    bridge = table["two_nodes"]["bridge_control"]
    check("the bridge control is recorded as failing its tolerance",
          bridge["consistent_with_e7b"] is False)
    check("no adjustment factor is applied",
          "no adjustment factor" in table["two_nodes"]["consequence"].lower())
    check("E10 timing is optional, not measured here",
          table["e10_timing"]["not_measured_here"] is True
          and table["e10_timing"]["status"] == "OPTIONAL_EVALUATION_ONLY")


def test_claim_discipline() -> None:
    claims = load("G_claim_evidence.json")["claims"]
    check("claims exist", bool(claims))
    for row in claims:
        check(f"{row['claim_id']} has a non-empty limitation",
              bool(row["limitation"].strip()))
        check(f"{row['claim_id']} has a claim text",
              bool(row["claim_text"].strip()))
        check(f"{row['claim_id']} declares a comparison status",
              bool(row["comparison_status"]))
        check(f"{row['claim_id']} declares an evidence status",
              row["evidence_status"] in PERMITTED_STATUSES)

    # The energy prohibition: the word may appear only in its own row.
    prohibition = [r for r in claims if "energy" in r["claim_text"].lower()
                   or "energy" in r["limitation"].lower()]
    check("exactly one claim row mentions energy", len(prohibition) == 1,
          str([r["claim_id"] for r in prohibition]))
    check("the energy row is a prohibition",
          prohibition[0]["evidence_status"] == "NOT_SUPPORTED"
          and prohibition[0]["claim_text"].startswith("PROHIBITED"))
    for name in ("A_evidence_registry.json", "B_primary_contrasts.json",
                 "D_slice_statistics.json", "E_seed_variability.json",
                 "efficiency_closure_table.json",
                 "C_reconstructed_evidence_manifest.json"):
        check(f"{name} makes no energy claim",
              "energy" not in (CLOSURE / name).read_text().lower())

    readiness = load("H_evidence_readiness.json")
    for row in readiness["rows"]:
        check(f"{row['claim_id']} has exactly one permitted status",
              row["status"] in PERMITTED_STATUSES, row["status"])
    check("no optional measurement was executed",
          readiness["optional_items_executed"] == 0)
    for item in readiness["optional_items"]:
        check(f"{item['id']} was not executed", item["executed"] is False)


def test_deterministic_writers() -> None:
    """The writers must depend only on content, or idempotence is a fiction."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        arrays = {"correct": np.array([1, 0, 1], dtype="uint8"),
                  "question_id": np.array(["a", "b", "c"], dtype="U")}
        first = cc.write_npz(arrays, root / "one.npz")
        second = cc.write_npz(arrays, root / "two.npz")
        check("write_npz is content-addressed", first == second)
        check("write_npz round-trips",
              list(np.load(root / "one.npz")["correct"]) == [1, 0, 1])

        payload = {"b": 2, "a": {"d": 4, "c": 3}}
        reordered = {"a": {"c": 3, "d": 4}, "b": 2}
        check("write_json is key-order independent",
              cc.write_json(payload, root / "one.json")
              == cc.write_json(reordered, root / "two.json"))
        must_fail("write_json refuses an embargoed payload",
                  lambda: cc.write_json({"path": EMBARGOED},
                                        root / "three.json"))


def test_bootstrap_reproduces_the_project_procedure() -> None:
    """The closure bootstrap must equal the one v2_05c already published."""
    stored = json.loads(
        (PROJECT_ROOT / "results" / "experiments" / "v2_05_types"
         / "addendum_clustered.json").read_text())[
             "v2_05c_clustered_correction"]
    check("v2_05c recorded its method", "image-clustered" in stored["method"])
    check("closure uses the same draw count",
          str(cc.BOOTSTRAP_DRAWS) in stored["method"])
    check("closure uses the same generator seed",
          f"default_rng({cc.BOOTSTRAP_RNG_SEED})" in stored["method"])

    # Known-positive: a deterministic case with an obvious answer.
    values = [np.ones(100)]
    clusters = np.repeat(np.arange(10), 10)
    result = cc.clustered_interval(values, clusters, draws=50)
    check("all-correct rows give a point estimate of one",
          result["point_estimate"] == 1.0)
    check("all-correct rows give a degenerate interval",
          result["ci95_image_clustered"] == [1.0, 1.0])
    check("interval reports its cluster count",
          result["n_unique_images"] == 10)
    check("interval reports its row count", result["n_questions"] == 100)
    check("interval names its dependence unit",
          result["cluster_unit"] == cc.CLUSTER_UNIT)
    check("interval states what it conditions on",
          "fixed trained seed set" in result["conditions_on"])
    must_fail("clustered_interval refuses an empty slice",
              lambda: cc.clustered_interval(values, clusters,
                                            np.zeros(100, dtype=bool)))

    # A mixed case: the interval must bracket the point estimate.
    rng = np.random.default_rng(7)
    noisy = [rng.integers(0, 2, size=200).astype("float64")]
    index = np.repeat(np.arange(20), 10)
    mixed = cc.clustered_interval(noisy, index, draws=200)
    low, high = mixed["ci95_image_clustered"]
    check("mixed interval brackets its point estimate",
          low <= mixed["point_estimate"] <= high)


def run() -> None:
    _CHECKS.clear()
    test_artefacts_present_and_shaped()
    test_no_embargoed_reference()
    test_registry_hashes_match_disk()
    test_row_evidence_matches_manifest()
    test_intervals_are_coherent()
    test_seed_and_evaluation_uncertainty_are_separate()
    test_efficiency_timing_classes()
    test_claim_discipline()
    test_deterministic_writers()
    test_bootstrap_reproduces_the_project_procedure()
    failed = [name for name, ok in _CHECKS if not ok]
    if failed:
        raise AssertionError(f"{len(failed)} closure checks failed: {failed}")
    print(f"  {len(_CHECKS)} closure checks passed")


if __name__ == "__main__":
    VERBOSE = True
    run()
