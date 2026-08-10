"""Render the E9 result package as markdown tables.

Reads `e9_results.json` and `e9_efficiency.json` only. Computes nothing: every
number here is copied from a frozen artefact, so the report and the artefacts
cannot drift. PRIMARY, SECONDARY DIAGNOSTIC and AUXILIARY labels travel with
the numbers into every table.
"""

from __future__ import annotations

import json
import sys

import e9_common as e9

MODEL_LABEL = {"smolvlm_256m": "SmolVLM-256M (PRIMARY)",
               "smolvlm_500m": "SmolVLM-500M (SECONDARY)"}


def fmt(value, places: int = 5) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    return f"{value:.{places}f}"


def interval(entry: dict) -> str:
    return (f"{entry['mean_difference']:+.5f} "
            f"[{entry['ci_lower']:+.5f}, {entry['ci_upper']:+.5f}]")


def main() -> int:
    results = json.loads(
        (e9.RESULTS_DIR / "e9_results.json").read_text())["e9_results"]
    lines: list[str] = []
    add = lines.append

    add("# E9 result tables\n")
    # The G17 wording is precise about what was and was not touched: contents
    # were never opened, but a pre-score implementation did resolve the path.
    add("Development set only. Clean-test contents were never opened, read, "
        "scored or used. A pre-score implementation briefly resolved and "
        "stat()ed the embargoed path, which violated the project's stricter "
        "G17 path-level rule; it was detected and repaired before any "
        "scientific score was produced (g17_remediation.json).\n")
    add(f"Framing: {results['framing']}\n")

    # --- Table 1: primary open generation --------------------------------
    add("\n## Table 1 (PRIMARY) - open generation accuracy\n")
    add("| model | condition | denominator | n | raw exact | G21 normalised |")
    add("|---|---|---|---|---|---|")
    for key, entry in results["scored_passes"].items():
        if entry["readout"] != "open":
            continue
        model = MODEL_LABEL[key.split("/")[0]]
        for view in entry["views"].values():
            add(f"| {model} | {entry['condition']} "
                f"({entry['condition_role']}) | {view['denominator']} | "
                f"{view['n']:,} | {fmt(view['raw_exact'], 6)} | "
                f"{fmt(view['normalised_exact'], 6)} |")

    # --- Table 2: constrained diagnostic ---------------------------------
    add("\n## Table 2 (SECONDARY DIAGNOSTIC) - trie-constrained generation\n")
    add("Answer support: top-1000. E8B's classifiers choose among 100. The "
        "two tasks are not equated.\n")
    add("| model | condition | denominator | n | raw exact | G21 normalised |")
    add("|---|---|---|---|---|---|")
    for key, entry in results["scored_passes"].items():
        if entry["readout"] != "constrained":
            continue
        model = MODEL_LABEL[key.split("/")[0]]
        for view in entry["views"].values():
            add(f"| {model} | {entry['condition']} | {view['denominator']} | "
                f"{view['n']:,} | {fmt(view['raw_exact'], 6)} | "
                f"{fmt(view['normalised_exact'], 6)} |")

    # --- Table 3: open versus constrained --------------------------------
    add("\n## Table 3 - open minus constrained (format and support cost)\n")
    add("| model | condition | difference | 95% CI | outcome |")
    add("|---|---|---|---|---|")
    for key, entry in results["open_versus_constrained"].items():
        model, condition = key.split("/")
        add(f"| {MODEL_LABEL[model]} | {condition} | "
            f"{entry['mean_difference']:+.5f} | "
            f"[{entry['ci_lower']:+.5f}, {entry['ci_upper']:+.5f}] | "
            f"{entry['outcome']} |")

    # --- Table 4: visual reliance ----------------------------------------
    add("\n## Table 4 (PRIMARY) - visual reliance, normal minus deranged\n")
    add("The deranged-image condition is the matched comparison across E8B "
        "and E9. E9 is one deterministic run; E8B is three trained seeds.\n")
    add("| system | readout | drop | 95% CI | seeds | outcome |")
    add("|---|---|---|---|---|---|")
    for key, entry in results["internal_contrasts"].items():
        if not key.endswith("normal_minus_deranged"):
            continue
        model, readout, _ = key.split("/")
        add(f"| {MODEL_LABEL[model]} | {readout} | "
            f"{entry['mean_difference']:+.5f} | "
            f"[{entry['ci_lower']:+.5f}, {entry['ci_upper']:+.5f}] | "
            f"none (untrained) | {entry['outcome']} |")
    for cell, entry in results["e8b_reference_reliance"].items():
        add(f"| {cell} | R1/classifier | {entry['mean_difference']:+.5f} | "
            f"[{entry['ci_lower']:+.5f}, {entry['ci_upper']:+.5f}] | "
            f"{', '.join(f'{d:+.5f}' for d in entry['per_seed'])} | "
            f"three trained seeds |")

    # --- Table 5: auxiliary blank ----------------------------------------
    add("\n## Table 5 (AUXILIARY / UNMATCHED) - normal minus blank image\n")
    add("Not equated with E8B `fixed_image`, which is a feature-space mean.\n")
    add("| model | readout | drop | 95% CI | outcome |")
    add("|---|---|---|---|---|")
    for key, entry in results["internal_contrasts"].items():
        if not key.endswith("normal_minus_blank"):
            continue
        model, readout, _ = key.split("/")
        add(f"| {MODEL_LABEL[model]} | {readout} | "
            f"{entry['mean_difference']:+.5f} | "
            f"[{entry['ci_lower']:+.5f}, {entry['ci_upper']:+.5f}] | "
            f"{entry['outcome']} |")

    # --- Table 6: E9 against E8B -----------------------------------------
    add("\n## Table 6 (CONTEXTUAL) - E9 minus E8B, paired, raw denominator\n")
    add("Never a leaderboard claim. Training data, multimodal pretraining, "
        "answer support and output format all differ. The E9 side is one "
        "deterministic checkpoint; the E8B side is three trained seeds "
        "entering each draw as the within-draw seed mean. These are not "
        "equivalent sources of uncertainty.\n")
    add("| comparison | difference | 95% CI | outcome |")
    add("|---|---|---|---|")
    for key, entry in results["versus_e8b_paired"].items():
        add(f"| {key} | {entry['mean_difference']:+.5f} | "
            f"[{entry['ci_lower']:+.5f}, {entry['ci_upper']:+.5f}] | "
            f"{entry['outcome'].split(' (')[0]} |")

    # --- Table 7: E8B reference accuracy ---------------------------------
    add("\n## Table 7 - E8B reference accuracy, raw denominator\n")
    add("Recomputed here from E8B's frozen per-row vectors; reproduces E8B "
        "Table 6.\n")
    add("| cell | readout | G21 normalised | per seed |")
    add("|---|---|---|---|")
    for cell, entry in results["e8b_reference_accuracy"].items():
        add(f"| {cell} | {entry['readout']} | "
            f"{fmt(entry['normalised_exact'], 6)} | "
            f"{', '.join(f'{v:.5f}' for v in entry['per_seed'])} |")

    # --- Table 8: format diagnostics -------------------------------------
    add("\n## Table 8 - emission and format diagnostics\n")
    add("| pass | mean tokens | max | empty | overlong | in top-100 | "
        "in top-1000 | distinct | raw minus normalised |")
    add("|---|---|---|---|---|---|---|---|---|")
    for key, entry in results["scored_passes"].items():
        d = entry["diagnostics"]
        add(f"| {key} | {d['mean_generated_tokens']} | "
            f"{d['max_generated_tokens']} | {fmt(d['empty_rate'], 6)} | "
            f"{fmt(d['overlong_rate'], 6)} | "
            f"{fmt(d['emission_in_top100_rate'], 4)} | "
            f"{fmt(d['emission_in_top1000_rate'], 4)} | "
            f"{d['distinct_emissions']:,} | "
            f"{fmt(d['raw_minus_normalised'], 6)} |")

    # --- Table 9: efficiency ---------------------------------------------
    efficiency = results.get("efficiency")
    if efficiency:
        add("\n## Table 9 (PRIMARY) - same-node serial efficiency\n")
        add(f"E7b protocol: batch 1, {efficiency['warmup']} warm-up then "
            f"{efficiency['iterations']} timed queries, "
            f"{efficiency['passes']} passes.\n")
        add("| system | role | readout | warm median (ms) | per-pass | "
            "QPS | gen tokens | peak MiB | loaded params | node |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for name, entry in efficiency["aggregate"].items():
            add(f"| {name} | {entry['role']} | {entry['readout']} | "
                f"{entry['warm_median_ms']:.3f} | "
                f"{', '.join(f'{m:.2f}' for m in entry['per_pass_median_ms'])}"
                f" | {entry['queries_per_second']:.1f} | "
                f"{entry['mean_generated_tokens']} | "
                f"{entry['peak_memory_reserved_bytes'] / 2**20:.0f} | "
                f"{entry['total_parameters']:,} | {entry['node']} |")
        bridge = efficiency.get("e7b_bridge")
        if bridge:
            add("\n**E7b bridge control.** fusion on "
                f"{bridge['e9_node']}: {bridge['e9_node_median_ms']:.3f} ms "
                f"against the frozen {bridge['historical_node']} value "
                f"{bridge['e7b_historical_median_ms']:.3f} ms, relative "
                f"difference {bridge['relative_difference']:+.4f} against a "
                f"pre-registered tolerance of "
                f"{bridge['pre_registered_tolerance']:.2f}. Consistent with "
                f"E7b: **{fmt(bridge['consistent_with_e7b'])}**. "
                f"{bridge['policy']}\n")
        add(f"\n**R1 not timed.** {efficiency['r1_exclusion']['reason']} "
            f"{efficiency['r1_exclusion']['consequence']}\n")

    # --- Table 10: historical reference ----------------------------------
    historical = results["e7b_historical_reference"]
    add(f"\n## Table 10 (HISTORICAL / REFERENCE, {historical['node']}) - "
        "frozen E7b values\n")
    add(f"{historical['status']} {historical['denominator']}.\n")
    add("| system | accuracy | warm serial (ms) |")
    add("|---|---|---|")
    for name, entry in historical["systems"].items():
        add(f"| {name} | {entry['accuracy']:.5f} | "
            f"{entry['warm_serial_ms']:.3f} |")

    # --- Table 11: resources ---------------------------------------------
    ledger = results["resource_ledger"]
    determinism = results["determinism"]
    add("\n## Table 11 - resources, determinism and integrity\n")
    add("| quantity | value |")
    add("|---|---|")
    add(f"| E9 branch GPU-hours charged | {ledger['total_gpu_hours']:.4f} |")
    add(f"| pre-registered branch halt | {ledger['branch_halt_hours']:.1f} |")
    add(f"| headroom | {ledger['headroom_hours']:.4f} |")
    add(f"| determinism rows | {determinism['rows']:,} |")
    add("| greedy asserted | "
        f"{fmt(determinism['greedy_asserted'])} |")
    add("| replication bitwise identical | "
        f"{fmt(determinism['tokens_bitwise_identical'])} |")
    add(f"| optional cells dropped | {len(results['optional_dropped'])} |")
    add("| clean-test contents opened, read, scored or used | "
        f"{fmt(results['clean_test_accessed'])} |")
    add("| embargoed path resolved by a pre-score implementation | yes, "
        "repaired before any score (g17_remediation.json) |")
    add(f"| frozen protocol sha256 | `{results['frozen_protocol_sha256']}` |")

    path = e9.RESULTS_DIR / "e9_results_tables.md"
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(f"wrote {path.name} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
