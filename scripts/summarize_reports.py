#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd


CONTROLS = ["mention", "negation", "identity", "reverse", "unrelated"]


def read_csv(report_dir: Path, name: str) -> pd.DataFrame | None:
    path = report_dir / name
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return None if df.empty else df


def print_table(df: pd.DataFrame, decimals: int = 4) -> None:
    if df.empty:
        print("(no rows)")
        return
    print(df.to_string(index=False, float_format=lambda x: f"{x:.{decimals}f}"))


def condition_table(
    df: pd.DataFrame,
    value: str,
    conditions: list[str] | None = None,
) -> pd.DataFrame:
    if conditions is not None:
        df = df[df["condition"].isin(conditions)]
    return (
        df.groupby("condition")[value]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )


def summarize_behavior(report_dir: Path) -> None:
    df = read_csv(report_dir, "behavior.csv")
    if df is None:
        return
    print("\n=== Behavioral scoring ===")
    print_table(condition_table(df, "p_target_vs_source"))

    means = df.groupby("condition")["p_target_vs_source"].mean()
    if "mapping" in means:
        gaps = [
            {"control": control, "mapping_minus_control": means["mapping"] - means[control]}
            for control in CONTROLS
            if control in means
        ]
        print("\nMapping-control probability gaps:")
        print_table(pd.DataFrame(gaps))

    calibrated = read_csv(report_dir, "behavior_calibrated.csv")
    if calibrated is not None:
        summary = condition_table(
            calibrated,
            "calibrated_redefinition_score",
        )
        print("\nCalibrated redefinition scores:")
        print_table(summary)


def layer_control_gaps(
    df: pd.DataFrame,
    value: str,
    controls: list[str] = CONTROLS,
) -> pd.DataFrame:
    means = df.groupby(["layer", "condition"])[value].mean().unstack("condition")
    if "mapping" not in means:
        return pd.DataFrame()
    present = [control for control in controls if control in means]
    out = pd.DataFrame({"layer": means.index, "mapping": means["mapping"].values})
    for control in present:
        out[f"minus_{control}"] = means["mapping"].values - means[control].values
    gap_cols = [column for column in out if column.startswith("minus_")]
    if gap_cols:
        out["mean_control_gap"] = out[gap_cols].mean(axis=1)
        return out.sort_values("mean_control_gap", ascending=False)
    return out.sort_values("mapping", ascending=False)


def summarize_movement(report_dir: Path, top_k: int) -> None:
    projection = read_csv(report_dir, "movement_projection.csv")
    if projection is not None:
        held_out = projection[projection["template_split"] == "test_template"]
        print("\n=== Held-out movement projection ===")
        gaps = layer_control_gaps(
            held_out,
            "target_minus_source_projection",
        )
        print(f"Top {top_k} layers by mapping-control projection gap:")
        print_table(gaps.head(top_k))

    movement = read_csv(report_dir, "movement.csv")
    if movement is not None:
        print("\nRaw cosine movement, averaged over layers and examples:")
        print_table(
            condition_table(
                movement,
                "cos_to_target_minus_source",
                ["mapping", *CONTROLS],
            )
        )


def summarize_probes(report_dir: Path, top_k: int) -> None:
    df = read_csv(report_dir, "probe.csv")
    if df is None:
        return
    print("\n=== Linear probes ===")
    condition_rows = df[df["eval_type"] == "condition_prob"]
    gaps = layer_control_gaps(condition_rows, "p_target")
    print(f"Top {top_k} layers by mapping-control probe selectivity:")
    print_table(gaps.head(top_k))

    baseline = df[
        (df["eval_type"] == "baseline_auc")
        & (df["template_split"] == "test_template")
    ]
    if len(baseline):
        print(
            "\nHeld-out baseline AUC: "
            f"mean={baseline['auc'].mean():.4f}, "
            f"std={baseline['auc'].std():.4f}, "
            f"n={len(baseline)}"
        )

    random_df = read_csv(report_dir, "probe_random_label_controls.csv")
    if random_df is not None and len(random_df):
        mapping_random = random_df[random_df["condition"] == "mapping"]
        if len(mapping_random):
            print(
                "Random-label mapping P(target): "
                f"mean={mapping_random['p_target_random_label'].mean():.4f}, "
                f"std={mapping_random['p_target_random_label'].std():.4f}"
            )


def paired_patching_effects(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["example_id", "pair_id", "template_id", "layer"]
    baseline = (
        df[df["intervention"] == "unpatched"][keys + ["target_pref_logit"]]
        .drop_duplicates(keys)
        .rename(columns={"target_pref_logit": "unpatched_target_pref"})
    )
    patched = df[df["intervention"] != "unpatched"].merge(
        baseline,
        on=keys,
        how="inner",
    )
    patched["target_pref_change"] = (
        patched["target_pref_logit"] - patched["unpatched_target_pref"]
    )
    return (
        patched.groupby(["layer", "intervention", "alpha"])
        .agg(
            mean_p_target=("p_target_vs_source", "mean"),
            mean_target_pref_change=("target_pref_change", "mean"),
            std_target_pref_change=("target_pref_change", "std"),
            n=("target_pref_change", "size"),
        )
        .reset_index()
    )


def summarize_patching(report_dir: Path, top_k: int) -> None:
    df = read_csv(report_dir, "patching.csv")
    if df is None:
        return
    print("\n=== Causal patching ===")
    effects = paired_patching_effects(df)

    alpha_one = effects[
        np.isclose(effects["alpha"], 1.0)
        & effects["intervention"].isin(
            [
                "subtract_train_mean_delta",
                "subtract_wrong_pair_delta",
                "subtract_random_norm_matched",
            ]
        )
    ].sort_values("mean_target_pref_change")
    if len(alpha_one):
        print("Alpha=1 subtraction effects; negative means target preference decreased:")
        print_table(alpha_one)

    replacements = effects[
        effects["intervention"].isin(
            ["replace_with_source_baseline", "replace_with_target_baseline"]
        )
    ].sort_values(["layer", "intervention"])
    if len(replacements):
        print("\nReplacement controls relative to the paired unpatched score:")
        print_table(replacements)

    true_delta = effects[
        effects["intervention"] == "subtract_train_mean_delta"
    ].sort_values("mean_target_pref_change")
    if len(true_delta):
        print(f"\nTop {top_k} strongest train-delta reversals:")
        print_table(true_delta.head(top_k))


def render_summary(report_dir: Path, top_k: int) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(f"Report directory: {report_dir.resolve()}")
        summarize_behavior(report_dir)
        summarize_movement(report_dir, top_k)
        summarize_probes(report_dir, top_k)
        summarize_patching(report_dir, top_k)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize existing concept-redefinition CSV reports."
    )
    parser.add_argument(
        "report_dir",
        type=Path,
        help="Directory containing behavior.csv, movement.csv, probe.csv, etc.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary text path. Defaults to <report_dir>/analysis_summary.txt.",
    )
    args = parser.parse_args()

    if not args.report_dir.is_dir():
        raise SystemExit(f"Report directory does not exist: {args.report_dir}")

    summary = render_summary(args.report_dir, args.top_k)
    print(summary, end="")

    output = args.output or args.report_dir / "analysis_summary.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summary, encoding="utf-8")
    print(f"\nSaved summary to {output}")


if __name__ == "__main__":
    main()
