#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import os
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


def position_condition_table(
    df: pd.DataFrame,
    value: str,
    conditions: list[str] | None = None,
) -> pd.DataFrame:
    if "position" not in df:
        return condition_table(df, value, conditions)
    if conditions is not None:
        df = df[df["condition"].isin(conditions)]
    return (
        df.groupby(["position", "condition"])[value]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["position", "mean"], ascending=[True, False])
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
    group_columns = ["layer"]
    if "position" in df:
        group_columns.insert(0, "position")
    means = df.groupby([*group_columns, "condition"])[value].mean().unstack(
        "condition"
    )
    if "mapping" not in means:
        return pd.DataFrame()
    present = [control for control in controls if control in means]
    out = means[["mapping"]].reset_index()
    for control in present:
        out[f"minus_{control}"] = (
            means["mapping"].values - means[control].values
        )
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
            position_condition_table(
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
        print("\nHeld-out baseline AUC by position:")
        if "position" in baseline:
            print_table(
                baseline.groupby("position").auc.agg(["mean", "std", "count"]).reset_index()
            )
        else:
            print(
                f"mean={baseline['auc'].mean():.4f}, "
                f"std={baseline['auc'].std():.4f}, "
                f"n={len(baseline)}"
            )

    random_df = read_csv(report_dir, "probe_random_label_controls.csv")
    if random_df is not None and len(random_df):
        mapping_random = random_df[random_df["condition"] == "mapping"]
        if len(mapping_random):
            print("Random-label mapping P(target) by position:")
            if "position" in mapping_random:
                print_table(
                    mapping_random.groupby("position")
                    .p_target_random_label.agg(["mean", "std", "count"])
                    .reset_index()
                )
            else:
                print(
                    f"mean={mapping_random['p_target_random_label'].mean():.4f}, "
                    f"std={mapping_random['p_target_random_label'].std():.4f}"
                )


def summarize_mapping_detector(report_dir: Path, top_k: int) -> None:
    results = read_csv(report_dir, "mapping_detector.csv")
    if results is None:
        return
    print("\n=== Mapping-vs-control detector ===")
    summary = (
        results.groupby(["split_type", "position"])
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_average_precision=("average_precision", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            folds=("roc_auc", "size"),
        )
        .reset_index()
        .sort_values(["split_type", "mean_roc_auc"], ascending=[True, False])
    )
    print_table(summary)

    print(f"\nTop {top_k} detector layer/position results:")
    print_table(
        results.sort_values("roc_auc", ascending=False)
        .head(top_k)[
            [
                "split_type",
                "held_out_group",
                "position",
                "layer",
                "roc_auc",
                "average_precision",
                "balanced_accuracy",
                "n_test",
            ]
        ]
    )

    random_results = read_csv(
        report_dir,
        "mapping_detector_random_label_controls.csv",
    )
    if random_results is not None:
        random_summary = (
            random_results.groupby(["split_type", "position"]).roc_auc
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        print("\nRandom-label detector ROC AUC:")
        print_table(random_summary)


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


def comparison_records(
    report_dirs: list[Path],
    labels: list[str],
) -> pd.DataFrame:
    columns = [
        "model",
        "metric",
        "condition",
        "split_type",
        "position",
        "layer",
        "value",
    ]
    records = []
    for report_dir, label in zip(report_dirs, labels):
        behavior = read_csv(report_dir, "behavior.csv")
        if behavior is not None:
            for condition, value in (
                behavior.groupby("condition").p_target_vs_source.mean().items()
            ):
                records.append(
                    {
                        "model": label,
                        "metric": "behavior_p_target",
                        "condition": condition,
                        "split_type": None,
                        "position": None,
                        "layer": None,
                        "value": value,
                    }
                )

        projection = read_csv(report_dir, "movement_projection.csv")
        if projection is not None:
            held_out = projection[
                projection["template_split"] == "test_template"
            ]
            gaps = layer_control_gaps(
                held_out,
                "target_minus_source_projection",
            )
            for row in gaps.itertuples():
                records.append(
                    {
                        "model": label,
                        "metric": "movement_mean_control_gap",
                        "condition": "mapping_minus_controls",
                        "split_type": "held_out_templates",
                        "position": getattr(row, "position", None),
                        "layer": row.layer,
                        "value": row.mean_control_gap,
                    }
                )

        probe = read_csv(report_dir, "probe.csv")
        if probe is not None:
            gaps = layer_control_gaps(
                probe[probe["eval_type"] == "condition_prob"],
                "p_target",
            )
            for row in gaps.itertuples():
                records.append(
                    {
                        "model": label,
                        "metric": "source_target_probe_mean_control_gap",
                        "condition": "mapping_minus_controls",
                        "split_type": "held_out_templates",
                        "position": getattr(row, "position", None),
                        "layer": row.layer,
                        "value": row.mean_control_gap,
                    }
                )

        detector_summary = None
        detector = read_csv(report_dir, "mapping_detector.csv")
        if detector is not None:
            detector_summary = (
                detector.groupby(["split_type", "position", "layer"]).roc_auc
                .mean()
                .reset_index()
            )
            for row in detector_summary.itertuples():
                records.append(
                    {
                        "model": label,
                        "metric": "mapping_detector_roc_auc",
                        "condition": "mapping_vs_controls",
                        "split_type": row.split_type,
                        "position": row.position,
                        "layer": row.layer,
                        "value": row.roc_auc,
                    }
                )

        random_summary = None
        random_detector = read_csv(
            report_dir,
            "mapping_detector_random_label_controls.csv",
        )
        if random_detector is not None:
            random_summary = (
                random_detector.groupby(
                    ["split_type", "position", "layer"]
                ).roc_auc.mean().reset_index()
            )
            for row in random_summary.itertuples():
                records.append(
                    {
                        "model": label,
                        "metric": "mapping_detector_random_label_roc_auc",
                        "condition": "random_labels",
                        "split_type": row.split_type,
                        "position": row.position,
                        "layer": row.layer,
                        "value": row.roc_auc,
                    }
                )
        if detector_summary is not None and random_summary is not None:
            adjusted = detector_summary.merge(
                random_summary,
                on=["split_type", "position", "layer"],
                suffixes=("_detector", "_random"),
            )
            for row in adjusted.itertuples():
                records.append(
                    {
                        "model": label,
                        "metric": "mapping_detector_minus_random_roc_auc",
                        "condition": "detector_minus_random_labels",
                        "split_type": row.split_type,
                        "position": row.position,
                        "layer": row.layer,
                        "value": (
                            row.roc_auc_detector - row.roc_auc_random
                        ),
                    }
                )
    return pd.DataFrame(records, columns=columns)


def best_comparison_rows(
    comparison: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    subset = comparison[comparison["metric"] == metric].dropna(
        subset=["position", "layer"]
    )
    if subset.empty:
        return subset
    group_columns = ["model", "position"]
    if subset["split_type"].notna().any():
        group_columns.insert(1, "split_type")
    best_indices = subset.groupby(group_columns)["value"].idxmax()
    return subset.loc[best_indices].sort_values(group_columns)


def mean_comparison_rows(
    comparison: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    subset = comparison[comparison["metric"] == metric].dropna(
        subset=["position", "layer"]
    )
    if subset.empty:
        return subset
    group_columns = ["model", "position"]
    if subset["split_type"].notna().any():
        group_columns.insert(1, "split_type")
    return (
        subset.groupby(group_columns).value.mean().reset_index()
        .sort_values(group_columns)
    )


def render_comparison(
    report_dirs: list[Path],
    labels: list[str],
    comparison: pd.DataFrame,
) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print("\n=== Base/Instruct comparison ===")
        print(
            "Runs: "
            + ", ".join(
                f"{label}={report_dir.resolve()}"
                for label, report_dir in zip(labels, report_dirs)
            )
        )
        if comparison.empty:
            print("\nNo comparable report CSVs were found.")
            return buffer.getvalue()

        behavior = comparison[
            comparison["metric"] == "behavior_p_target"
        ]
        if len(behavior):
            print("\nBehavioral P(target) by condition:")
            print_table(
                behavior[
                    ["model", "condition", "value"]
                ].sort_values(["condition", "model"])
            )

        for metric, title in [
            (
                "movement_mean_control_gap",
                "Best held-out movement mapping-control gap by position:",
            ),
            (
                "source_target_probe_mean_control_gap",
                "Best source-target probe mapping-control gap by position:",
            ),
            (
                "mapping_detector_roc_auc",
                "Best mapping-detector ROC AUC by split and position:",
            ),
            (
                "mapping_detector_minus_random_roc_auc",
                "Best detector-minus-random ROC AUC by split and position:",
            ),
        ]:
            mean_rows = mean_comparison_rows(comparison, metric)
            if len(mean_rows):
                print(f"\n{title.replace('Best ', 'Mean across layers: ')}")
                print_table(mean_rows)
            best = best_comparison_rows(comparison, metric)
            if len(best):
                print(f"\n{title}")
                columns = [
                    column
                    for column in [
                        "model",
                        "split_type",
                        "position",
                        "layer",
                        "value",
                    ]
                    if column in best and best[column].notna().any()
                ]
                print_table(best[columns])
    return buffer.getvalue()


def render_summary(report_dir: Path, top_k: int) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(f"Report directory: {report_dir.resolve()}")
        summarize_behavior(report_dir)
        summarize_movement(report_dir, top_k)
        summarize_probes(report_dir, top_k)
        summarize_mapping_detector(report_dir, top_k)
        summarize_patching(report_dir, top_k)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize existing concept-redefinition CSV reports."
    )
    parser.add_argument(
        "report_dirs",
        nargs="+",
        type=Path,
        help="One or more directories containing experiment report CSVs.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels matching report_dirs, e.g. --labels Base Instruct.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Summary text path. Defaults to analysis_summary.txt for one run "
            "or comparison_summary.txt in the common parent for multiple runs."
        ),
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=None,
        help="Long-form comparison CSV path for multi-run analysis.",
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="For multiple runs, omit the full per-run summaries.",
    )
    args = parser.parse_args()

    for report_dir in args.report_dirs:
        if not report_dir.is_dir():
            raise SystemExit(f"Report directory does not exist: {report_dir}")
    labels = args.labels or [path.name for path in args.report_dirs]
    if len(labels) != len(args.report_dirs):
        raise SystemExit("--labels must have one value per report directory")

    summary_parts = []
    if not args.comparison_only or len(args.report_dirs) == 1:
        for report_dir, label in zip(args.report_dirs, labels):
            if len(args.report_dirs) > 1:
                summary_parts.append(f"\n######## {label} ########\n")
            summary_parts.append(render_summary(report_dir, args.top_k))

    comparison = pd.DataFrame()
    if len(args.report_dirs) > 1:
        comparison = comparison_records(args.report_dirs, labels)
        summary_parts.append(
            render_comparison(
                args.report_dirs,
                labels,
                comparison,
            )
        )
    summary = "".join(summary_parts)
    print(summary, end="")

    if len(args.report_dirs) == 1:
        default_parent = args.report_dirs[0]
        default_name = "analysis_summary.txt"
    else:
        default_parent = Path(
            os.path.commonpath(
                [str(path.resolve()) for path in args.report_dirs]
            )
        )
        default_name = "comparison_summary.txt"
    output = args.output or default_parent / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summary, encoding="utf-8")
    print(f"\nSaved summary to {output}")

    if len(args.report_dirs) > 1:
        comparison_output = (
            args.comparison_csv
            or default_parent / "model_comparison.csv"
        )
        comparison_output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(comparison_output, index=False)
        print(f"Saved comparison CSV to {comparison_output}")


if __name__ == "__main__":
    main()
