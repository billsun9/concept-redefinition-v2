#!/usr/bin/env python
from __future__ import annotations

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from redef.utils import (
    artifact_dir,
    load_yaml,
    read_jsonl,
    report_dir,
    set_seed,
    validate_activation_artifacts,
)


CONDITIONS = ["mapping", "mention", "negation", "identity", "reverse", "unrelated"]


def fit_probe(features, labels, regularization_c):
    return make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            C=regularization_c,
            solver="liblinear",
            max_iter=2000,
        ),
    ).fit(features, labels)


def anchor_positions(position):
    if position in {"definition_source", "definition_target"}:
        return "definition_source", "definition_target"
    return position, position


def valid_vector(vector):
    return bool(np.isfinite(vector).all())


def baseline_examples(meta, acts, layer_i, position_to_index, position, split):
    source_position, target_position = anchor_positions(position)
    if source_position not in position_to_index or target_position not in position_to_index:
        return None, None
    rows = meta[
        (meta.template_split == split)
        & meta.condition.isin(["source_baseline", "target_baseline"])
    ]
    features = []
    labels = []
    for row in rows.itertuples():
        anchor_position = (
            source_position
            if row.condition == "source_baseline"
            else target_position
        )
        vector = acts[
            int(row.row_idx),
            position_to_index[anchor_position],
            layer_i,
        ]
        if valid_vector(vector):
            features.append(vector)
            labels.append(0 if row.condition == "source_baseline" else 1)
    if not features:
        return None, None
    return np.stack(features), np.asarray(labels)


def plot_probe(df, random_df, positions, report_root):
    ncols = min(3, len(positions))
    nrows = int(np.ceil(len(positions) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()
    for axis, position in zip(axes, positions):
        psub = df[
            (df.eval_type == "condition_prob")
            & (df.position == position)
        ]
        for condition in CONDITIONS:
            sub = psub[psub.condition == condition]
            if len(sub):
                stats = sub.groupby("layer").p_target.agg(["mean", "sem"])
                axis.plot(stats.index, stats["mean"], marker="o", label=condition)
                axis.fill_between(
                    stats.index,
                    stats["mean"] - stats["sem"],
                    stats["mean"] + stats["sem"],
                    alpha=0.15,
                )
        random_sub = random_df[
            (random_df.position == position)
            & (random_df.condition == "mapping")
        ]
        if len(random_sub):
            mean = random_sub.groupby("layer").p_target_random_label.mean()
            axis.plot(
                mean.index,
                mean.values,
                linestyle="--",
                label="mapping random labels",
            )
        axis.axhline(0.5, linestyle="--", linewidth=1)
        axis.set_title(position)
        axis.set_xlabel("Hooked decoder layer")
        axis.set_ylabel("Probe P(target concept)")
    for axis in axes[len(positions):]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Pair-specific source-vs-target probes by token position")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(report_root / "probe_by_position.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["run"].get("seed", 0))
    artifact_root = artifact_dir(cfg)
    report_root = report_dir(cfg)
    data = np.load(artifact_root / "activations.npz", allow_pickle=True)
    acts = data["activations"]
    layers = [int(layer) for layer in data["layers"].tolist()]
    positions = [str(position) for position in data["positions"].tolist()]
    activation_meta = read_jsonl(artifact_root / "activation_meta.jsonl")
    validate_activation_artifacts(cfg, artifact_root, acts, activation_meta)
    meta = pd.DataFrame(activation_meta).reset_index().rename(
        columns={"index": "row_idx"}
    )
    position_to_index = {position: i for i, position in enumerate(positions)}
    regularization_c = float(cfg["probe"].get("regularization_C", 0.05))
    n_random = int(cfg["probe"].get("n_random_label_controls", 20))
    rng = random.Random(cfg["run"].get("seed", 0))

    records = []
    random_records = []
    for position in positions:
        eval_position_i = position_to_index[position]
        for layer_i, layer in enumerate(layers):
            for pair_id, pair_group in meta.groupby("pair_id"):
                train_x, train_y = baseline_examples(
                    pair_group,
                    acts,
                    layer_i,
                    position_to_index,
                    position,
                    "train_template",
                )
                if (
                    train_x is None
                    or len(train_y) < 4
                    or len(np.unique(train_y)) < 2
                ):
                    continue
                try:
                    classifier = fit_probe(train_x, train_y, regularization_c)
                except Exception:
                    continue

                for split in ["train_template", "test_template"]:
                    test_x, test_y = baseline_examples(
                        pair_group,
                        acts,
                        layer_i,
                        position_to_index,
                        position,
                        split,
                    )
                    if (
                        test_x is not None
                        and len(test_y) >= 2
                        and len(np.unique(test_y)) == 2
                    ):
                        probabilities = classifier.predict_proba(test_x)[:, 1]
                        records.append(
                            {
                                "pair_id": pair_id,
                                "position": position,
                                "layer": layer,
                                "eval_type": "baseline_auc",
                                "template_split": split,
                                "condition": "source_vs_target_baseline",
                                "auc": roc_auc_score(test_y, probabilities),
                                "p_target": np.nan,
                            }
                        )

                test_rows = pair_group[
                    pair_group.template_split == "test_template"
                ]
                valid_rows = []
                features = []
                for row in test_rows.itertuples():
                    vector = acts[
                        int(row.row_idx),
                        eval_position_i,
                        layer_i,
                    ]
                    if valid_vector(vector):
                        valid_rows.append(row)
                        features.append(vector)
                if features:
                    probabilities = classifier.predict_proba(
                        np.stack(features)
                    )[:, 1]
                    for row, probability in zip(valid_rows, probabilities):
                        records.append(
                            {
                                "pair_id": pair_id,
                                "position": position,
                                "layer": layer,
                                "eval_type": "condition_prob",
                                "template_split": row.template_split,
                                "condition": row.condition,
                                "auc": np.nan,
                                "p_target": float(probability),
                            }
                        )

                control_rows = test_rows[test_rows.condition.isin(CONDITIONS)]
                valid_control_rows = []
                control_features = []
                for row in control_rows.itertuples():
                    vector = acts[
                        int(row.row_idx),
                        eval_position_i,
                        layer_i,
                    ]
                    if valid_vector(vector):
                        valid_control_rows.append(row)
                        control_features.append(vector)
                if not control_features:
                    continue
                control_features = np.stack(control_features)
                for random_i in range(n_random):
                    permuted = train_y.copy()
                    rng.shuffle(permuted)
                    if len(np.unique(permuted)) < 2:
                        continue
                    try:
                        random_classifier = fit_probe(
                            train_x,
                            permuted,
                            regularization_c,
                        )
                    except Exception:
                        continue
                    probabilities = random_classifier.predict_proba(
                        control_features
                    )[:, 1]
                    for row, probability in zip(
                        valid_control_rows,
                        probabilities,
                    ):
                        random_records.append(
                            {
                                "pair_id": pair_id,
                                "position": position,
                                "layer": layer,
                                "random_iter": random_i,
                                "condition": row.condition,
                                "p_target_random_label": float(probability),
                            }
                        )

    probe = pd.DataFrame(records)
    random_probe = pd.DataFrame(random_records)
    probe.to_csv(report_root / "probe.csv", index=False)
    random_probe.to_csv(
        report_root / "probe_random_label_controls.csv",
        index=False,
    )

    selectivity = []
    condition_rows = probe[probe.eval_type == "condition_prob"]
    for keys, group in condition_rows.groupby(["pair_id", "position", "layer"]):
        values = group.groupby("condition").p_target.mean().to_dict()
        if "mapping" not in values:
            continue
        for control in CONDITIONS[1:]:
            if control in values:
                selectivity.append(
                    {
                        "pair_id": keys[0],
                        "position": keys[1],
                        "layer": keys[2],
                        "control": control,
                        "mapping_minus_control_probe": (
                            values["mapping"] - values[control]
                        ),
                    }
                )
    pd.DataFrame(selectivity).to_csv(
        report_root / "probe_selectivity.csv",
        index=False,
    )
    plot_probe(probe, random_probe, positions, report_root)
    print(f"Wrote position-aware source-target probe outputs to {report_root}")
    if len(condition_rows):
        print(
            condition_rows.groupby(["position", "condition"]).p_target.mean()
        )


if __name__ == "__main__":
    main()
