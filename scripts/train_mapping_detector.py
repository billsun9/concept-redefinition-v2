#!/usr/bin/env python
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
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


POSITIVE = "mapping"
NEGATIVES = ["mention", "identity", "negation", "reverse", "unrelated"]
DETECTOR_CONDITIONS = [POSITIVE, *NEGATIVES]


def fit_detector(features, labels, regularization_c):
    return make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            C=regularization_c,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
        ),
    ).fit(features, labels)


def valid_rows(meta, acts, position_i, layer_i):
    subset = meta[meta.condition.isin(DETECTOR_CONDITIONS)].copy()
    mask = [
        np.isfinite(acts[int(row.row_idx), position_i, layer_i]).all()
        for row in subset.itertuples()
    ]
    return subset[np.asarray(mask, dtype=bool)]


def split_specs(rows):
    specs = [
        (
            "held_out_templates",
            "test_template",
            rows.template_split == "train_template",
            rows.template_split == "test_template",
        ),
        (
            "held_out_pairs",
            "test_pair",
            rows.pair_split == "train_pair",
            rows.pair_split == "test_pair",
        ),
    ]
    for category in sorted(rows.concept_category.unique()):
        specs.append(
            (
                "held_out_category",
                category,
                rows.concept_category != category,
                rows.concept_category == category,
            )
        )
    return specs


def evaluate_split(
    classifier,
    rows,
    acts,
    position_i,
    layer_i,
    test_mask,
):
    test_rows = rows[test_mask]
    features = np.stack(
        [
            acts[int(row.row_idx), position_i, layer_i]
            for row in test_rows.itertuples()
        ]
    )
    labels = (test_rows.condition == POSITIVE).astype(int).to_numpy()
    probabilities = classifier.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(labels, probabilities),
        "average_precision": average_precision_score(labels, probabilities),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "n_test": len(labels),
        "test_positive_rate": float(labels.mean()),
    }
    return test_rows, labels, probabilities, metrics


def plot_detector(results, report_root):
    positions = results.position.unique().tolist()
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
        subset = results[results.position == position]
        for split_type, group in subset.groupby("split_type"):
            stats = group.groupby("layer").roc_auc.agg(["mean", "sem"])
            axis.plot(
                stats.index,
                stats["mean"],
                marker="o",
                label=split_type,
            )
            axis.fill_between(
                stats.index,
                stats["mean"] - stats["sem"].fillna(0),
                stats["mean"] + stats["sem"].fillna(0),
                alpha=0.15,
            )
        axis.axhline(0.5, linestyle="--", linewidth=1)
        axis.set_title(position)
        axis.set_xlabel("Hooked decoder layer")
        axis.set_ylabel("Mapping detector ROC AUC")
        axis.set_ylim(0, 1)
    for axis in axes[len(positions):]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Mapping-vs-control detector under strict generalization splits")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(report_root / "mapping_detector_by_position.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["run"].get("seed", 0))
    rng = np.random.default_rng(cfg["run"].get("seed", 0))
    artifact_root = artifact_dir(cfg)
    report_root = report_dir(cfg)
    data = np.load(artifact_root / "activations.npz", allow_pickle=True)
    acts = data["activations"]
    layers = [int(layer) for layer in data["layers"].tolist()]
    positions = [str(position) for position in data["positions"].tolist()]
    metadata = read_jsonl(artifact_root / "activation_meta.jsonl")
    validate_activation_artifacts(cfg, artifact_root, acts, metadata)
    meta = pd.DataFrame(metadata).reset_index().rename(
        columns={"index": "row_idx"}
    )
    detector_cfg = cfg.get("mapping_detector", {})
    regularization_c = float(detector_cfg.get("regularization_C", 0.02))
    n_random = int(detector_cfg.get("n_random_label_controls", 10))

    result_records = []
    prediction_records = []
    random_records = []
    for position_i, position in enumerate(positions):
        for layer_i, layer in enumerate(layers):
            rows = valid_rows(meta, acts, position_i, layer_i)
            for split_type, held_out_group, train_mask, test_mask in split_specs(rows):
                train_rows = rows[train_mask]
                test_rows = rows[test_mask]
                if not len(train_rows) or not len(test_rows):
                    continue
                train_labels = (
                    train_rows.condition == POSITIVE
                ).astype(int).to_numpy()
                test_labels = (
                    test_rows.condition == POSITIVE
                ).astype(int).to_numpy()
                if (
                    len(np.unique(train_labels)) < 2
                    or len(np.unique(test_labels)) < 2
                ):
                    continue
                train_features = np.stack(
                    [
                        acts[int(row.row_idx), position_i, layer_i]
                        for row in train_rows.itertuples()
                    ]
                )
                classifier = fit_detector(
                    train_features,
                    train_labels,
                    regularization_c,
                )
                evaluated_rows, labels, probabilities, metrics = evaluate_split(
                    classifier,
                    rows,
                    acts,
                    position_i,
                    layer_i,
                    test_mask,
                )
                result_records.append(
                    {
                        "position": position,
                        "layer": layer,
                        "split_type": split_type,
                        "held_out_group": held_out_group,
                        "n_train": len(train_rows),
                        "train_positive_rate": float(train_labels.mean()),
                        **metrics,
                    }
                )
                for row, label, probability in zip(
                    evaluated_rows.itertuples(),
                    labels,
                    probabilities,
                ):
                    prediction_records.append(
                        {
                            "example_id": row.example_id,
                            "pair_id": row.pair_id,
                            "pair_group_id": row.pair_group_id,
                            "concept_category": row.concept_category,
                            "template_id": row.template_id,
                            "condition": row.condition,
                            "position": position,
                            "layer": layer,
                            "split_type": split_type,
                            "held_out_group": held_out_group,
                            "is_mapping": int(label),
                            "p_mapping": float(probability),
                        }
                    )

                for random_i in range(n_random):
                    permuted_labels = rng.permutation(train_labels)
                    if len(np.unique(permuted_labels)) < 2:
                        continue
                    random_classifier = fit_detector(
                        train_features,
                        permuted_labels,
                        regularization_c,
                    )
                    _, _, _, random_metrics = evaluate_split(
                        random_classifier,
                        rows,
                        acts,
                        position_i,
                        layer_i,
                        test_mask,
                    )
                    random_records.append(
                        {
                            "position": position,
                            "layer": layer,
                            "split_type": split_type,
                            "held_out_group": held_out_group,
                            "random_iter": random_i,
                            **random_metrics,
                        }
                    )

    results = pd.DataFrame(result_records)
    predictions = pd.DataFrame(prediction_records)
    random_results = pd.DataFrame(random_records)
    results.to_csv(report_root / "mapping_detector.csv", index=False)
    predictions.to_csv(
        report_root / "mapping_detector_predictions.csv",
        index=False,
    )
    random_results.to_csv(
        report_root / "mapping_detector_random_label_controls.csv",
        index=False,
    )
    if len(results):
        plot_detector(results, report_root)
        print(
            results.groupby(["split_type", "position"]).roc_auc.mean()
        )
    print(f"Wrote mapping detector outputs to {report_root}")


if __name__ == "__main__":
    main()
