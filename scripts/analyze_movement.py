#!/usr/bin/env python
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from redef.utils import (
    artifact_dir,
    cosine,
    load_yaml,
    read_jsonl,
    report_dir,
    validate_activation_artifacts,
)


CONDITIONS = ["mapping", "mention", "negation", "identity", "reverse", "unrelated"]


def anchor_positions(position: str) -> tuple[str, str]:
    if position in {"definition_source", "definition_target"}:
        return "definition_source", "definition_target"
    return position, position


def finite_vector(vector: np.ndarray) -> bool:
    return bool(np.isfinite(vector).all())


def position_grid(positions: list[str]):
    ncols = min(3, len(positions))
    nrows = int(np.ceil(len(positions) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
        sharex=True,
    )
    return fig, axes.ravel()


def plot_by_position(
    df: pd.DataFrame,
    positions: list[str],
    value: str,
    output,
    title: str,
    ylabel: str,
    template_split: str | None = None,
) -> None:
    fig, axes = position_grid(positions)
    for axis, position in zip(axes, positions):
        psub = df[df.position == position]
        if template_split is not None:
            psub = psub[psub.template_split == template_split]
        for condition in CONDITIONS:
            sub = psub[psub.condition == condition]
            if len(sub):
                stats = sub.groupby("layer")[value].agg(["mean", "sem"])
                axis.plot(stats.index, stats["mean"], marker="o", label=condition)
                axis.fill_between(
                    stats.index,
                    stats["mean"] - stats["sem"],
                    stats["mean"] + stats["sem"],
                    alpha=0.15,
                )
        axis.axhline(0, linestyle="--", linewidth=1)
        axis.set_title(position)
        axis.set_xlabel("Hooked decoder layer")
        axis.set_ylabel(ylabel)
    for axis in axes[len(positions):]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    artifact_root = artifact_dir(cfg)
    report_root = report_dir(cfg)
    data = np.load(artifact_root / "activations.npz", allow_pickle=True)
    acts = data["activations"]  # [N,P,L,D]
    layers = [int(layer) for layer in data["layers"].tolist()]
    positions = [str(position) for position in data["positions"].tolist()]
    meta_rows = read_jsonl(artifact_root / "activation_meta.jsonl")
    validate_activation_artifacts(cfg, artifact_root, acts, meta_rows)
    meta = pd.DataFrame(meta_rows).reset_index().rename(columns={"index": "row_idx"})
    position_to_index = {position: i for i, position in enumerate(positions)}

    centered = acts.copy()
    if bool(cfg["experiment"].get("use_centering", True)):
        centered = centered - np.nanmean(centered, axis=0, keepdims=True)

    records = []
    for (pair_id, template_id), group in meta.groupby(["pair_id", "template_id"]):
        condition_to_index = {
            row.condition: int(row.row_idx) for row in group.itertuples()
        }
        if not {"source_baseline", "target_baseline"} <= set(condition_to_index):
            continue
        source_row = condition_to_index["source_baseline"]
        target_row = condition_to_index["target_baseline"]
        for position in positions:
            source_position, target_position = anchor_positions(position)
            if source_position not in position_to_index or target_position not in position_to_index:
                continue
            source_pi = position_to_index[source_position]
            target_pi = position_to_index[target_position]
            for layer_i, layer in enumerate(layers):
                source = centered[source_row, source_pi, layer_i]
                target = centered[target_row, target_pi, layer_i]
                if not finite_vector(source) or not finite_vector(target):
                    continue
                for condition, row_i in condition_to_index.items():
                    vector = centered[row_i, position_to_index[position], layer_i]
                    if not finite_vector(vector):
                        continue
                    records.append(
                        {
                            "pair_id": pair_id,
                            "template_id": template_id,
                            "pair_split": group.iloc[0].pair_split,
                            "template_split": group.iloc[0].template_split,
                            "condition": condition,
                            "position": position,
                            "layer": layer,
                            "cos_to_target_minus_source": (
                                cosine(vector, target) - cosine(vector, source)
                            ),
                            "cos_to_target": cosine(vector, target),
                            "cos_to_source": cosine(vector, source),
                        }
                    )
    movement = pd.DataFrame(records)

    adjusted = []
    for keys, group in movement.groupby(
        ["pair_id", "template_id", "position", "layer"]
    ):
        values = {
            row.condition: row.cos_to_target_minus_source
            for row in group.itertuples()
        }
        for control in CONDITIONS[1:]:
            if "mapping" in values and control in values:
                adjusted.append(
                    {
                        "pair_id": keys[0],
                        "template_id": keys[1],
                        "position": keys[2],
                        "layer": keys[3],
                        "adjustment_control": control,
                        "mapping_minus_control": (
                            values["mapping"] - values[control]
                        ),
                    }
                )
    pd.DataFrame(adjusted).to_csv(
        report_root / "movement_control_adjusted.csv",
        index=False,
    )

    projection_records = []
    for pair_id, pair_group in meta.groupby("pair_id"):
        train_templates = pair_group[
            pair_group.template_split == "train_template"
        ].template_id.unique()
        for position in positions:
            source_position, target_position = anchor_positions(position)
            if source_position not in position_to_index or target_position not in position_to_index:
                continue
            source_pi = position_to_index[source_position]
            target_pi = position_to_index[target_position]
            eval_pi = position_to_index[position]
            for layer_i, layer in enumerate(layers):
                directions = []
                for template_id in train_templates:
                    subset = pair_group[pair_group.template_id == template_id]
                    condition_to_index = {
                        row.condition: int(row.row_idx) for row in subset.itertuples()
                    }
                    if not {"source_baseline", "target_baseline"} <= set(
                        condition_to_index
                    ):
                        continue
                    source = centered[
                        condition_to_index["source_baseline"],
                        source_pi,
                        layer_i,
                    ]
                    target = centered[
                        condition_to_index["target_baseline"],
                        target_pi,
                        layer_i,
                    ]
                    if finite_vector(source) and finite_vector(target):
                        directions.append(target - source)
                if not directions:
                    continue
                direction = np.mean(directions, axis=0)
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                for template_id, subset in pair_group.groupby("template_id"):
                    condition_to_index = {
                        row.condition: int(row.row_idx) for row in subset.itertuples()
                    }
                    if "source_baseline" not in condition_to_index:
                        continue
                    source = centered[
                        condition_to_index["source_baseline"],
                        source_pi,
                        layer_i,
                    ]
                    if not finite_vector(source):
                        continue
                    for condition, row_i in condition_to_index.items():
                        vector = centered[row_i, eval_pi, layer_i]
                        if not finite_vector(vector):
                            continue
                        projection_records.append(
                            {
                                "pair_id": pair_id,
                                "template_id": template_id,
                                "template_split": subset.iloc[0].template_split,
                                "pair_split": subset.iloc[0].pair_split,
                                "condition": condition,
                                "position": position,
                                "layer": layer,
                                "target_minus_source_projection": float(
                                    np.dot(vector - source, direction)
                                ),
                            }
                        )
    projection = pd.DataFrame(projection_records)
    movement.to_csv(report_root / "movement.csv", index=False)
    projection.to_csv(report_root / "movement_projection.csv", index=False)

    plot_by_position(
        movement,
        positions,
        "cos_to_target_minus_source",
        report_root / "movement_by_position.png",
        "Layerwise representational movement by token position",
        "cos(x,target) - cos(x,source)",
    )
    plot_by_position(
        projection,
        positions,
        "target_minus_source_projection",
        report_root / "movement_projection_by_position.png",
        "Held-out-template projection by token position",
        "projection onto target-source direction",
        template_split="test_template",
    )
    print(f"Wrote position-aware movement analyses to {report_root}")
    if len(movement):
        print(
            movement[movement.condition.isin(CONDITIONS)]
            .groupby(["position", "condition"])
            .cos_to_target_minus_source.mean()
        )


if __name__ == "__main__":
    main()
