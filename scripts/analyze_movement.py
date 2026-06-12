#!/usr/bin/env python
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from redef.utils import (
    load_yaml,
    read_jsonl,
    cosine,
    artifact_dir,
    report_dir,
    validate_activation_artifacts,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    artifact_root = artifact_dir(cfg)
    report_root = report_dir(cfg)
    data = np.load(artifact_root / "activations.npz", allow_pickle=True)
    acts = data["activations"]  # [N,L,D]
    layers = data["layers"].tolist()
    meta = read_jsonl(artifact_root / "activation_meta.jsonl")
    validate_activation_artifacts(cfg, artifact_root, acts, meta)
    df = pd.DataFrame(meta).reset_index().rename(columns={"index": "row_idx"})
    use_centering = bool(cfg["experiment"].get("use_centering", True))
    centered = acts.copy()
    if use_centering:
        centered = centered - centered.mean(axis=0, keepdims=True)

    records = []
    # Raw/cosine movement metric within matched pair/template carrier.
    for (pair_id, template_id), g in df.groupby(["pair_id", "template_id"]):
        cond_to_idx = {r.condition: int(r.row_idx) for r in g.itertuples()}
        if "source_baseline" not in cond_to_idx or "target_baseline" not in cond_to_idx:
            continue
        si, ti = cond_to_idx["source_baseline"], cond_to_idx["target_baseline"]
        for cond, idx in cond_to_idx.items():
            for li, layer in enumerate(layers):
                x = centered[idx, li]
                src = centered[si, li]
                tgt = centered[ti, li]
                records.append({
                    "pair_id": pair_id, "template_id": template_id,
                    "pair_split": g.iloc[0].pair_split, "template_split": g.iloc[0].template_split,
                    "condition": cond, "layer": int(layer),
                    "cos_to_target_minus_source": cosine(x, tgt) - cosine(x, src),
                    "cos_to_target": cosine(x, tgt),
                    "cos_to_source": cosine(x, src),
                })
    mdf = pd.DataFrame(records)
    # Target-mention adjusted score: mapping movement minus mention-control movement.
    adj = []
    for keys, g in mdf.groupby(["pair_id", "template_id", "layer"]):
        d = {r.condition: r.cos_to_target_minus_source for r in g.itertuples()}
        for base in ["mention", "negation", "identity", "reverse", "unrelated"]:
            if "mapping" in d and base in d:
                adj.append({"pair_id": keys[0], "template_id": keys[1], "layer": keys[2],
                            "adjustment_control": base,
                            "mapping_minus_control": d["mapping"] - d[base]})
    pd.DataFrame(adj).to_csv(report_root / "movement_control_adjusted.csv", index=False)

    # Projection on train-template target-source directions; evaluate all templates but report split.
    proj_records = []
    for pair_id, gp in df.groupby("pair_id"):
        train_templates = gp[gp.template_split == "train_template"].template_id.unique().tolist()
        for li, layer in enumerate(layers):
            dirs = []
            for tid in train_templates:
                sub = gp[gp.template_id == tid]
                c2i = {r.condition: int(r.row_idx) for r in sub.itertuples()}
                if "source_baseline" in c2i and "target_baseline" in c2i:
                    dirs.append(centered[c2i["target_baseline"], li] - centered[c2i["source_baseline"], li])
            if not dirs:
                continue
            dvec = np.mean(dirs, axis=0)
            norm = np.linalg.norm(dvec) + 1e-8
            dvec = dvec / norm
            for tid, sub in gp.groupby("template_id"):
                c2i = {r.condition: int(r.row_idx) for r in sub.itertuples()}
                if "source_baseline" not in c2i:
                    continue
                src = centered[c2i["source_baseline"], li]
                for cond, idx in c2i.items():
                    score = float(np.dot(centered[idx, li] - src, dvec))
                    proj_records.append({"pair_id": pair_id, "template_id": tid,
                                         "template_split": sub.iloc[0].template_split,
                                         "pair_split": sub.iloc[0].pair_split,
                                         "condition": cond, "layer": int(layer),
                                         "target_minus_source_projection": score})
    pdf = pd.DataFrame(proj_records)
    mdf.to_csv(report_root / "movement.csv", index=False)
    pdf.to_csv(report_root / "movement_projection.csv", index=False)

    # Plots
    plt.figure(figsize=(9,5))
    for cond in ["mapping", "mention", "negation", "identity", "reverse", "unrelated"]:
        sub = mdf[mdf.condition == cond]
        if len(sub):
            mean = sub.groupby("layer").cos_to_target_minus_source.mean()
            sem = sub.groupby("layer").cos_to_target_minus_source.sem()
            plt.plot(mean.index, mean.values, marker="o", label=cond)
            plt.fill_between(mean.index, mean.values-sem.values, mean.values+sem.values, alpha=0.15)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Hooked decoder layer")
    plt.ylabel("cos(x,target) - cos(x,source)")
    plt.title("Layerwise representational movement with matched controls")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_root / "movement.png", dpi=160)

    plt.figure(figsize=(9,5))
    for cond in ["mapping", "mention", "negation", "identity"]:
        sub = pdf[(pdf.condition == cond) & (pdf.template_split == "test_template")]
        if len(sub):
            mean = sub.groupby("layer").target_minus_source_projection.mean()
            sem = sub.groupby("layer").target_minus_source_projection.sem()
            plt.plot(mean.index, mean.values, marker="o", label=cond)
            plt.fill_between(mean.index, mean.values-sem.values, mean.values+sem.values, alpha=0.15)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Hooked decoder layer")
    plt.ylabel("projection onto held-out target-source direction")
    plt.title("Held-out-template projection analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_root / "movement_projection.png", dpi=160)
    print(f"Wrote movement analyses to {report_root}")
    print(mdf[mdf.condition.isin(["mapping","mention","negation","identity"])].groupby("condition").cos_to_target_minus_source.mean())

if __name__ == "__main__":
    main()
