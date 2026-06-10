#!/usr/bin/env python
from __future__ import annotations
import argparse, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from redef.utils import (
    load_yaml,
    read_jsonl,
    ensure_dir,
    set_seed,
    validate_activation_artifacts,
)


def fit_probe(X, y, C):
    # Small n, large d: strong L2 regularization is intentional.
    clf = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                        LogisticRegression(C=C, penalty="l2", solver="liblinear", max_iter=2000))
    clf.fit(X, y)
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["run"].get("seed", 0))
    out_dir = ensure_dir(cfg["run"]["output_dir"])
    data = np.load(out_dir / "activations.npz", allow_pickle=True)
    acts = data["activations"]
    layers = data["layers"].tolist()
    activation_meta = read_jsonl(out_dir / "activation_meta.jsonl")
    validate_activation_artifacts(cfg, out_dir, acts, activation_meta)
    meta = pd.DataFrame(activation_meta).reset_index().rename(columns={"index":"row_idx"})
    C = float(cfg["probe"].get("regularization_C", 0.05))
    n_rand = int(cfg["probe"].get("n_random_label_controls", 20))
    rng = random.Random(cfg["run"].get("seed", 0))

    records = []
    rand_records = []
    for layer_pos, layer in enumerate(layers):
        for pair_id, gp in meta.groupby("pair_id"):
            train = gp[(gp.template_split == "train_template") & (gp.condition.isin(["source_baseline", "target_baseline"]))]
            if train.condition.nunique() < 2 or len(train) < 4:
                continue
            X = acts[train.row_idx.values, layer_pos, :]
            y = np.array([0 if c == "source_baseline" else 1 for c in train.condition.values])
            if len(np.unique(y)) < 2:
                continue
            try:
                clf = fit_probe(X, y, C)
            except Exception:
                continue
            # Held-out baseline AUC by template split.
            for split in ["train_template", "test_template"]:
                test_base = gp[(gp.template_split == split) & (gp.condition.isin(["source_baseline", "target_baseline"]))]
                if len(test_base) >= 2 and test_base.condition.nunique() == 2:
                    yb = np.array([0 if c == "source_baseline" else 1 for c in test_base.condition.values])
                    pb = clf.predict_proba(acts[test_base.row_idx.values, layer_pos, :])[:,1]
                    auc = roc_auc_score(yb, pb) if len(set(yb)) == 2 else np.nan
                    records.append({"pair_id": pair_id, "layer": int(layer), "eval_type": "baseline_auc", "template_split": split,
                                    "condition": "source_vs_target_baseline", "auc": auc, "p_target": np.nan})
            # Evaluate all held-out-template conditions; this is the non-tautological probe readout.
            test = gp[gp.template_split == "test_template"]
            if len(test):
                probs = clf.predict_proba(acts[test.row_idx.values, layer_pos, :])[:,1]
                for row, p in zip(test.itertuples(), probs):
                    records.append({"pair_id": pair_id, "layer": int(layer), "eval_type": "condition_prob",
                                    "template_split": row.template_split, "condition": row.condition,
                                    "auc": np.nan, "p_target": float(p)})
            # Random-label controls: same features, permuted labels. Report held-out mapping/controls.
            for ri in range(n_rand):
                y_perm = y.copy()
                rng.shuffle(y_perm)
                if len(set(y_perm)) < 2:
                    continue
                try:
                    rclf = fit_probe(X, y_perm, C)
                except Exception:
                    continue
                test = gp[(gp.template_split == "test_template") & (gp.condition.isin(["mapping", "mention", "negation", "identity"]))]
                if len(test):
                    probs = rclf.predict_proba(acts[test.row_idx.values, layer_pos, :])[:,1]
                    for row, p in zip(test.itertuples(), probs):
                        rand_records.append({"pair_id": pair_id, "layer": int(layer), "random_iter": ri,
                                             "condition": row.condition, "p_target_random_label": float(p)})
    df = pd.DataFrame(records)
    rdf = pd.DataFrame(rand_records)
    df.to_csv(out_dir / "probe.csv", index=False)
    rdf.to_csv(out_dir / "probe_random_label_controls.csv", index=False)

    plt.figure(figsize=(9,5))
    conds = ["mapping", "mention", "negation", "identity", "reverse", "unrelated"]
    for cond in conds:
        sub = df[(df.eval_type == "condition_prob") & (df.condition == cond)]
        if len(sub):
            mean = sub.groupby("layer").p_target.mean()
            sem = sub.groupby("layer").p_target.sem()
            plt.plot(mean.index, mean.values, marker="o", label=cond)
            plt.fill_between(mean.index, mean.values-sem.values, mean.values+sem.values, alpha=0.15)
    if len(rdf):
        sub = rdf[rdf.condition == "mapping"]
        mean = sub.groupby("layer").p_target_random_label.mean()
        plt.plot(mean.index, mean.values, linestyle="--", label="mapping random-label control")
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xlabel("Hooked decoder layer")
    plt.ylabel("Probe P(target concept)")
    plt.title("Pair-specific probes trained on baseline anchors; evaluated on held-out templates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "probe.png", dpi=160)

    # Selectivity summary: mapping minus mention/negation.
    sel = []
    for keys, g in df[df.eval_type == "condition_prob"].groupby(["pair_id", "layer"]):
        d = g.groupby("condition").p_target.mean().to_dict()
        if "mapping" in d:
            for ctrl in ["mention", "negation", "identity", "reverse", "unrelated"]:
                if ctrl in d:
                    sel.append({"pair_id": keys[0], "layer": keys[1], "control": ctrl, "mapping_minus_control_probe": d["mapping"] - d[ctrl]})
    pd.DataFrame(sel).to_csv(out_dir / "probe_selectivity.csv", index=False)
    print(f"Wrote probe outputs to {out_dir}")
    if len(df):
        print(df[df.eval_type == "condition_prob"].groupby("condition").p_target.mean())

if __name__ == "__main__":
    main()
