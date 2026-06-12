#!/usr/bin/env python
from __future__ import annotations
import argparse, random, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from redef.utils import (load_yaml, read_jsonl, artifact_dir, report_dir, load_model_and_tokenizer, maybe_chat_format,
                         select_layers, continuation_logprob, score_with_patch, save_json, run_metadata,
                         validate_activation_artifacts)


def sigmoid(x):
    return 1/(1+math.exp(-max(min(x, 60), -60)))


def compute_train_deltas(meta, acts, position_pos, layer_pos):
    # Non-oracle: mean over train templates per pair of mapping - source_baseline.
    out = {}
    for pair_id, gp in meta.groupby("pair_id"):
        deltas = []
        for tid, sub in gp[gp.template_split == "train_template"].groupby("template_id"):
            c2i = {r.condition: int(r.row_idx) for r in sub.itertuples()}
            if "mapping" in c2i and "source_baseline" in c2i:
                deltas.append(
                    acts[c2i["mapping"], position_pos, layer_pos]
                    - acts[c2i["source_baseline"], position_pos, layer_pos]
                )
        if deltas:
            out[pair_id] = np.mean(deltas, axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    artifact_root = artifact_dir(cfg)
    report_root = report_dir(cfg)
    rng = np.random.default_rng(cfg["run"].get("seed", 0))
    py_rng = random.Random(cfg["run"].get("seed", 0))
    data = np.load(artifact_root / "activations.npz", allow_pickle=True)
    acts = data["activations"]
    collected_layers = data["layers"].tolist()
    positions = [str(position) for position in data["positions"].tolist()]
    if "query_source" not in positions:
        raise RuntimeError(
            "Patching requires query_source activations. "
            "Add it to experiment.activation_positions and recollect activations."
        )
    query_position = positions.index("query_source")
    meta = pd.DataFrame(read_jsonl(artifact_root / "activation_meta.jsonl")).reset_index().rename(columns={"index":"row_idx"})
    validate_activation_artifacts(cfg, artifact_root, acts, meta.to_dict("records"))
    model, tok, device = load_model_and_tokenizer(cfg)
    patch_layers = select_layers(model, cfg["patching"].get("layers", collected_layers))
    patch_layers = [l for l in patch_layers if l in collected_layers]
    layer_to_pos = {int(l): i for i, l in enumerate(collected_layers)}
    alphas = [float(a) for a in cfg["patching"].get("alpha_values", [-1, -0.5, 0, 0.5, 1])]
    max_examples = int(cfg["patching"].get("max_examples", 200))
    use_chat = cfg["model"].get("use_chat_template", False)
    add_special_tokens = not use_chat

    # Candidate rows: held-out templates in mapping condition. This avoids deriving and testing on same prompt.
    rows = meta[(meta.condition == "mapping") & (meta.template_split == "test_template")].copy()
    if len(rows) > max_examples:
        rows = rows.sample(n=max_examples, random_state=cfg["run"].get("seed", 0))

    all_pairs = sorted(meta.pair_id.unique().tolist())
    records = []
    for layer in patch_layers:
        lpos = layer_to_pos[int(layer)]
        deltas = compute_train_deltas(meta, acts, query_position, lpos)
        for r in tqdm(list(rows.itertuples()), desc=f"patch layer {layer}"):
            if r.pair_id not in deltas:
                continue
            prompt = maybe_chat_format(tok, r.prompt, use_chat)
            cont_target = " " + r.target_label
            cont_source = " " + r.source_label
            # unpatched
            base_t = continuation_logprob(
                model, tok, prompt, cont_target, device, add_special_tokens=add_special_tokens
            )
            base_s = continuation_logprob(
                model, tok, prompt, cont_source, device, add_special_tokens=add_special_tokens
            )
            query_idxs = list(r.query_token_indices)
            true_delta = torch.tensor(deltas[r.pair_id], dtype=torch.float32)
            # wrong-pair with similar norm if possible
            wrong_choices = [p for p in all_pairs if p != r.pair_id and p in deltas]
            wrong_pair = py_rng.choice(wrong_choices) if wrong_choices else r.pair_id
            wrong_delta = torch.tensor(deltas[wrong_pair], dtype=torch.float32)
            rand = torch.tensor(rng.normal(size=true_delta.shape), dtype=torch.float32)
            rand = rand / (rand.norm() + 1e-8) * (true_delta.norm() + 1e-8)
            # source/target replace positive/negative controls from the same held-out template.
            sub = meta[(meta.pair_id == r.pair_id) & (meta.template_id == r.template_id)]
            c2i = {row.condition: int(row.row_idx) for row in sub.itertuples()}
            source_vec = torch.tensor(
                acts[c2i["source_baseline"], query_position, lpos],
                dtype=torch.float32,
            ) if "source_baseline" in c2i else None
            target_vec = torch.tensor(
                acts[c2i["target_baseline"], query_position, lpos],
                dtype=torch.float32,
            ) if "target_baseline" in c2i else None
            interventions = [
                ("unpatched", None, 0.0, "add", None),
                ("subtract_train_mean_delta", true_delta, None, "subtract", None),
                ("add_train_mean_delta", true_delta, None, "add", None),
                ("subtract_wrong_pair_delta", wrong_delta, None, "subtract", wrong_pair),
                ("subtract_random_norm_matched", rand, None, "subtract", None),
            ]
            if source_vec is not None:
                interventions.append(("replace_with_source_baseline", source_vec, 1.0, "replace", None))
            if target_vec is not None:
                interventions.append(("replace_with_target_baseline", target_vec, 1.0, "replace", None))

            for name, vec, alpha_marker, mode, donor in interventions:
                if name == "unpatched":
                    records.append({"example_id": r.example_id, "pair_id": r.pair_id, "template_id": r.template_id,
                                    "layer": int(layer), "intervention": name, "alpha": 0.0, "donor_pair": donor,
                                    "lp_target": base_t, "lp_source": base_s,
                                    "target_pref_logit": base_t - base_s,
                                    "p_target_vs_source": sigmoid(base_t - base_s)})
                    continue
                alpha_list = [alpha_marker] if alpha_marker is not None else alphas
                for alpha in alpha_list:
                    lt = score_with_patch(
                        model, tok, prompt, cont_target, int(layer), query_idxs, vec,
                        float(alpha), mode, device, add_special_tokens=add_special_tokens
                    )
                    ls = score_with_patch(
                        model, tok, prompt, cont_source, int(layer), query_idxs, vec,
                        float(alpha), mode, device, add_special_tokens=add_special_tokens
                    )
                    records.append({"example_id": r.example_id, "pair_id": r.pair_id, "template_id": r.template_id,
                                    "layer": int(layer), "intervention": name, "alpha": float(alpha), "donor_pair": donor,
                                    "lp_target": lt, "lp_source": ls,
                                    "target_pref_logit": lt - ls,
                                    "p_target_vs_source": sigmoid(lt - ls),
                                    "delta_norm": float(true_delta.norm())})
    df = pd.DataFrame(records)
    df.to_csv(report_root / "patching.csv", index=False)
    save_json(report_root / "run_meta_patching.json", run_metadata(cfg, cfg["data"]["generated_path"]) | {"patch_layers": patch_layers})

    if len(df):
        plt.figure(figsize=(10,5))
        plot_df = df[df.intervention.isin(["unpatched", "subtract_train_mean_delta", "subtract_wrong_pair_delta", "subtract_random_norm_matched", "replace_with_source_baseline", "replace_with_target_baseline"])]
        # For line interventions, plot alpha. For replace/unpatched, alpha fixed.
        for name, sub in plot_df.groupby("intervention"):
            g = sub.groupby("alpha").p_target_vs_source.mean().sort_index()
            plt.plot(g.index, g.values, marker="o", label=name)
        plt.xlabel("alpha")
        plt.ylabel("P(target label over source label)")
        plt.title("Held-out-template query-token patching; non-oracle train-template deltas")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(report_root / "patching.png", dpi=160)
    print(f"Wrote patching results to {report_root/'patching.csv'}")

if __name__ == "__main__":
    main()
