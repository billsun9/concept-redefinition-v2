#!/usr/bin/env python
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
import torch
from redef.utils import (
    load_yaml,
    read_jsonl,
    artifact_dir,
    report_dir,
    load_model_and_tokenizer,
    validate_activation_artifacts,
)


def top_tokens(tok, scores, k):
    idx = np.argsort(scores)[::-1][:k]
    return [{"token_id": int(i), "token": tok.decode([int(i)]), "score": float(scores[i])} for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    artifact_root = artifact_dir(cfg)
    report_root = report_dir(cfg)
    data = np.load(artifact_root / "activations.npz", allow_pickle=True)
    acts = data["activations"]
    layers = data["layers"].tolist()
    activation_meta = read_jsonl(artifact_root / "activation_meta.jsonl")
    validate_activation_artifacts(cfg, artifact_root, acts, activation_meta)
    meta = pd.DataFrame(activation_meta).reset_index().rename(columns={"index":"row_idx"})
    model, tok, device = load_model_and_tokenizer(cfg)
    emb = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    # Use LM head when available; for tied embeddings this is equivalent up to final norm/interface caveats.
    if hasattr(model, "lm_head"):
        unemb = model.lm_head.weight.detach().float().cpu().numpy()
    else:
        unemb = emb
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    records = []
    for pair_id, gp in meta.groupby("pair_id"):
        for lpos, layer in enumerate(layers):
            deltas = []
            for tid, sub in gp[gp.template_split == "train_template"].groupby("template_id"):
                c2i = {r.condition: int(r.row_idx) for r in sub.itertuples()}
                if "mapping" in c2i and "source_baseline" in c2i:
                    deltas.append(acts[c2i["mapping"], lpos] - acts[c2i["source_baseline"], lpos])
            if not deltas:
                continue
            d = np.mean(deltas, axis=0)
            d_n = d / (np.linalg.norm(d) + 1e-8)
            # Not a basis decomposition: report vocabulary logit/effect directions only.
            emb_scores = emb_n @ d_n
            logit_scores = unemb @ d
            records.append({
                "pair_id": pair_id,
                "source": gp.iloc[0].source,
                "target": gp.iloc[0].target,
                "layer": int(layer),
                "delta_norm": float(np.linalg.norm(d)),
                "top_embedding_cosine_tokens": top_tokens(tok, emb_scores, args.top_k),
                "top_unembedding_logit_effect_tokens": top_tokens(tok, logit_scores, args.top_k),
            })
    with open(report_root / "vocab_effects.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} rows to {report_root/'vocab_effects.jsonl'}")

if __name__ == "__main__":
    main()
