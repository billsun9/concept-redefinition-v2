#!/usr/bin/env python
from __future__ import annotations
import argparse, math
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from redef.utils import load_yaml, read_jsonl, ensure_dir, load_model_and_tokenizer, maybe_chat_format, continuation_logprob, save_json, run_metadata


def sigmoid(x):
    return 1/(1+math.exp(-max(min(x, 60), -60)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out_dir = ensure_dir(cfg["run"]["output_dir"])
    rows = read_jsonl(cfg["data"]["generated_path"])
    model, tok, device = load_model_and_tokenizer(cfg)
    use_chat = cfg["model"].get("use_chat_template", False)
    add_special_tokens = not use_chat
    records = []
    for r in tqdm(rows, desc="behavior"):
        prompt = maybe_chat_format(tok, r["prompt"], use_chat)
        # Forced-choice labels are intentionally same length semantically; score exact label continuations.
        lp_a = continuation_logprob(
            model, tok, prompt, " A", device, add_special_tokens=add_special_tokens
        )
        lp_b = continuation_logprob(
            model, tok, prompt, " B", device, add_special_tokens=add_special_tokens
        )
        lp_target = lp_a if r["target_label"] == "A" else lp_b
        lp_source = lp_a if r["source_label"] == "A" else lp_b
        target_pref = lp_target - lp_source
        records.append({**{k: r[k] for k in r if k != "prompt"},
                        "prompt": r["prompt"],
                        "lp_A": lp_a, "lp_B": lp_b,
                        "lp_target_label": lp_target,
                        "lp_source_label": lp_source,
                        "target_pref_logit": target_pref,
                        "p_target_vs_source": sigmoid(target_pref),
                        "pred_label": "A" if lp_a > lp_b else "B",
                        "pred_is_target": (lp_target > lp_source)})
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "behavior.csv", index=False)
    # Calibrated score per pair/template: (mapping-source)/(target-source). Mention controls also reported.
    cal = []
    for keys, g in df.groupby(["pair_id", "template_id"]):
        d = {row.condition: row.target_pref_logit for row in g.itertuples()}
        denom = d.get("target_baseline", float("nan")) - d.get("source_baseline", float("nan"))
        for cond, val in d.items():
            cal_score = (val - d.get("source_baseline", float("nan"))) / denom if abs(denom) > 1e-8 else float("nan")
            cal.append({"pair_id": keys[0], "template_id": keys[1], "condition": cond, "calibrated_redefinition_score": cal_score})
    pd.DataFrame(cal).to_csv(out_dir / "behavior_calibrated.csv", index=False)
    save_json(out_dir / "run_meta_behavior.json", run_metadata(cfg, cfg["data"]["generated_path"]))
    print(df.groupby("condition")["p_target_vs_source"].agg(["mean","std","count"]))
    print(f"Wrote {out_dir/'behavior.csv'}")

if __name__ == "__main__":
    main()
