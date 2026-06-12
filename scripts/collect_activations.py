#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from redef.utils import (load_yaml, read_jsonl, write_jsonl, artifact_dir, load_model_and_tokenizer,
                         maybe_chat_format, map_prompt_span_to_formatted, token_char_span,
                         collect_layer_token_means, select_layers, model_layers, save_json, run_metadata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out_dir = artifact_dir(cfg)
    rows = read_jsonl(cfg["data"]["generated_path"])
    model, tok, device = load_model_and_tokenizer(cfg)
    layer_prefix, layer_stack = model_layers(model)
    layers = select_layers(model, cfg["experiment"].get("layers", "all"))
    use_chat = cfg["model"].get("use_chat_template", False)
    add_special_tokens = not use_chat
    metas = []
    acts = []
    for r in tqdm(rows, desc="activations"):
        text = maybe_chat_format(tok, r["prompt"], use_chat)
        start, end = map_prompt_span_to_formatted(
            r["prompt"],
            text,
            int(r["query_char_start"]),
            int(r["query_char_end"]),
        )
        idxs = token_char_span(
            tok,
            text,
            start,
            end,
            add_special_tokens=add_special_tokens,
        )
        adict = collect_layer_token_means(
            model,
            tok,
            text,
            idxs,
            layers,
            device,
            add_special_tokens=add_special_tokens,
        )
        mat = np.stack([adict[li] for li in layers], axis=0).astype("float32")
        acts.append(mat)
        metas.append({k: r[k] for k in r if k != "prompt"} | {
            "prompt": r["prompt"],
            "formatted_prompt_len_chars": len(text),
            "query_char_start": start,
            "query_char_end": end,
            "query_token_indices": idxs,
        })
    arr = np.stack(acts, axis=0)
    np.savez_compressed(out_dir / "activations.npz", activations=arr, layers=np.array(layers), layer_prefix=layer_prefix)
    write_jsonl(out_dir / "activation_meta.jsonl", metas)
    save_json(out_dir / "run_meta_activations.json", run_metadata(cfg, cfg["data"]["generated_path"]) | {"layers": layers, "hook_point": layer_prefix})
    print(f"Saved activations {arr.shape} to {out_dir/'activations.npz'}")

if __name__ == "__main__":
    main()
