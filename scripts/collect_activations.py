#!/usr/bin/env python
from __future__ import annotations
import argparse
import numpy as np
import torch
from tqdm import tqdm
from redef.utils import (load_yaml, read_jsonl, write_jsonl, artifact_dir, load_model_and_tokenizer,
                         maybe_chat_format, map_prompt_span_to_formatted, token_char_span,
                         collect_layer_position_means, select_layers, model_layers, save_json, run_metadata)


def mapped_token_indices(tok, raw_prompt, formatted_prompt, row, prefix, add_special_tokens):
    raw_start = row.get(f"{prefix}_char_start")
    raw_end = row.get(f"{prefix}_char_end")
    if raw_start is None or raw_end is None:
        return []
    start, end = map_prompt_span_to_formatted(
        raw_prompt,
        formatted_prompt,
        int(raw_start),
        int(raw_end),
    )
    return token_char_span(
        tok,
        formatted_prompt,
        start,
        end,
        add_special_tokens=add_special_tokens,
    )


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
    positions = list(
        cfg["experiment"].get(
            "activation_positions",
            ["query_source"],
        )
    )
    use_chat = cfg["model"].get("use_chat_template", False)
    add_special_tokens = not use_chat
    metas = []
    acts = []
    for r in tqdm(rows, desc="activations"):
        text = maybe_chat_format(tok, r["prompt"], use_chat)
        prompt_enc = tok(
            text,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        prompt_len = int(prompt_enc.input_ids.shape[1])
        answer_ids = tok(
            " " + r["correct_label"],
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        input_ids = torch.cat([prompt_enc.input_ids, answer_ids], dim=1).to(device)

        position_token_indices = {
            "definition_source": mapped_token_indices(
                tok,
                r["prompt"],
                text,
                r,
                "definition_source",
                add_special_tokens,
            ),
            "definition_target": mapped_token_indices(
                tok,
                r["prompt"],
                text,
                r,
                "definition_target",
                add_special_tokens,
            ),
            "query_source": mapped_token_indices(
                tok,
                r["prompt"],
                text,
                r,
                "query",
                add_special_tokens,
            ),
            "final_pre_answer": [prompt_len - 1],
            "answer_label_or_choice_token": list(
                range(prompt_len, int(input_ids.shape[1]))
            ),
        }
        unknown = set(positions) - set(position_token_indices)
        if unknown:
            raise ValueError(f"Unknown activation positions: {sorted(unknown)}")
        selected_indices = {
            position: position_token_indices[position] for position in positions
        }
        adict = collect_layer_position_means(
            model,
            input_ids,
            selected_indices,
            layers,
        )

        hidden_size = int(model.config.hidden_size)
        mat = np.full(
            (len(positions), len(layers), hidden_size),
            np.nan,
            dtype=np.float32,
        )
        for pi, position in enumerate(positions):
            for li, layer in enumerate(layers):
                if layer in adict[position]:
                    mat[pi, li] = adict[position][layer]
        acts.append(mat)
        metas.append({k: r[k] for k in r if k != "prompt"} | {
            "prompt": r["prompt"],
            "formatted_prompt_len_chars": len(text),
            "position_token_indices": selected_indices,
            "query_token_indices": position_token_indices["query_source"],
        })
    arr = np.stack(acts, axis=0)
    np.savez_compressed(
        out_dir / "activations.npz",
        activations=arr,
        layers=np.array(layers),
        positions=np.array(positions),
        layer_prefix=layer_prefix,
    )
    write_jsonl(out_dir / "activation_meta.jsonl", metas)
    save_json(
        out_dir / "run_meta_activations.json",
        run_metadata(cfg, cfg["data"]["generated_path"])
        | {
            "layers": layers,
            "positions": positions,
            "hook_point": layer_prefix,
            "activation_shape": list(arr.shape),
        },
    )
    print(
        "Saved activations "
        f"{arr.shape} [rows, positions, layers, hidden] "
        f"to {out_dir/'activations.npz'}"
    )

if __name__ == "__main__":
    main()
