#!/usr/bin/env python
from __future__ import annotations
import argparse, random
from pathlib import Path
from redef.utils import (
    load_json,
    load_yaml,
    read_jsonl,
    write_jsonl,
    ensure_dir,
    set_seed,
    save_json,
    run_metadata,
)

CARRIERS = [
    "Read the note carefully and answer using only the letter of the best option.",
    "Use the definitions in the note when answering. Reply with only A or B.",
    "Consider the note below. Choose the option that matches the queried word.",
    "For this item, answer the multiple-choice question with a single letter.",
    "You are checking word meanings in a short note. Select A or B only.",
    "Decide which option the queried word refers to in the note. Output just the letter.",
]


def control_sentence(kind, s, t, u):
    if kind == "source_baseline":
        return f"The word '{s}' has its ordinary English meaning."
    if kind == "target_baseline":
        return f"The word '{t}' has its ordinary English meaning."
    if kind == "mapping":
        return f"In this note, the word '{s}' means '{t}'."
    if kind == "mention":
        return f"The words '{s}' and '{t}' are both mentioned in this note, but no special definitions are introduced."
    if kind == "negation":
        return f"In this note, the word '{s}' does not mean '{t}'; '{s}' keeps its ordinary meaning."
    if kind == "identity":
        return f"In this note, the word '{s}' means '{s}', while '{t}' keeps its ordinary meaning."
    if kind == "reverse":
        return f"In this note, the word '{t}' means '{s}', while '{s}' keeps its ordinary meaning."
    if kind == "unrelated":
        return f"In this note, the word '{s}' means '{u}', and the word '{t}' is merely mentioned."
    raise ValueError(kind)


def target_is_correct(kind):
    return kind in {"target_baseline", "mapping"}


def query_word_for(kind, s, t):
    return t if kind == "target_baseline" else s


def build_prompt(carrier, sentence, query_word, option_a, option_b):
    # Put the queried word late and mark it by quoting it; activation scripts align to this final occurrence.
    return (
        f"{carrier}\n\n"
        f"Note: {sentence}\n\n"
        f"Question: In the note, the word '{query_word}' refers to which option?\n"
        f"A. {option_a}\n"
        f"B. {option_b}\n"
        f"Answer:"
    )


def query_span_in_prompt(prompt, query_word):
    question_start = prompt.index("Question:")
    options_start = prompt.index("\nA.", question_start)
    quoted_query = f"'{query_word}'"
    quoted_start = prompt.index(quoted_query, question_start, options_start)
    start = quoted_start + 1
    return start, start + len(query_word)


def definition_span_in_prompt(prompt, word):
    note_start = prompt.index("Note:")
    note_end = prompt.index("\n\nQuestion:", note_start)
    quoted = f"'{word}'"
    quoted_start = prompt.find(quoted, note_start, note_end)
    if quoted_start < 0:
        return None
    start = quoted_start + 1
    return start, start + len(word)


def canonical_pair_key(source, target):
    return "|".join(sorted([source, target]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["run"].get("seed", 0))
    out_path = Path(cfg["data"]["generated_path"])
    ensure_dir(out_path.parent)

    pairs = read_jsonl(cfg["data"]["pairs_path"])
    categories = load_json(cfg["data"]["categories_path"])
    max_pairs = cfg["data"].get("max_pairs")
    if max_pairs:
        pairs = pairs[: int(max_pairs)]
    controls = ["source_baseline", "target_baseline"] + list(cfg["experiment"].get("controls", []))
    n_templates = int(cfg["data"].get("templates_per_pair", len(CARRIERS)))
    carriers = CARRIERS[:n_templates]
    rng = random.Random(cfg["run"].get("seed", 0))
    pair_groups = {}
    rows = []
    for pi, p in enumerate(pairs):
        pair_key = canonical_pair_key(p["source"], p["target"])
        if pair_key not in categories:
            raise ValueError(f"Missing concept category for pair {pair_key!r}")
        if pair_key not in pair_groups:
            pair_groups[pair_key] = len(pair_groups)
        pair_group_index = pair_groups[pair_key]
        pair_group_id = f"g{pair_group_index:03d}"
        pair_split = (
            "test_pair" if (pair_group_index % 5 == 4) else "train_pair"
        )
        for ti, carrier in enumerate(carriers):
            template_split = "test_template" if (ti % 3 == 2) else "train_template"
            for kind in controls:
                s, t, u = p["source"], p["target"], p["unrelated"]
                q = query_word_for(kind, s, t)
                correct_target = target_is_correct(kind)
                # Counterbalance answer order deterministically.
                target_a = rng.random() < 0.5
                if target_a:
                    option_a, option_b = p["target_desc"], p["source_desc"]
                    correct = "A" if correct_target else "B"
                    target_label, source_label = "A", "B"
                else:
                    option_a, option_b = p["source_desc"], p["target_desc"]
                    correct = "B" if correct_target else "A"
                    target_label, source_label = "B", "A"
                sent = control_sentence(kind, s, t, u)
                prompt = build_prompt(carrier, sent, q, option_a, option_b)
                query_char_start, query_char_end = query_span_in_prompt(prompt, q)
                definition_source_span = definition_span_in_prompt(prompt, s)
                definition_target_span = definition_span_in_prompt(prompt, t)
                rows.append({
                    "example_id": f"{p['pair_id']}_t{ti}_{kind}",
                    "pair_id": p["pair_id"],
                    "pair_index": pi,
                    "pair_group_id": pair_group_id,
                    "pair_split": pair_split,
                    "concept_category": categories[pair_key],
                    "template_id": ti,
                    "template_split": template_split,
                    "condition": kind,
                    "source": s,
                    "target": t,
                    "unrelated": u,
                    "source_desc": p["source_desc"],
                    "target_desc": p["target_desc"],
                    "query_word": q,
                    "target_label": target_label,
                    "source_label": source_label,
                    "correct_label": correct,
                    "prompt": prompt,
                    "query_char_start": query_char_start,
                    "query_char_end": query_char_end,
                    "definition_source_char_start": (
                        definition_source_span[0]
                        if definition_source_span is not None
                        else None
                    ),
                    "definition_source_char_end": (
                        definition_source_span[1]
                        if definition_source_span is not None
                        else None
                    ),
                    "definition_target_char_start": (
                        definition_target_span[0]
                        if definition_target_span is not None
                        else None
                    ),
                    "definition_target_char_end": (
                        definition_target_span[1]
                        if definition_target_span is not None
                        else None
                    ),
                    "target_is_correct": correct_target,
                })
    write_jsonl(out_path, rows)
    save_json(out_path.parent / "dataset_meta.json", {"n_rows": len(rows), "config": cfg, "metadata": run_metadata(cfg, out_path)})
    print(f"Wrote {len(rows)} rows to {out_path}")

if __name__ == "__main__":
    main()
