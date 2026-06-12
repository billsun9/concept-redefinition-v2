# Concept Redefinition V2

This repo runs local white-box experiments for **in-context lexical/concept redefinition**.

The V1 scaffold was intentionally exploratory. V2 addresses the most important methodological problems:

- no per-example oracle delta as the main causal result;
- explicit matched target-mention controls;
- held-out template evaluation;
- pair-specific probes trained only on baseline anchors and evaluated on held-out templates;
- random-label probe controls;
- non-oracle train-template mean deltas for patching;
- wrong-pair and norm-matched random-vector patch controls;
- donor and recipient activations captured/patched at the exact same decoder-block hook point;
- forced-choice A/B label scoring instead of comparing unequal natural-language answer strings;
- run metadata, dataset hashes, package versions, and separate output directories per run.

The strongest defensible description is still **a pilot study of contextual lexical remapping**, not proof that a model literally “redefines concepts” in a human-like way.

---

## Install

```bash
cd concept-redefinition-v2
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Run tests:

```bash
pytest -q
```

The tests require downloading `sshleifer/tiny-gpt2` once from Hugging Face.

---

## Smoke test

This uses a tiny GPT-2 model and should run on CPU/GPU. The results are not scientifically meaningful; the purpose is to verify the pipeline.

```bash
bash scripts/run_all.sh config/smoke.yaml
```

Outputs go to:

```text
results/smoke/
```

By default, both large intermediates and small reports use this directory.

---

## A6000 run

The default config uses Qwen 2.5 7B Instruct with the model chat template:

```bash
bash scripts/run_all.sh config/default.yaml
```

Outputs go to:

```text
results/qwen25_7b_v2/
```

The output roots are configured separately:

```yaml
run:
  artifact_dir: results/qwen25_7b_v2
  report_dir: results/qwen25_7b_v2
```

- `artifact_dir` stores the generated dataset, activation metadata, and
  `activations.npz`. Put this on high-capacity work storage.
- `report_dir` stores CSVs, JSON diagnostics, run metadata, and PNG plots.
  This can point to a smaller VS Code-visible home directory.

Environment variables override the YAML paths:

```bash
export REDEF_ARTIFACT_DIR=/pmglocal/bys2107/research/concept-redefinition-v2/artifacts/qwen25_7b_v2
export REDEF_REPORT_DIR=/insomnia001/home/bys2107/research/concept-redefinition-v2/visualizations/qwen25_7b_v2
export REDEF_HF_CACHE_DIR=/pmglocal/bys2107/research/huggingface
```

`REDEF_HF_CACHE_DIR` is passed to both Hugging Face tokenizer and model
loading. If it is unset and `model.cache_dir` is `null`, Hugging Face uses its
normal cache configuration, including `HF_HOME`.

For the home/work split used on Insomnia, keep a local `copy_helper.sh` in the
repository root. It is intentionally ignored by Git because its paths are
user- and cluster-specific. The included local template:

1. copies source from `/insomnia001/home/...` to `/pmglocal/...`;
2. preserves existing work-directory artifacts during `rsync --delete`;
3. keeps model, Torch, generic, and temporary caches under `/pmglocal`;
4. writes CSVs, JSON diagnostics, metadata, and plots back to
   `visualizations/<run-id>/` in the home checkout;
5. runs `bash scripts/run_all.sh config/default.yaml` from the copied work
   checkout.

You can change the model in `config/default.yaml`. Good alternatives:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct
  dtype: bfloat16
  use_chat_template: true
```

or a base model:

```yaml
model:
  name: Qwen/Qwen2.5-7B
  dtype: bfloat16
  use_chat_template: false
```

For final experiments, prefer comparing at least one base model and one instruct model.

---

## Pipeline

### 1. Dataset construction

```bash
python scripts/make_dataset.py config/default.yaml
```

Creates matched prompts for each concept pair and template.

Conditions:

- `source_baseline`: source word has ordinary meaning.
- `target_baseline`: target word has ordinary meaning.
- `mapping`: source means target.
- `mention`: source and target are both mentioned, no mapping.
- `negation`: source explicitly does not mean target.
- `identity`: source means source; target is mentioned.
- `reverse`: target means source; source is queried.
- `unrelated`: source means unrelated word; target is mentioned.

This directly addresses the target-token exposure confound.

### 2. Behavioral forced-choice scoring

```bash
python scripts/run_behavior.py config/default.yaml
```

Outputs:

```text
behavior.csv
behavior_calibrated.csv
```

These files are written to `run.report_dir`.

The model chooses between randomized A/B labels, not between different-length natural-language completions. This avoids the most obvious sequence-length confound.

The calibrated score is:

```text
(condition_score - source_baseline_score) / (target_baseline_score - source_baseline_score)
```

where the condition score is target-label logprob minus source-label logprob.

### 3. Activation collection at exact hook points

```bash
python scripts/collect_activations.py config/default.yaml
```

Outputs:

```text
activations.npz
activation_meta.jsonl
```

These files are written to `run.artifact_dir`. `activations.npz` is the main
large experiment output.

Activations are captured with decoder-block forward hooks. Phase 4 patching uses the same hook interface, avoiding the final-hidden-state / block-output mismatch.

Dataset construction records the exact character span of the queried word in the `Question:` line. Activation collection maps that span through any chat-template wrapper before token alignment, so repeated words in answer options cannot redirect collection to the wrong token.

Downstream activation analyses validate the dataset hash, model identity, chat-template setting, row order, and row count before reusing saved artifacts.

### 4. Layerwise movement analysis

```bash
python scripts/analyze_movement.py config/default.yaml
```

Outputs:

```text
movement.csv
movement_control_adjusted.csv
movement_projection.csv
movement.png
movement_projection.png
```

These files are written to `run.report_dir`.

Two analyses are run:

1. cosine movement:

```text
cos(x, target_baseline) - cos(x, source_baseline)
```

2. projection onto target-minus-source directions estimated from train templates and evaluated on held-out templates.

The relevant plot is not just whether `mapping` moves toward target; it is whether `mapping` exceeds `mention`, `negation`, and `identity` controls.

### 5. Linear probes

```bash
python scripts/train_probe.py config/default.yaml
```

Outputs:

```text
probe.csv
probe_selectivity.csv
probe_random_label_controls.csv
probe.png
```

These files are written to `run.report_dir`.

For each pair/layer, the probe is trained only on train-template `source_baseline` vs `target_baseline` activations. It is then evaluated on held-out-template mapping and controls.

This still does not fully solve lexical-identity leakage, but it is less tautological than training/evaluating on the same template family. Random-label controls are included.

The important metric is selectivity:

```text
P_target(mapping) - P_target(mention/negation/identity)
```

not raw training accuracy.

### 6. Phase 4 causal patching / reversal

```bash
python scripts/patch_activations.py config/default.yaml
```

Outputs:

```text
patching.csv
patching.png
```

These files are written to `run.report_dir`.

Main intervention:

```text
subtract_train_mean_delta
```

where:

```text
delta = mean_over_train_templates(h_mapping - h_source_baseline)
```

This delta is then applied to held-out mapping prompts. This is no longer the per-example oracle delta.

Controls:

- `subtract_wrong_pair_delta`
- `subtract_random_norm_matched`
- `replace_with_source_baseline` positive control
- `replace_with_target_baseline` positive control
- `add_train_mean_delta`

`replace_with_source_baseline` is still an oracle-ish positive control and should not be interpreted as evidence for reusable directions. The main causal claim should rely on train-template delta transfer beating wrong-pair/random controls.

### 7. Vocabulary effects, not fake decomposition

```bash
python scripts/vocab_effects.py config/default.yaml
```

Outputs:

```text
vocab_effects.jsonl
```

This file is written to `run.report_dir`.

V2 intentionally renames this from “decomposition” to “vocabulary effects.” Token embeddings are a correlated overcomplete dictionary, not a basis. The script reports:

- top input-embedding cosine tokens for train-template deltas;
- top unembedding logit-effect tokens.

Treat these as qualitative diagnostics, not a unique explanation.

---

## What would count as a promising result?

A strong pilot result would look like:

1. Behavioral mapping condition is target-like, while mention/negation/identity controls remain source-like.
2. Mapping activations move toward target more than matched target-mention controls.
3. Pair-specific probes trained on baseline anchors classify held-out mapping as target-like more than controls and random-label probes.
4. Train-template mean deltas transfer to held-out templates and reduce target-like behavior when subtracted.
5. Wrong-pair and random norm-matched deltas do much less.

A weak or null result is also informative. For example, if `mention` and `mapping` look the same, then much of the apparent effect is lexical priming rather than redefinition.

---

## Recommended next improvements

- Add 200+ more concept pairs.
- Add more carrier templates, especially naturalistic passages with delayed queries.
- Add long-context distance experiments.
- Add head/MLP-specific patching instead of only whole block outputs.
- Add bootstrap confidence intervals with pair and template as statistical units.
- Add base-vs-instruct model comparisons.
- Add explicit held-out pair direction transfer, not just held-out template transfer.
- Add logit-lens/tuned-lens analysis at the same hook point.

---

## Known limitations

- Pair-specific probes can still decode lexical identity. Do not overclaim from probes alone.
- Whole-block patching can create off-distribution states. Interpret only relative to random/wrong-pair controls.
- The activation is the mean over the token span corresponding to the explicitly recorded queried word in the `Question:` line.
- Instruct models should use their chat template; base models should not.
- `vocab_effects.py` is qualitative.
