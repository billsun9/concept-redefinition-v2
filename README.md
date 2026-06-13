# Concept Redefinition V2

This repo runs local white-box experiments for **in-context lexical/concept redefinition**.

The V1 scaffold was intentionally exploratory. V2 addresses the most important methodological problems:

- no per-example oracle delta as the main causal result;
- explicit matched target-mention controls;
- held-out template evaluation;
- activations at definition, query, pre-answer, and answer-label positions;
- pair-specific probes trained only on baseline anchors and evaluated on held-out templates;
- a mapping-vs-control detector with held-out template, pair, and category splits;
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

This uses a tiny GPT-2 model on CPU. The results are not scientifically
meaningful; the purpose is to verify the pipeline.

The smoke config explicitly uses CPU so it also works on nodes whose installed
PyTorch build is incompatible with the available NVIDIA driver.

```bash
bash scripts/run_all.sh config/smoke.yaml
```

Outputs go to:

```text
results/smoke/
```

By default, both large intermediates and small reports use this directory.

---

## Run profiles

Three explicit configurations are provided:

| Profile | Config | Model | Run ID |
| --- | --- | --- | --- |
| `smoke` | `config/smoke.yaml` | `sshleifer/tiny-gpt2` | `smoke` |
| `base` | `config/qwen_base.yaml` | `Qwen/Qwen2.5-7B` | `qwen25_7b_base_v2` |
| `instruct` | `config/qwen_instruct.yaml` | `Qwen/Qwen2.5-7B-Instruct` | `qwen25_7b_instruct_v2` |

`run.id` is an experiment identifier and output namespace. It is recorded in
metadata and is normally used as the final artifact/report directory name. It
does not select or load the model; `model.name` does that.

Run a profile directly:

```bash
bash scripts/run_all.sh config/smoke.yaml
bash scripts/run_all.sh config/qwen_base.yaml
bash scripts/run_all.sh config/qwen_instruct.yaml
```

`config/default.yaml` remains the Qwen 2.5 7B Instruct default for backwards
compatibility.

---

## A6000 runs

The default config uses Qwen 2.5 7B Instruct with the model chat template:

```bash
bash scripts/run_all.sh config/qwen_instruct.yaml
```

Outputs go to:

```text
results/qwen25_7b_instruct_v2/
```

The output roots are configured separately:

```yaml
run:
  artifact_dir: results/qwen25_7b_instruct_v2
  report_dir: results/qwen25_7b_instruct_v2
```

- `artifact_dir` stores the generated dataset, activation metadata, and
  `activations.npz`. Put this on high-capacity work storage.
- `report_dir` stores CSVs, JSON diagnostics, run metadata, and PNG plots.
  This can point to a smaller VS Code-visible home directory.

Environment variables override the YAML paths:

```bash
export REDEF_ARTIFACT_DIR=/pmglocal/bys2107/research/concept-redefinition-v2/artifacts/qwen25_7b_instruct_v2
export REDEF_REPORT_DIR=/insomnia001/home/bys2107/research/concept-redefinition-v2/visualizations/qwen25_7b_instruct_v2
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
5. selects a tracked config and matching output namespace from one argument.

Use the helper as:

```bash
./copy_helper.sh smoke
./copy_helper.sh base
./copy_helper.sh instruct
```

With no argument, it runs `smoke`. The selected profile controls the config
file, and the helper reads `RUN_ID` from `run.id` in that YAML. This prevents
smoke, base, and instruct outputs from sharing a directory.

### Analyze existing reports from home storage

Report analysis does not load a model or need access to `/pmglocal`. From the
home repository:

```bash
cd /insomnia001/home/bys2107/research/concept-redefinition-v2

python -u scripts/summarize_reports.py visualizations/smoke
python -u scripts/summarize_reports.py visualizations/qwen25_7b_base_v2
python -u scripts/summarize_reports.py visualizations/qwen25_7b_instruct_v2

python -u scripts/summarize_reports.py \
  visualizations/qwen25_7b_base_v2 \
  visualizations/qwen25_7b_instruct_v2 \
  --labels Base Instruct \
  --comparison-only
```

The script reads whichever report CSVs are present and prints:

- behavioral condition means and mapping-control gaps;
- calibrated behavioral scores;
- held-out movement projection gaps and best layers;
- probe selectivity, held-out baseline AUC, and random-label baselines;
- paired patching effects relative to each example's unpatched score.

For one directory, it writes `analysis_summary.txt` into that visualization
directory. For multiple directories, it writes `comparison_summary.txt` and
`model_comparison.csv` into their common parent directory. No change to
`copy_helper.sh` is required for report-only analysis.

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

The activation tensor has shape:

```text
[examples, positions, layers, hidden_size]
```

The configured positions are:

- `definition_source`: the first quoted source word in the `Note:` section;
- `definition_target`: the first quoted target word in the `Note:` section;
- `query_source`: the queried word in the `Question:` section;
- `final_pre_answer`: the final prompt token before the answer continuation;
- `answer_label_or_choice_token`: the teacher-forced correct A/B choice token or tokens.

Definition positions that do not exist in a baseline prompt are stored as
missing values and skipped by downstream analyses. At definition positions,
the source anchor is `source_baseline/definition_source` and the target anchor
is `target_baseline/definition_target`.

Dataset construction records exact character spans in the `Note:` and
`Question:` sections. Activation collection maps those spans through any chat
template before token alignment, so repeated words in the question or options
cannot redirect collection to the wrong token.

Because five positions are collected by default, `activations.npz` can be
roughly five times larger than the previous query-only artifact. It remains in
`run.artifact_dir`, which should point to work storage for full runs.

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
movement_by_position.png
movement_projection_by_position.png
```

These files are written to `run.report_dir`.

Two analyses are run:

1. cosine movement:

```text
cos(x, target_baseline) - cos(x, source_baseline)
```

2. projection onto target-minus-source directions estimated from train templates and evaluated on held-out templates.

Both outputs include `position` and `layer`. The relevant comparison is whether
`mapping` exceeds each of `mention`, `identity`, `negation`, `unrelated`, and
`reverse` at the same position and layer.

### 5. Linear probes

```bash
python scripts/train_probe.py config/default.yaml
```

Outputs:

```text
probe.csv
probe_selectivity.csv
probe_random_label_controls.csv
probe_by_position.png
```

These files are written to `run.report_dir`.

For each pair, position, and layer, the probe is trained only on train-template
`source_baseline` vs `target_baseline` activations. It is then evaluated on
held-out-template mapping and controls.

This still does not fully solve lexical-identity leakage, but it is less tautological than training/evaluating on the same template family. Random-label controls are included.

The important metric is selectivity:

```text
P_target(mapping) - P_target(each matched control)
```

not raw training accuracy.

### 6. Mapping-vs-control detector

```bash
python scripts/train_mapping_detector.py config/default.yaml
```

Outputs:

```text
mapping_detector.csv
mapping_detector_predictions.csv
mapping_detector_random_label_controls.csv
mapping_detector_by_position.png
```

This detector asks whether a contextual remapping rule is active rather than
whether an activation resembles the source or target concept. Positive
examples are `mapping`; negatives are `mention`, `identity`, `negation`,
`reverse`, and `unrelated`.

It is trained and evaluated independently at every position and layer under:

- held-out carrier templates;
- held-out concept-pair groups, with reverse-direction pairs kept together;
- leave-one-concept-category-out evaluation.

The detector uses balanced logistic regression, configurable L2
regularization, and shuffled-training-label controls. Primary metrics are ROC
AUC, average precision, and balanced accuracy. Category labels live in
`data/concept_categories.json`.

Important limitation: the current condition sentences use one wording family
per condition. A detector can therefore exploit definition syntax as well as
the model's internal rule state. Held-out pairs and categories remove lexical
concept leakage, but they do not remove condition-template leakage. Treat this
detector as a rule-context detector until multiple independently split
paraphrase families are added.

### 7. Phase 4 causal patching / reversal

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

Patching and vocabulary effects continue to use the `query_source` activation
slice explicitly.

### 8. Vocabulary effects, not fake decomposition

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
2. Mapping exceeds every matched control at specific positions and layers.
3. The final pre-answer state becomes target-like even if the query-source state remains source-like.
4. A mapping-vs-control detector generalizes to held-out templates, pair groups, and categories while random-label controls remain near chance.
5. Train-template mean deltas transfer to held-out templates and reduce target-like behavior when subtracted.
6. Wrong-pair and random norm-matched deltas do much less.

A weak or null result is also informative. For example, if `mention` and `mapping` look the same, then much of the apparent effect is lexical priming rather than redefinition.

---

## Recommended next improvements

- Add 200+ more concept pairs.
- Add more carrier templates, especially naturalistic passages with delayed queries.
- Add multiple definition-sentence paraphrase families and hold them out in the mapping detector.
- Add long-context distance experiments.
- Add head/MLP-specific patching instead of only whole block outputs.
- Add bootstrap confidence intervals with pair and template as statistical units.
- Add base-vs-instruct model comparisons.
- Add logit-lens/tuned-lens analysis at the same hook point.

---

## Known limitations

- Pair-specific probes can still decode lexical identity. Do not overclaim from probes alone.
- Whole-block patching can create off-distribution states. Interpret only relative to random/wrong-pair controls.
- Multi-token positions are represented by their mean decoder-block output.
- `answer_label_or_choice_token` uses the teacher-forced correct choice and can contain direct behavioral/output information; interpret it separately from pre-answer states.
- Instruct models should use their chat template; base models should not.
- `vocab_effects.py` is qualitative.
