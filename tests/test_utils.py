from transformers import AutoTokenizer
import torch
import transformers
from redef.utils import (
    final_occurrence_span,
    load_yaml,
    map_prompt_span_to_formatted,
    model_dtype_kwargs,
    patch_hidden_states,
    token_char_span,
)

def test_alignment_tiny_gpt2():
    tok = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2', use_fast=True)
    text = "dog cat dog"
    s,e = final_occurrence_span(text, 'dog')
    idxs = token_char_span(tok, text, s, e)
    assert len(idxs) >= 1


def test_map_prompt_span_to_chat_wrapper():
    raw = "Question: Is 'red' the queried word?"
    start = raw.index("red")
    formatted = f"<user>\n{raw}\n</user>\n<assistant>\n"
    mapped_start, mapped_end = map_prompt_span_to_formatted(
        raw, formatted, start, start + len("red")
    )
    assert formatted[mapped_start:mapped_end] == "red"


def test_load_yaml_applies_storage_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run:
  id: test
  artifact_dir: results/test
  report_dir: results/test
model:
  name: example/model
  cache_dir: null
data:
  pairs_path: data/concept_pairs.jsonl
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("REDEF_ARTIFACT_DIR", "/work/artifacts/test")
    monkeypatch.setenv("REDEF_REPORT_DIR", "/home/reports/test")
    monkeypatch.setenv("REDEF_HF_CACHE_DIR", "/work/huggingface")

    cfg = load_yaml(config_path)

    assert cfg["run"]["artifact_dir"] == "/work/artifacts/test"
    assert cfg["run"]["report_dir"] == "/home/reports/test"
    assert cfg["data"]["generated_path"] == "/work/artifacts/test/dataset.jsonl"
    assert cfg["model"]["cache_dir"] == "/work/huggingface"


def test_model_dtype_keyword_tracks_transformers_major_version(monkeypatch):
    monkeypatch.setattr(transformers, "__version__", "4.57.6")
    assert model_dtype_kwargs("auto") == {"torch_dtype": "auto"}

    monkeypatch.setattr(transformers, "__version__", "5.0.0")
    assert model_dtype_kwargs("auto") == {"dtype": "auto"}


def test_patch_hidden_states_casts_vector_to_recipient_dtype():
    hidden = torch.zeros((1, 3, 4), dtype=torch.bfloat16)
    patch_vector = torch.ones(4, dtype=torch.float32)

    patched = patch_hidden_states(
        hidden,
        token_indices=[1],
        patch_vector=patch_vector,
        alpha=0.5,
        mode="add",
    )

    assert patched.dtype == torch.bfloat16
    assert torch.equal(
        patched[0, 1],
        torch.full((4,), 0.5, dtype=torch.bfloat16),
    )
