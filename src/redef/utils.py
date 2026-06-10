from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_or_none() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model_and_tokenizer(cfg: Dict[str, Any]):
    model_name = cfg["model"]["name"]
    device = get_device(cfg["model"].get("device", "auto"))
    dtype_name = cfg["model"].get("dtype", "auto")
    if dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float32":
        dtype = torch.float32
    else:
        dtype = "auto"
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=cfg["model"].get("trust_remote_code", False))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    return model, tok, device


def model_layers(model) -> Tuple[str, torch.nn.ModuleList]:
    # Common decoder-only layouts: GPT2, LLaMA/Qwen/Mistral/Gemma.
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "transformer.h", model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "model.layers", model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return "gpt_neox.layers", model.gpt_neox.layers
    raise ValueError("Could not infer decoder layer stack for this model. Add it to redef.utils.model_layers().")


def select_layers(model, requested: List[int] | str) -> List[int]:
    _, layers = model_layers(model)
    n = len(layers)
    if requested == "all":
        return list(range(n))
    out = []
    for x in requested:
        i = int(x)
        if i < 0:
            i = n + i
        if 0 <= i < n:
            out.append(i)
    return sorted(set(out))


def maybe_chat_format(tok, text: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return text
    if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
        msgs = [{"role": "user", "content": text}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text


def token_char_span(tok, text: str, char_start: int, char_end: int) -> List[int]:
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    idxs = []
    for i, (s, e) in enumerate(offsets):
        # special tokens often have (0,0); exclude those unless span truly intersects.
        if e <= s:
            continue
        if max(s, char_start) < min(e, char_end):
            idxs.append(i)
    if not idxs:
        raise ValueError(f"Could not align char span {char_start}:{char_end} in text: {text[:200]!r}")
    return idxs


def final_occurrence_span(text: str, needle: str) -> Tuple[int, int]:
    i = text.rfind(needle)
    if i < 0:
        raise ValueError(f"Needle {needle!r} not found in text")
    return i, i + len(needle)


def label_token_ids(tok, labels=("A", "B")) -> Dict[str, List[int]]:
    # Try variants because tokenizers differ; choose the shortest encoding.
    out = {}
    for lab in labels:
        variants = [lab, " " + lab, "\n" + lab]
        encs = []
        for v in variants:
            ids = tok(v, add_special_tokens=False).input_ids
            if ids:
                encs.append(ids)
        out[lab] = sorted(encs, key=len)[0]
    return out


@torch.no_grad()
def continuation_logprob(model, tok, prompt: str, continuation: str, device: str) -> float:
    # Total sequence log-probability for exact continuation. Suitable when continuations are same-token labels.
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    c_ids = tok(continuation, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_ids = torch.cat([p_ids, c_ids], dim=1)
    logits = model(input_ids).logits
    logp = torch.log_softmax(logits, dim=-1)
    total = 0.0
    # token j in input is predicted by logits at j-1.
    start = p_ids.shape[1]
    for pos in range(start, input_ids.shape[1]):
        tok_id = input_ids[0, pos]
        total += float(logp[0, pos - 1, tok_id].detach().cpu())
    return total


def hidden_from_module_output(output):
    # HF decoder block outputs can be tensor or tuple(hidden, ...).
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_hidden_in_output(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden,) + tuple(output[1:])
    return new_hidden


@torch.no_grad()
def collect_layer_token_means(model, tok, text: str, token_indices: List[int], layers: List[int], device: str) -> Dict[int, np.ndarray]:
    _, layer_stack = model_layers(model)
    captured: Dict[int, torch.Tensor] = {}
    handles = []
    wanted = set(layers)
    def make_hook(layer_idx):
        def hook(module, inp, out):
            h = hidden_from_module_output(out).detach()  # [B,S,D]
            captured[layer_idx] = h[0, token_indices, :].mean(dim=0).float().cpu()
        return hook
    for li in layers:
        handles.append(layer_stack[li].register_forward_hook(make_hook(li)))
    try:
        enc = tok(text, return_tensors="pt", add_special_tokens=True).to(device)
        _ = model(**enc)
    finally:
        for h in handles:
            h.remove()
    missing = wanted - set(captured.keys())
    if missing:
        raise RuntimeError(f"Missing activations for layers {sorted(missing)}")
    return {k: v.numpy() for k, v in captured.items()}


@torch.no_grad()
def score_with_patch(
    model,
    tok,
    prompt: str,
    continuation: str,
    patch_layer: int,
    patch_token_indices: List[int],
    patch_vector: torch.Tensor,
    alpha: float,
    mode: str,
    device: str,
) -> float:
    _, layer_stack = model_layers(model)
    patch_vector = patch_vector.to(device=device)
    def hook(module, inp, out):
        h = hidden_from_module_output(out)
        new_h = h.clone()
        if mode == "add":
            new_h[0, patch_token_indices, :] = new_h[0, patch_token_indices, :] + alpha * patch_vector
        elif mode == "subtract":
            new_h[0, patch_token_indices, :] = new_h[0, patch_token_indices, :] - alpha * patch_vector
        elif mode == "replace":
            new_h[0, patch_token_indices, :] = patch_vector
        else:
            raise ValueError(mode)
        return replace_hidden_in_output(out, new_h)
    handle = layer_stack[patch_layer].register_forward_hook(hook)
    try:
        return continuation_logprob(model, tok, prompt, continuation, device)
    finally:
        handle.remove()


def cosine(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def center_vectors(x: np.ndarray, mean: Optional[np.ndarray]) -> np.ndarray:
    if mean is None:
        return x
    return x - mean


def run_metadata(cfg: Dict[str, Any], dataset_path: str | Path) -> Dict[str, Any]:
    import transformers, sklearn, pandas
    return {
        "run_id": cfg["run"].get("id", "default"),
        "model_name": cfg["model"]["name"],
        "model_revision": cfg["model"].get("revision"),
        "use_chat_template": cfg["model"].get("use_chat_template", False),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path) if Path(dataset_path).exists() else None,
        "git_commit": git_commit_or_none(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pandas.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def save_json(path: str | Path, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
