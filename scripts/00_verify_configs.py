#!/usr/bin/env python3
"""Phase 0 config.json verification. STOP on mismatch."""
from __future__ import annotations
import json, sys, urllib.request
from pathlib import Path

OUT = Path("/root/mktg3389/env")
OUT.mkdir(parents=True, exist_ok=True)
LOG = Path("/root/mktg3389/logs/00-environment-verification.log")

EXPECTED = {
    "Qwen/Qwen3-14B": {"num_hidden_layers": 40, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936, "rope_scaling": None},
    "Qwen/Qwen3-0.6B": {"num_hidden_layers": 28, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936},
    "Qwen/Qwen3-1.7B": {"num_hidden_layers": 28, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936},
}

def fetch_config(repo: str) -> dict:
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "mktg3389-phase0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def head_dim(cfg: dict) -> int:
    if "head_dim" in cfg and cfg["head_dim"] is not None:
        return int(cfg["head_dim"])
    return int(cfg["hidden_size"]) // int(cfg["num_attention_heads"])

failures = []
with LOG.open("a") as log:
    def emit(msg: str):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    emit(f"===== config verification =====")
    for repo, exp in EXPECTED.items():
        try:
            cfg = fetch_config(repo)
        except Exception as e:
            failures.append(f"{repo}: fetch failed: {e}")
            emit(f"FAIL fetch {repo}: {e}")
            continue
        snap = {
            "repo": repo,
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "num_key_value_heads": cfg.get("num_key_value_heads"),
            "num_attention_heads": cfg.get("num_attention_heads"),
            "hidden_size": cfg.get("hidden_size"),
            "head_dim": head_dim(cfg),
            "vocab_size": cfg.get("vocab_size"),
            "rope_scaling": cfg.get("rope_scaling"),
            "max_position_embeddings": cfg.get("max_position_embeddings"),
            "_commit_hint": cfg.get("_name_or_path"),
        }
        path = OUT / (repo.replace("/", "__") + ".config.json")
        path.write_text(json.dumps({"snapshot": snap, "raw": cfg}, indent=2))
        emit(f"saved {path}")
        for k, v in exp.items():
            got = snap.get(k) if k != "head_dim" else snap["head_dim"]
            if k == "rope_scaling":
                ok = got in (None, {}) or got is None
                # prompt wants rope_scaling: null specifically for 14B
                if repo.endswith("Qwen3-14B") and got not in (None,):
                    failures.append(f"{repo}: rope_scaling expected null, got {got!r}")
                    emit(f"FAIL {repo} rope_scaling={got!r}")
                elif repo.endswith("Qwen3-14B"):
                    emit(f"OK {repo} rope_scaling={got!r}")
                continue
            if got != v:
                failures.append(f"{repo}: {k} expected {v}, got {got}")
                emit(f"FAIL {repo} {k}: expected {v}, got {got}")
            else:
                emit(f"OK {repo} {k}={got}")

    if failures:
        emit("STOP: config mismatches:")
        for f in failures:
            emit("  - " + f)
        sys.exit(2)
    emit("CONFIG_OK")
