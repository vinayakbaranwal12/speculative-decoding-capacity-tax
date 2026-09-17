#!/usr/bin/env python3
"""
MKTG-3389 speculative-decoding capacity sweep runner.

Adapted from context-length-inference-cost@9fd62363 harness/sweep_orchestrator.py.
Runs locally against a remote or local vLLM endpoint after each boot, or boots
vLLM on the same machine when --local-serve is set.

Absolute rules (do not relax):
- Never invent measured values.
- Cross-check KV pool tokens vs peak concurrency * context; >10% => suspect.
- Confirm YaRN factor 4.0 in every 128K boot log.
- Append-only results; checkpoint after every run.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("MKTG3389_ROOT", "/root/mktg3389"))
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
TRACES = ROOT / "traces"
MANIFEST = RESULTS / "run_manifest.jsonl"
RAW = RESULTS / "raw_sweep.jsonl"
SWEEP_LOG = LOGS / "sweep"

FIXED = {
    "tp": 1,
    "gpu_memory_utilization": 0.9,
    "prefix_caching": False,
}

REGIMES = {
    "2K": {"max_model_len": 2048, "rope": None, "context": 2048},
    "8K": {"max_model_len": 8192, "rope": None, "context": 8192},
    "32K": {"max_model_len": 32768, "rope": None, "context": 32768},
    "128K": {
        "max_model_len": 131072,
        "rope": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
        "context": 131072,
    },
}

ARMS = {
    "baseline": None,
    "ngram": {
        "method": "ngram",
        "num_speculative_tokens": 3,
        "prompt_lookup_max": 4,
        "prompt_lookup_min": 2,
    },
    "eagle3": {
        "model": "RedHatAI/Qwen3-14B-speculator.eagle3",
        "num_speculative_tokens": 3,
        "method": "eagle3",
    },
    "draft-0.6B": {
        "method": "draft_model",
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 3,
    },
    "draft-1.7B": {
        "method": "draft_model",
        "model": "Qwen/Qwen3-1.7B",
        "num_speculative_tokens": 3,
    },
}

LADDER = [1, 5, 25, 50, 100, 200]
TARGET = "Qwen/Qwen3-14B"
ATTENTION_BACKEND = os.environ.get("MKTG3389_ATTENTION_BACKEND", "")  # set after first boot


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()


def run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def parse_boot_log(text: str) -> dict:
    out = {
        "kv_cache_tokens": None,
        "concurrency_multiplier": None,
        "attention_backend": None,
        "rope_factor": None,
        "startup_complete": "Application startup complete" in text,
        "suspect": False,
        "suspect_reasons": [],
    }
    m = re.search(r"GPU KV cache size:\s*([0-9,\.]+)\s*tokens", text)
    if m:
        out["kv_cache_tokens"] = float(m.group(1).replace(",", ""))
    m = re.search(r"Maximum concurrency for\s*([0-9,\.]+)\s*tokens per request:\s*([0-9\.]+)x", text)
    if m:
        out["concurrency_multiplier"] = float(m.group(2))
        out["concurrency_context_tokens"] = float(m.group(1).replace(",", ""))
    m = re.search(r"Using\s+(\S+)\s+attention backend", text, re.I)
    if m:
        out["attention_backend"] = m.group(1)
    # YaRN factor
    m = re.search(r"factor[\"']?\s*[:=]\s*([0-9\.]+)", text)
    if m:
        out["rope_factor"] = float(m.group(1))
    if "yarn" in text.lower() and re.search(r"factor[\"']?\s*[:=]\s*4(\.0)?\b", text):
        out["rope_factor"] = 4.0
    return out


def cross_check(boot: dict, context: int) -> dict:
    """Absolute Rule 3."""
    tokens = boot.get("kv_cache_tokens")
    mult = boot.get("concurrency_multiplier")
    if tokens is None or mult is None:
        boot["suspect"] = True
        boot["suspect_reasons"].append("missing pool tokens or concurrency multiplier")
        return boot
    implied = mult * context
    # Prefer concurrency multiplier as source of truth per dossier; compare printed tokens to implied.
    if implied <= 0:
        boot["suspect"] = True
        boot["suspect_reasons"].append("non-positive implied pool")
        return boot
    rel = abs(tokens - implied) / implied
    boot["pool_crosscheck_rel_err"] = rel
    boot["pool_implied_from_concurrency"] = implied
    if rel > 0.10:
        boot["suspect"] = True
        boot["suspect_reasons"].append(
            f"pool tokens {tokens} vs concurrency*ctx {implied} rel_err={rel:.3f} > 0.10"
        )
    return boot


def serve_cmd(arm: str, regime: str) -> list[str]:
    r = REGIMES[regime]
    cfg = ARMS[arm]
    cmd = [
        "vllm", "serve", TARGET,
        "--tensor-parallel-size", str(FIXED["tp"]),
        "--gpu-memory-utilization", str(FIXED["gpu_memory_utilization"]),
        "--max-model-len", str(r["max_model_len"]),
        "--no-enable-prefix-caching",
    ]
    if ATTENTION_BACKEND:
        cmd += ["--attention-backend", ATTENTION_BACKEND]
    if r["rope"] is not None:
        # vLLM 0.27.1 has no --rope-scaling; pass YaRN via HF config overrides.
        cmd += ["--hf-overrides", json.dumps({"rope_scaling": r["rope"]})]
    if cfg is not None:
        cmd += ["--speculative-config", json.dumps(cfg)]
    return cmd


def boot_server(arm: str, regime: str) -> dict:
    SWEEP_LOG.mkdir(parents=True, exist_ok=True)
    boot_path = SWEEP_LOG / f"boot-{arm}-{regime}.log"
    if boot_path.exists():
        boot_path.unlink()
    # kill prior
    subprocess.run(["bash", "-lc", "pkill -f '[v]llm serve' || true"], check=False)
    time.sleep(2)
    cmd = serve_cmd(arm, regime)
    with boot_path.open("w") as boot_f:
        proc = subprocess.Popen(cmd, stdout=boot_f, stderr=subprocess.STDOUT, text=True)
    meta = {
        "arm": arm,
        "regime": regime,
        "cmd": cmd,
        "pid": proc.pid,
        "boot_log": str(boot_path),
        "started_at": utc(),
    }
    # wait ready
    ready = False
    for i in range(240):
        if proc.poll() is not None:
            meta["status"] = "boot_failed"
            meta["returncode"] = proc.returncode
            break
        try:
            r = run(["curl", "-sf", "http://127.0.0.1:8000/v1/models"], timeout=5)
            if r.returncode == 0:
                ready = True
                break
        except Exception:
            pass
        time.sleep(10)
    text = boot_path.read_text(errors="replace")
    parsed = parse_boot_log(text)
    parsed = cross_check(parsed, REGIMES[regime]["context"])
    if regime == "128K":
        if parsed.get("rope_factor") != 4.0:
            parsed["suspect"] = True
            parsed["suspect_reasons"].append(
                f"128K rope factor not confirmed as 4.0 (got {parsed.get('rope_factor')})"
            )
            meta["status"] = "rope_factor_unconfirmed_STOP"
    meta["boot_parse"] = parsed
    meta["ready"] = ready
    if ready and meta.get("status") != "rope_factor_unconfirmed_STOP":
        meta["status"] = "suspect" if parsed["suspect"] else "ok"
    elif "status" not in meta:
        meta["status"] = "boot_timeout"
    append_jsonl(MANIFEST, meta)
    return meta


def kill_server() -> None:
    subprocess.run(["bash", "-lc", "pkill -f '[v]llm serve' || true"], check=False)
    time.sleep(3)


def load_prompts(trace: str, limit: int | None = 64) -> list[str]:
    """Load prompts for a named trace. Prefer local files under traces/.

    limit=None returns the full pool (required for cycling real traces).
    """
    mapping = {
        "sharegpt": TRACES / "sharegpt_prompts.jsonl",
        "swebench": TRACES / "swebench_prompts.jsonl",
        "cnn_short": TRACES / "cnn_short_prompts.jsonl",
        "cnn_long": TRACES / "cnn_long_prompts.jsonl",
        "edgar_json": TRACES / "edgar_json" / "records.jsonl",
    }
    path = mapping[trace]
    if not path.exists():
        raise FileNotFoundError(f"missing trace file {path}")
    prompts = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "prompt" in obj:
                prompts.append(obj["prompt"])
            elif "text" in obj:
                prompts.append(obj["text"])
            elif "messages" in obj:
                for m in reversed(obj["messages"]):
                    if m.get("role") == "user":
                        prompts.append(m["content"])
                        break
            if limit is not None and len(prompts) >= limit:
                break
    if not prompts:
        raise RuntimeError(f"no prompts loaded from {path}")
    return prompts


def run_bench(arm: str, regime: str, trace: str, concurrency: int, num_prompts: int, boot_meta: dict,
              *, min_duration_s: float = 0.0, pass_tag: str | None = None) -> dict:
    """Use vllm bench serve against localhost with real custom traces + Qwen3 sampling.

    If min_duration_s > 0 and bench duration is below that floor, double num_prompts and
    re-run (max 4 attempts). Only the final attempt is appended to raw_sweep.jsonl.
    """
    out_json = SWEEP_LOG / f"{arm}-{regime}-{trace}-c{concurrency}.bench.json"
    out_log = SWEEP_LOG / f"{arm}-{regime}-{trace}-c{concurrency}.log"
    output_len = {"cnn_short": 64, "edgar_json": 320}.get(trace, 256)

    pool = load_prompts(trace, limit=None)
    if not pool:
        raise RuntimeError(f"empty prompt pool for {trace}")

    n = int(num_prompts)
    attempts: list[dict] = []
    result: dict = {}
    metrics_text = ""
    max_attempts = 4 if min_duration_s > 0 else 1
    bench_parse_error = None

    for attempt in range(1, max_attempts + 1):
        ds = SWEEP_LOG / f"dataset-{arm}-{regime}-{trace}-c{concurrency}.jsonl"
        with ds.open("w") as f:
            for prompt in itertools.islice(itertools.cycle(pool), n):
                f.write(json.dumps({"prompt": prompt}) + "\n")
        repetition_factor = n / len(pool)

        cmd = [
            "vllm", "bench", "serve",
            "--backend", "openai",
            "--base-url", "http://127.0.0.1:8000",
            "--model", TARGET,
            "--endpoint", "/v1/completions",
            "--dataset-name", "custom",
            "--dataset-path", str(ds),
            "--custom-output-len", str(output_len),
            "--skip-chat-template",
            "--num-prompts", str(n),
            "--max-concurrency", str(concurrency),
            "--temperature", "0.6",
            "--top-p", "0.95",
            "--top-k", "20",
            "--seed", "0",
            "--save-result",
            "--result-filename", str(out_json),
            "--save-detailed",
        ]
        started = utc()
        mode = "w" if attempt == 1 else "a"
        with out_log.open(mode) as lf:
            lf.write(f"\n=== ATTEMPT {attempt} n={n} @ {started} ===\n")
            lf.write("CMD " + " ".join(cmd) + "\n")
            lf.flush()
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
        finished = utc()

        bench = None
        duration = None
        bench_parse_error = None
        if out_json.exists():
            try:
                bench = json.loads(out_json.read_text())
                duration = bench.get("duration")
            except Exception as e:
                bench_parse_error = str(e)

        attempts.append({
            "attempt": attempt,
            "num_prompts": n,
            "duration": duration,
            "returncode": proc.returncode,
            "completed": (bench or {}).get("completed"),
            "failed": (bench or {}).get("failed"),
        })
        print(
            f"  ATTEMPT {attempt} c={concurrency} n={n} duration={duration} "
            f"completed={(bench or {}).get('completed')} failed={(bench or {}).get('failed')}",
            flush=True,
        )

        result = {
            "arm": arm,
            "regime": regime,
            "trace": trace,
            "concurrency": concurrency,
            "num_prompts": n,
            "pool_size": len(pool),
            "repetition_factor": repetition_factor,
            "output_len": output_len,
            "dataset_name": "custom",
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "started_at": started,
            "finished_at": finished,
            "duration": duration,
            "duration_floor_s": min_duration_s,
            "duration_floor_met": (duration is not None and duration >= min_duration_s) if min_duration_s > 0 else True,
            "attempts": attempts,
            "returncode": proc.returncode,
            "boot_status": boot_meta.get("status"),
            "boot_suspect": boot_meta.get("boot_parse", {}).get("suspect"),
            "boot_parse": boot_meta.get("boot_parse"),
            "bench_json": str(out_json) if out_json.exists() else None,
            "log": str(out_log),
            "cmd": cmd,
            "pass": pass_tag,
        }
        if bench is not None:
            result["bench"] = bench
        if bench_parse_error:
            result["bench_parse_error"] = bench_parse_error

        if min_duration_s <= 0:
            break
        if duration is not None and duration >= min_duration_s:
            break
        if attempt < max_attempts:
            n = max(n * 2, n + 200)
            print(f"  DURATION_FLOOR_RETRY need>={min_duration_s}s got={duration}; bump n->{n}", flush=True)

    metrics_text = ""
    try:
        metrics_text = run(["curl", "-sf", "http://127.0.0.1:8000/metrics"], timeout=30).stdout
        (SWEEP_LOG / f"{arm}-{regime}-{trace}-c{concurrency}.metrics.txt").write_text(metrics_text)
    except Exception as e:
        metrics_text = f"METRICS_FAIL {e}"
    accept_lines = [
        ln for ln in metrics_text.splitlines()
        if re.search(r"speculat|accept|draft", ln, re.I) and not ln.startswith("#")
    ]
    result["spec_metric_lines"] = accept_lines
    append_jsonl(RAW, result)
    return result



def schedule() -> list[tuple[str, str]]:
    boots = []
    for regime in ["2K", "8K", "32K", "128K"]:
        for arm in ["baseline", "ngram", "eagle3", "draft-0.6B"]:
            boots.append((arm, regime))
    boots.append(("draft-1.7B", "8K"))  # sensitivity
    return boots


def schedule_request_phase_2k() -> list[tuple[str, str]]:
    """Request-phase re-run: 2K only. Capacity ledger already locked — do not redo other regimes."""
    return [(arm, "2K") for arm in ["baseline", "ngram", "eagle3", "draft-0.6B"]]


def completed_boots_from_manifest() -> set[tuple[str, str]]:
    """Boots already recorded as ready/ok in run_manifest.jsonl (for resume)."""
    done: set[tuple[str, str]] = set()
    if not MANIFEST.exists():
        return done
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        arm, regime = obj.get("arm"), obj.get("regime")
        if not arm or not regime:
            continue
        if obj.get("ready") and obj.get("status") in ("ok", "suspect"):
            done.add((arm, regime))
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-boot", nargs=2, metavar=("ARM", "REGIME"))
    ap.add_argument("--skip-traces", action="store_true", help="capacity-only: one short random run per concurrency")
    ap.add_argument("--traces", default="sharegpt,swebench,cnn_short,cnn_long,edgar_json")
    ap.add_argument("--resume", action="store_true", help="skip boots already ok/suspect in run_manifest.jsonl")
    ap.add_argument(
        "--request-phase-2k",
        action="store_true",
        help="re-run request phase only at 2K (custom traces + Qwen3 sampling); three concurrency points",
    )
    ap.add_argument(
        "--concurrency-points",
        default="",
        help="comma-separated concurrency overrides (e.g. 1,25,50). Default: full ladder capped at ceiling, or 1,25,50 for --request-phase-2k",
    )
    ap.add_argument(
        "--extend-ladder",
        action="store_true",
        help="extend 2K request-phase: conc 1,5,10,25,35,50,75,100 (+draft 170); "
             "num_prompts=max(8*c,200); 60s duration floor; append to raw_sweep",
    )
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    SWEEP_LOG.mkdir(parents=True, exist_ok=True)

    global ATTENTION_BACKEND
    ab_file = ROOT / "env" / "attention_backend.txt"
    if ab_file.exists() and not ATTENTION_BACKEND:
        ATTENTION_BACKEND = ab_file.read_text().strip()
    if args.only_boot:
        boots = [tuple(args.only_boot)]
    elif args.extend_ladder or args.request_phase_2k:
        boots = schedule_request_phase_2k()
    else:
        boots = schedule()
    if args.resume and not args.only_boot:
        done = completed_boots_from_manifest()
        if done:
            print(f"RESUME_SKIP {sorted(done)}", flush=True)
            boots = [b for b in boots if b not in done]
            print(f"RESUME_REMAINING {boots}", flush=True)
    traces = [] if args.skip_traces else [t.strip() for t in args.traces.split(",") if t.strip()]

    for arm, regime in boots:
        print(f"=== BOOT {arm} {regime} @ {utc()} ===", flush=True)
        meta = boot_server(arm, regime)
        print(json.dumps({"status": meta.get("status"), "boot_parse": meta.get("boot_parse")}, indent=2), flush=True)
        if meta.get("status") == "rope_factor_unconfirmed_STOP":
            print("STOP: 128K rope factor not confirmed", flush=True)
            kill_server()
            return 3
        if not meta.get("ready"):
            print("BOOT_FAILED — skipping runs for this config", flush=True)
            kill_server()
            continue
        # Freeze attention backend from first successful boot
        if not ATTENTION_BACKEND:
            ab = (meta.get("boot_parse") or {}).get("attention_backend")
            if ab:
                ATTENTION_BACKEND = ab.replace("AttentionBackendEnum.", "")
                if "FLASHINFER" in ATTENTION_BACKEND.upper():
                    ATTENTION_BACKEND = "FLASHINFER"
                elif "FLASH_ATTN" in ATTENTION_BACKEND.upper() or "FLASHATTENTION" in ATTENTION_BACKEND.upper():
                    ATTENTION_BACKEND = "FLASH_ATTN"
                (ROOT / "env" / "attention_backend.txt").write_text(ATTENTION_BACKEND + "\n")
                print("FROZEN_ATTENTION_BACKEND", ATTENTION_BACKEND, flush=True)

        mult = (meta.get("boot_parse") or {}).get("concurrency_multiplier") or 1.0
        ceiling = max(1, int(mult))
        if args.concurrency_points.strip():
            concs = [int(x) for x in args.concurrency_points.split(",") if x.strip()]
            concs = [c for c in concs if c <= ceiling] or [min(ceiling, 1)]
        elif args.extend_ladder:
            # Verified ladder; draft ceiling 170 included; higher arm ceilings deferred (time)
            preferred = [1, 5, 10, 25, 35, 50, 75, 100]
            if arm == "draft-0.6B":
                preferred = preferred + [170]
            concs = [c for c in preferred if c <= ceiling]
            if not concs:
                concs = [1]
        elif args.request_phase_2k:
            # Three points per runbook — not the full ladder
            preferred = [1, 25, 50]
            concs = [c for c in preferred if c <= ceiling]
            if not concs:
                concs = [1]
            elif concs[-1] != min(ceiling, 50) and ceiling not in concs and ceiling <= 50:
                concs.append(ceiling)
                concs = sorted(set(concs))
        else:
            concs = [c for c in LADDER if c <= ceiling]
            if ceiling not in concs:
                concs.append(ceiling)
                concs = sorted(set(concs))
        print("CONCURRENCY_POINTS", concs, "ceiling", ceiling, flush=True)

        if meta.get("boot_parse", {}).get("suspect"):
            print("BOOT_SUSPECT — runs logged but must not fold into summary without human review", flush=True)

        run_traces = traces or ["_capacity_probe"]
        for trace in run_traces:
            for conc in concs:
                if args.extend_ladder:
                    n = max(8 * conc, 200)  # uniform across ALL domains
                    min_dur = 60.0
                    pass_tag = "extend_ladder_8pt"
                else:
                    n = max(50, conc * 8)
                    if regime in ("32K", "128K"):
                        n = max(conc * 2, min(n, 64))
                    if args.request_phase_2k:
                        n = max(32, min(n, 128))  # keep smoke/full pass bounded
                    min_dur = 0.0
                    pass_tag = "request_phase_2k" if args.request_phase_2k else None
                print(f"RUN {arm} {regime} {trace} c={conc} n={n}", flush=True)
                try:
                    if trace == "_capacity_probe":
                        p = TRACES / "sharegpt_prompts.jsonl"
                        if not p.exists():
                            TRACES.mkdir(parents=True, exist_ok=True)
                            p.write_text(json.dumps({"prompt": "Write a short paragraph about GPUs."}) + "\n")
                        trace_name = "sharegpt"
                    else:
                        trace_name = trace
                    run_bench(
                        arm, regime, trace_name, conc, n, meta,
                        min_duration_s=min_dur, pass_tag=pass_tag,
                    )
                except Exception as e:
                    append_jsonl(RAW, {
                        "arm": arm, "regime": regime, "trace": trace, "concurrency": conc,
                        "error": str(e), "at": utc(), "boot_status": meta.get("status"),
                        "pass": pass_tag,
                    })
                    print("RUN_FAIL", e, flush=True)
        kill_server()
    print("SWEEP_FINISHED", utc(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
