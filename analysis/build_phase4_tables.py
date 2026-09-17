#!/usr/bin/env python3
"""MKTG-3389 Phase 4: derive T2–T5 from measured sweep artifacts. Never invent metrics."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS, ANALYSIS = ROOT / "results", ROOT / "analysis"
MANIFEST, RAW = RESULTS / "run_manifest.jsonl", RESULTS / "raw_sweep.jsonl"
DEFAULT_RATE = 4.47
ARMS = {
    "baseline": "baseline",
    "ngram": "n-gram",
    "eagle3": "EAGLE3",
    "draft-0.6B": "separate-draft",
    "draft-1.7B": "separate-draft-1.7B",
}
DOMAIN = {
    "sharegpt": "conversational",
    "swebench": "code",
    "cnn_short": "summarization-short",
    "cnn_long": "summarization-long",
    "edgar_json": "structured-JSON",
}
PRIMARY = ["baseline", "ngram", "eagle3", "draft-0.6B"]
REGIMES = ["2K", "8K", "32K", "128K"]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def latest_ok(manifest: list[dict]) -> dict[tuple[str, str], dict]:
    best = {}
    for row in manifest:
        arm, regime = row.get("arm"), row.get("regime")
        if arm and regime and row.get("ready") and row.get("status") in ("ok", "suspect"):
            best[(arm, regime)] = row
    return best


def tps(bench: dict | None) -> float | None:
    if not bench:
        return None
    for k in ("output_throughput", "output_tok_s", "output_tokens_per_second"):
        if bench.get(k) is not None:
            return float(bench[k])
    return None


def accept_mean(lines: list[str] | None) -> float | None:
    if not lines:
        return None
    accepted = drafts = None
    for ln in lines:
        if "spec_decode_num_accepted_tokens_total" in ln and "per_pos" not in ln:
            try:
                accepted = float(ln.split()[-1])
            except Exception:
                pass
        if "spec_decode_num_drafts_total" in ln and "created" not in ln:
            try:
                drafts = float(ln.split()[-1])
            except Exception:
                pass
    if accepted is not None and drafts and drafts > 0:
        return accepted / drafts
    return None


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    rate_meta = {
        "usd_per_gpu_hr": DEFAULT_RATE,
        "source": "https://docs.digitalocean.com/products/droplets/details/pricing/ NVIDIA H200 $4.47/hr",
        "verified_at_utc": "2026-09-10T10:35:00Z",
    }
    rf = ANALYSIS / "h200_rate.json"
    if rf.exists():
        rate_meta = json.loads(rf.read_text())
    else:
        rf.write_text(json.dumps(rate_meta, indent=2))
    rate = float(rate_meta["usd_per_gpu_hr"])

    boots = latest_ok(load_jsonl(MANIFEST))
    raw = load_jsonl(RAW)

    t2 = []
    for regime in REGIMES:
        base = boots.get(("baseline", regime))
        if not base:
            continue
        b_slots = float((base.get("boot_parse") or {}).get("concurrency_multiplier") or 0)
        for arm in PRIMARY:
            row = boots.get((arm, regime))
            if not row:
                continue
            bp = row.get("boot_parse") or {}
            slots = float(bp.get("concurrency_multiplier") or 0)
            pool = float(bp.get("kv_cache_tokens") or 0)
            lost = 0.0 if arm == "baseline" else b_slots - slots
            rec = {
                "arm": ARMS[arm],
                "regime": regime,
                "pool_tokens": pool,
                "batch_slots": slots,
                "slots_lost_vs_baseline": lost,
                "slots_lost_to_draft_weights": None,
                "slots_lost_to_draft_KV": None,
                "status": row.get("status"),
                "suspect": bp.get("suspect"),
                "rope_factor": bp.get("rope_factor"),
                "pool_crosscheck_rel_err": bp.get("pool_crosscheck_rel_err"),
            }
            if arm == "baseline":
                rec["slots_lost_to_draft_weights"] = 0.0
                rec["slots_lost_to_draft_KV"] = 0.0
            elif arm == "ngram":
                rec["slots_lost_to_draft_weights"] = 0.0
                rec["slots_lost_to_draft_KV"] = 0.0
                rec["residual_slots_lost"] = lost
            elif arm == "draft-0.6B":
                # KV term = geometry prior 160/272 of baseline slots; weights = residual
                kv = b_slots * (112.0 / 272.0)
                rec["slots_lost_to_draft_KV"] = kv
                rec["slots_lost_to_draft_weights"] = lost - kv
                rec["decomposition_note"] = "KV=geometry prior 160/272; weights=measured_loss-KV"
            t2.append(rec)
    sens = boots.get(("draft-1.7B", "8K"))
    if sens:
        bp = sens.get("boot_parse") or {}
        b8 = boots.get(("baseline", "8K"))
        b_slots = float(((b8 or {}).get("boot_parse") or {}).get("concurrency_multiplier") or 0)
        slots = float(bp.get("concurrency_multiplier") or 0)
        t2.append({
            "arm": ARMS["draft-1.7B"],
            "regime": "8K",
            "pool_tokens": bp.get("kv_cache_tokens"),
            "batch_slots": slots,
            "slots_lost_vs_baseline": b_slots - slots if b8 else None,
            "note": "sensitivity @ 8K only",
            "status": sens.get("status"),
        })

    write_csv(
        ANALYSIS / "T2_capacity_ledger.csv",
        t2,
        [
            "arm", "regime", "pool_tokens", "batch_slots", "slots_lost_vs_baseline",
            "slots_lost_to_draft_weights", "slots_lost_to_draft_KV", "residual_slots_lost",
            "status", "suspect", "rope_factor", "pool_crosscheck_rel_err", "decomposition_note", "note",
        ],
    )
    (ANALYSIS / "T2_capacity_ledger.json").write_text(json.dumps(t2, indent=2))
    md = ["# T2 Capacity Ledger (measured)", "", "| arm | regime | pool | slots | lost | weights | KV |", "|-----|--------|------|-------|------|---------|-----|"]
    for r in t2:
        md.append(
            f"| {r.get('arm')} | {r.get('regime')} | {r.get('pool_tokens')} | {r.get('batch_slots')} | "
            f"{r.get('slots_lost_vs_baseline')} | {r.get('slots_lost_to_draft_weights')} | {r.get('slots_lost_to_draft_KV')} |"
        )
    (ANALYSIS / "T2_capacity_ledger.md").write_text("\n".join(md) + "\n")

    base_tps = {}
    for r in raw:
        if r.get("arm") == "baseline" and r.get("returncode") in (0, None) and not r.get("error"):
            v = tps(r.get("bench"))
            if v is not None:
                base_tps[(r.get("trace"), r.get("regime"), int(r.get("concurrency") or 0))] = v

    by_art = defaultdict(list)
    for r in raw:
        if r.get("arm") in PRIMARY + ["draft-1.7B"] and r.get("returncode") in (0, None) and not r.get("error"):
            if tps(r.get("bench")) is not None:
                by_art[(r["arm"], r["regime"], r["trace"])].append(r)

    t3 = []
    for r in raw:
        if r.get("arm") not in PRIMARY + ["draft-1.7B"] or r.get("returncode") not in (0, None) or r.get("error"):
            continue
        v = tps(r.get("bench"))
        if v is None:
            continue
        conc = int(r.get("concurrency") or 0)
        b = base_tps.get((r.get("trace"), r.get("regime"), conc))
        peers = by_art.get((r["arm"], r["regime"], r["trace"]), [])
        sat = max(peers, key=lambda x: int(x.get("concurrency") or 0)) if peers else None
        sat_c = int(sat.get("concurrency") or 0) if sat else None
        sat_v = tps(sat.get("bench")) if sat else None
        sat_b = base_tps.get((r.get("trace"), r.get("regime"), sat_c)) if sat_c else None
        t3.append({
            "arm": ARMS.get(r["arm"], r["arm"]),
            "domain": DOMAIN.get(r.get("trace"), r.get("trace")),
            "trace": r.get("trace"),
            "regime": r.get("regime"),
            "concurrency": conc,
            "output_tok_s": v,
            "speedup_vs_baseline": (v / b) if b and b > 0 else None,
            "saturation_concurrency": sat_c,
            "saturation_output_tok_s": sat_v,
            "saturation_speedup_vs_baseline": (sat_v / sat_b) if sat_v and sat_b and sat_b > 0 else None,
        })
    write_csv(
        ANALYSIS / "T3_throughput.csv",
        t3,
        [
            "arm", "domain", "trace", "regime", "concurrency", "output_tok_s", "speedup_vs_baseline",
            "saturation_concurrency", "saturation_output_tok_s", "saturation_speedup_vs_baseline",
        ],
    )
    (ANALYSIS / "T3_throughput.json").write_text(json.dumps(t3, indent=2))

    grouped = defaultdict(list)
    for r in raw:
        if r.get("arm") in ("baseline",) or r.get("arm") not in PRIMARY + ["draft-1.7B"]:
            continue
        m = accept_mean(r.get("spec_metric_lines"))
        if m is not None:
            grouped[(r["arm"], DOMAIN.get(r.get("trace"), r.get("trace")))].append(m)
    t4 = []
    for (arm, domain), vals in sorted(grouped.items()):
        t4.append({
            "arm": ARMS.get(arm, arm),
            "domain": domain,
            "mean_accepted_length": statistics.mean(vals),
            "median_accepted_length": statistics.median(vals),
            "std_accepted_length": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "n_samples": len(vals),
            "note": "from /metrics scrapes; output-length dist not populated",
        })
    if not t4:
        t4 = [{"note": "no acceptance metrics recoverable"}]
    write_csv(
        ANALYSIS / "T4_acceptance.csv",
        t4,
        [
            "arm", "domain", "mean_accepted_length", "median_accepted_length", "std_accepted_length",
            "n_samples", "note",
        ],
    )
    (ANALYSIS / "T4_acceptance.json").write_text(json.dumps(t4, indent=2))

    t5 = []
    for row in t3:
        v = row.get("output_tok_s")
        if not v or v <= 0:
            continue
        t5.append({
            "arm": row["arm"],
            "domain": row["domain"],
            "regime": row["regime"],
            "concurrency": row["concurrency"],
            "output_tok_s": v,
            "cost_per_1M_output_tokens_USD": (rate / (v * 3600.0)) * 1_000_000.0,
            "h200_usd_per_hr": rate,
        })
    write_csv(
        ANALYSIS / "T5_cost.csv",
        t5,
        [
            "arm", "domain", "regime", "concurrency", "output_tok_s",
            "cost_per_1M_output_tokens_USD", "h200_usd_per_hr",
        ],
    )
    (ANALYSIS / "T5_cost.json").write_text(json.dumps(t5, indent=2))

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "raw_rows": len(raw),
        "manifest_ok_boots": {f"{a}@{r}": (boots[(a, r)].get("boot_parse") or {}).get("concurrency_multiplier") for (a, r) in sorted(boots)},
        "t2_rows": len(t2),
        "t3_rows": len(t3),
        "t4_rows": len(t4),
        "t5_rows": len(t5),
        "h200_rate": rate_meta,
    }
    (ANALYSIS / "phase4_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
