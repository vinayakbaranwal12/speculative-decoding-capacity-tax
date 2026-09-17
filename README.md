# Speculative Decoding Capacity Tax — Evidence

Benchmark evidence for the DigitalOcean Community article *Speculative
Decoding's Hidden Cost: The Capacity Tax No One Measures* (Qwen3-14B on a
single H200, stock vLLM 0.27.1).

## Abstract

Speculative decoding is usually evaluated on throughput alone. This sweep
measures the other side of the ledger: how much KV-cache batch-slot capacity
a draft model costs before it serves a single request, and how that fixed
capacity tax interacts with the throughput crossover that speculative
decoding is normally sold on. One H200, one dense 14B target, four context
regimes (2K/8K/32K/128K), three speculative arms (n-gram, a separate
0.6B/1.7B draft model, and shared-trunk EAGLE3) against a no-speculation
baseline, across five real-world traces (ShareGPT, SWE-bench, CNN/DailyMail
short and long, SEC EDGAR).

## Findings

1. **A separate-weights draft model gives up 43.8% of the KV pool's batch
   slots at 2K before serving a single request**, purely to hold its own
   weights and its own per-request KV cache.
2. **The draft model's own weights are a small fraction of that cost.**
   Decomposing the separate-draft arm's 132.41 lost slots at 2K: roughly
   124.4 slots (94%) from KV geometry, 4.9 (4%) from fixed speculative-decoding
   engine overhead, and only 2.1 (2%) from the draft weights themselves.
3. **EAGLE3's shared-trunk design avoids almost all of this tax** — it adds
   a draft head, not a second full KV cache, so its capacity cost is under a
   seventh of the separate-draft arm's.
4. **Throughput crossovers (measured at 2K) sit far below each arm's
   capacity ceiling.** Separate-draft crosses below baseline output tok/s at
   roughly 17–30 concurrent requests depending on domain; its capacity
   ceiling at that point is still 169.71 slots. The operator hits the
   throughput problem long before the capacity problem.
5. **At 128K, the binding constraint flips.** Baseline holds 4.72 batch
   slots; separate-draft holds 2.65. Neither arm can reach the concurrency
   where a throughput crossover would even be observable — capacity alone
   sets the ceiling on how many requests can be served at all.
6. **p99 inter-token latency does not follow one rule across domains.**
   Baseline has the lowest p99 tail on short-input domains (ShareGPT,
   SWE-bench) and the highest on domains with either high request turnover
   (CNN-short) or long inputs (EDGAR) — two distinct mechanisms producing
   the same symptom.

## Methodology

Full detail in [`METHODS.md`](METHODS.md), including the four pinned Hugging
Face model revisions, the `hf/hub` cache-path correction, hardware/software
pins, arm and trace definitions, and pointers to
[`harness/sweep_runner.py`](harness/sweep_runner.py) and
[`RERUN_NOTES.md`](RERUN_NOTES.md) for the step-by-step debugging history
(EDGAR trace rebuild, SPEED-Bench licensing exclusion, request-phase re-run).

## Known limitations

- The throughput crossover curve was measured **at 2K only**. There is no
  local trace reaching 8K, so any statement combining a 128K capacity
  measurement with a crossover concurrency is combining two different
  context lengths, not reporting a 128K-measured crossover.
- Per-domain acceptance-rate and ITL tables are **2K-only**, for the reason
  documented in `RERUN_NOTES.md` (Step 0b): SPEED-Bench's evaluation license
  permits publishing results but not redistributing its dataset, so a
  context-axis acceptance sweep using SPEED-Bench inputs is out of scope for
  a redistributable public run.
- Single hardware target (one H200), single dense model family (Qwen3).
  Findings on capacity mechanics (KV geometry dominates the tax) should
  generalize; specific slot counts and crossover concurrencies are
  Qwen3-14B/H200-specific.

## Reproduction instructions

```bash
git clone https://github.com/vinayakbaranwal12/speculative-decoding-capacity-tax.git
cd speculative-decoding-capacity-tax

# Rebuild the T2-T5 tables from the raw sweep + manifest
python3 analysis/build_phase4_tables.py

# Re-run a single sweep point on your own H200 Droplet
python3 harness/sweep_runner.py --help
```

No third-party Python packages are required to re-tabulate the published
results (`requirements.txt` — standard library only). Re-running the sweep
itself requires vLLM 0.27.1 and an H200-class GPU; see `METHODS.md` for the
exact pins and the four Hugging Face model revisions used.

Verify file integrity against [`SHA256SUMS.txt`](SHA256SUMS.txt).

## Repository contents

| Path | What |
| --- | --- |
| `results/raw_sweep.jsonl` | 350 measured runs (embedded vLLM bench JSON) — boot phase |
| `results/run_manifest.jsonl` | 21 boot records (failed + successful 128K included) |
| `results_request_phase/` | Request-phase re-run: per-request acceptance and ITL data (`pool_size`, `repetition_factor`, sampling params) — a distinct measurement phase from `results/`, not a duplicate |
| `logs/sweep/` | 1,417 per-run boot/bench/metrics files |
| `logs/03-sweep-master.log` | Master sweep log |
| `logs/smoke/` | Draft-model smoke test |
| `analysis/` | T2–T5 tables (capacity ledger, throughput, acceptance, cost) + H200 rate pin |
| `traces/` | Domain prompt sets (ShareGPT, SWE-bench, CNN short/long, EDGAR) |
| `harness/` | Sweep runner used on the Droplet |
| `env/` | Model revisions, attention-backend pin, FlashInfer patch note |
| `figures/` | Published chart PNGs (capacity decomposition, batch slots by regime, ITL p99 by domain) |

## Pins

- H200 Droplet · vLLM **0.27.1** · `Qwen/Qwen3-14B` · util **0.9** · no prefix cache · **FLASHINFER**
- 128K YaRN: `--hf-overrides` factor **4.0**
- Cost rate: **$4.47/GPU-hr** (DO docs, verified 2026-09-10)

## Citation

```
Vinayak Baranwal (2026). Speculative decoding capacity tax on Qwen3-14B.
https://github.com/vinayakbaranwal12/speculative-decoding-capacity-tax
```

## License

MIT License. See [LICENSE](LICENSE).
