# Methodology

Short-form summary. For the full step-by-step debugging history (EDGAR trace
rebuild, SPEED-Bench licensing exclusion, request-phase re-run), see
[`RERUN_NOTES.md`](RERUN_NOTES.md). For the sweep driver itself, see
[`harness/sweep_runner.py`](harness/sweep_runner.py).

## Hardware and pin

One DigitalOcean H200 GPU Droplet (`gpu-h200x1-141gb`), vLLM **0.27.1**,
target `Qwen/Qwen3-14B`, `--gpu-memory-utilization 0.9`, `FLASHINFER`
attention backend, prefix caching off unless the arm requires it. 128K
regime uses YaRN via `--hf-overrides` with rope factor `4.0`
(see `env/attention_backend_evidence.txt` and `env/flashinfer_patch_note.txt`).

## Model revisions

Captured directly from the Droplet's Hugging Face cache before it was
destroyed, so the exact weights behind every measured number are pinned:

```
=== HF model revisions, before destroying this Droplet ===
cache root: /root/mktg3389/hf/hub
captured: 2026-09-16T07:56:17Z

Qwen/Qwen3-14B -> 40c069824f4251a91eefaf281ebe4c544efd3e18
Qwen/Qwen3-0.6B -> c1899de289a04d12100db370d81485cdf75e47ca
Qwen/Qwen3-1.7B -> 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
RedHatAI/Qwen3-14B-speculator.eagle3 -> 2d489f5a8e490507e4fba701a5515e291f211fdc

=== also compare default cache Qwen3-14B if present ===
default-cache Qwen/Qwen3-14B -> 40c069824f4251a91eefaf281ebe4c544efd3e18
```

Full file: [`env/hf_model_revisions.txt`](env/hf_model_revisions.txt).

**Path correction:** the Hugging Face cache root on the Droplet was
`/root/mktg3389/hf/hub`, not `/root/mktg3389/hf/`. `hf/` on its own is the
`HF_HOME` root; the actual blob/snapshot cache vLLM reads from is the `hub/`
subdirectory under it. Anyone reproducing this sweep who points `HF_HOME` at
`hf/` and expects the pinned revisions above to resolve will miss this
distinction.

## Arms

- `baseline` — no speculative decoding
- `n-gram` — prompt lookup, no separate draft model
- `separate-draft` — `Qwen/Qwen3-0.6B` (and `Qwen/Qwen3-1.7B` for one
  cross-family check) as an independent draft model
- `EAGLE3` — `RedHatAI/Qwen3-14B-speculator.eagle3`, shared-trunk draft head

## Context regimes

2K, 8K, 32K, 128K. The request-phase (acceptance, per-domain ITL) measurement
is 2K-only; see the note in `RERUN_NOTES.md` (Step 0b) on why SPEED-Bench's
license kept a context-axis acceptance sweep out of the redistributable
package.

## Traces

ShareGPT, SWE-bench, CNN/DailyMail (short and long output), and EDGAR
(SEC filing excerpts, rebuilt from primary filing documents — see
`RERUN_NOTES.md` Step 0a for the fix that replaced the earlier
submissions-metadata substitute).

## Reproducing a single point

```bash
python3 harness/sweep_runner.py --help
```

`sweep_runner.py` is the driver used on the Droplet; `analysis/build_phase4_tables.py`
regenerates the T2–T5 tables in `analysis/` from `results/raw_sweep.jsonl` and
`results/run_manifest.jsonl`.
