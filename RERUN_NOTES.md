# MKTG-3389 re-run notes (request phase only)

## Step 0a — EDGAR (DONE)

**Finding:** `scripts/01_build_edgar_traces.py` previously built prompts from
`data.sec.gov` submissions metadata + companyfacts only ("filing excerpt
substitute"). Measured prompt length ~121–227 tokens. That is **not** document
extraction.

**Fix:** builder now fetches the primary filing document from SEC Archives,
strips HTML, truncates, and appends `FILING BODY (truncated):` plus reported
financial facts. Manifest `n_with_filing_body` must equal record count.
Body truncated so prompts fit the **2K** regime with 320-token output headroom.

## Step 0b — SPEED-Bench / speed_bench licence (DONE)

NVIDIA Evaluation Dataset License Agreement:
- **Allowed:** download, use, measure; publish **results**.
- **Forbidden:** distribute, embed, or host the Dataset (in whole or in part)
  in a public evidence repo.

**Decision:** do **not** put SPEED-Bench inputs in the public evidence package.
Step 5 context-axis acceptance via SPEED-Bench is therefore **out of scope**
for a redistributable run. Article reports context dimension from the
**capacity ledger** (already measured; do not redo) and states why the
acceptance table is single-regime (2K).

## Step 0c — load_prompts(limit=None) (DONE)

`load_prompts` now accepts `limit: int | None`; `None` returns the full pool.

## Smoke diagnose (2026-09-10)

- vLLM 0.27.1 flag is `--skip-chat-template`, not `--custom-skip-chat-template` (runbook typo).
- Custom dataset path requires `vllm[bench]` extras (pandas); install before request-phase.

## Step 4 smoke gate (2026-09-10T17:10Z UTC)

CLI fix applied before retry: `--skip-chat-template` (not `--custom-skip-chat-template`); installed pandas for custom dataset.

| Arm | Gate1 custom | completed | accept% | accept_len | pos-0 |
|-----|--------------|-----------|---------|------------|-------|
| baseline | PASS | 50/50 | — | — | — |
| ngram | PASS | 50/50 | 36.6 | 2.09 | 0.491 |
| eagle3 | PASS | 50/50 | 31.4 | 1.94 | **0.511** |
| draft-0.6B | PASS | 50/50 | 47.4 | 2.42 | 0.652 |

Decision: **Proceed (outcome 1).** EAGLE3 position-0 rose from first-sweep 0.215 toward 0.7 on real SWE-bench + Qwen sampling. n-gram collapsed from inflated ~76% to ~37%, consistent with removing greedy+`--ignore-eos` repetition loops. Not all-arms-collapse.

## Step 5–6 request-phase results (2026-09-10)

60/60 rows, 0 failures after EDGAR truncate-to-1720-tokens (char estimate had undercounted dense XBRL).

### Acceptance @ concurrency=1

| Arm | Trace | pos-0 | accept% | accept_len | tok/s |
|-----|-------|------:|--------:|-----------:|------:|
| baseline | cnn_long | — | — | — | 110.6 |
| baseline | cnn_short | — | — | — | 107.4 |
| baseline | edgar_json | — | — | — | 108.8 |
| baseline | sharegpt | — | — | — | 111.4 |
| baseline | swebench | — | — | — | 111.0 |
| draft-0.6B | cnn_long | 0.676 | 48.7 | 2.46 | 140.6 |
| draft-0.6B | cnn_short | 0.610 | 41.3 | 2.24 | 122.5 |
| draft-0.6B | edgar_json | 0.713 | 60.9 | 2.83 | 153.2 |
| draft-0.6B | sharegpt | 0.657 | 47.0 | 2.41 | 142.6 |
| draft-0.6B | swebench | 0.656 | 48.1 | 2.44 | 142.7 |
| eagle3 | cnn_long | 0.558 | 32.2 | 1.97 | 168.0 |
| eagle3 | cnn_short | 0.546 | 31.5 | 1.94 | 156.1 |
| eagle3 | edgar_json | 0.648 | 50.0 | 2.50 | 196.4 |
| eagle3 | sharegpt | 0.538 | 31.5 | 1.94 | 172.8 |
| eagle3 | swebench | 0.493 | 30.1 | 1.90 | 166.1 |
| ngram | cnn_long | 0.365 | 23.8 | 1.71 | 108.2 |
| ngram | cnn_short | 0.413 | 26.7 | 1.80 | 105.9 |
| ngram | edgar_json | 0.491 | 35.8 | 2.07 | 128.8 |
| ngram | sharegpt | 0.431 | 29.4 | 1.88 | 106.6 |
| ngram | swebench | 0.492 | 34.8 | 2.04 | 117.4 |

Evidence: `/Users/vbaranwal/Documents/DRAFTS/mktg3389-request-phase-rerun/mktg3389-request-phase-rerun.tgz`
Droplet still up: `<DROPLET_IP>` (`/root/mktg3389-rerun`). Destroy only when you say so.


## Number corrections (carry-forward guard)

- n-gram on ShareGPT acceptance is **29.4%**, not 27%.
- n-gram peak domain is **EDGAR at 35.8%**, not SWE-bench at 38%.
- These match `results_request_phase/request_phase_summary.jsonl` @ c=1 from the first corrected 2K pass.

## Extended ladder (ran; see COMPLETE below)

- Same sampling as the 3-point re-run: temp 0.6 / top-p 0.95 / top-k 20 / seed 0; `--dataset-name custom`; `--skip-chat-template`; no `--ignore-eos`. Directly comparable to the 3-point re-run; **not** comparable to the first synthetic sweep.
- Concurrency: 1, 5, 10, 25, 35, 50, 75, 100; draft-0.6B also 170. Higher arm ceilings deferred.
- `num_prompts = max(8 * concurrency, 200)` for **all** domains (equalizes edgar vs others).
- 60s duration floor: re-run with bumped n if bench duration < 60s.
- Capacity ledger / boot-phase: not touched.
- Trace files used for this ladder:
- `edgar_json/records.jsonl` sha256[:16]=f94ba010e87bf1de (307477 bytes)
- `sharegpt_prompts.jsonl` sha256[:16]=a4da232f110a551f (106732 bytes)
- `swebench_prompts.jsonl` sha256[:16]=40137dee29822b33 (99615 bytes)
- `cnn_short_prompts.jsonl` sha256[:16]=68b7802d2c7ba75a (210671 bytes)
- `cnn_long_prompts.jsonl` sha256[:16]=5774e1fbc076051f (211567 bytes)


## Extended ladder COMPLETE (2026-09-11T03:12:12Z)

- Finished `2026-09-11T03:12:12Z`, exit=0. Capacity ledger untouched. Droplet left up pending destroy decision.
- Rows: total raw_sweep=225; extend_ladder_8pt=165 (baseline/ngram/eagle3 x40 + draft-0.6B x45 including c=170).
- Sampling identical to 3-point re-run (0.6/0.95/20, custom dataset, skip-chat-template, no ignore-eos) — directly comparable to that pass; not comparable to the synthetic first sweep.
- Audit (every extend row): short<60s=0; completed!=n=0; failed>0=0; floor_fail=0; errors=0; missing grid points=0.
- Duration floor enforced (105/165 rows needed >=1 retry). Final duration min=60.483s, med=94.506s, max=588.992s.
- Concurrency achieved: 1,5,10,25,35,50,75,100 all arms; draft also 170. Higher arm ceilings (284-302) not run (time).
