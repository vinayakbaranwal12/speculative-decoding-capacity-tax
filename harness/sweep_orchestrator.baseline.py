#!/usr/bin/env python3
"""
Per-Droplet sweep driver for MKTG-3270 (v4).

v4 changes (Patch A/B/C, applied together):
  - Patch A: probe_and_size no longer clamps concurrency to 64. Concurrency comes
    from the boot-log ceiling, unclamped. num_prompts is sized off measured probe
    throughput with a minimum-turnover floor (8x concurrency), not a fixed
    15-request probe with a 50/2000 floor/ceiling.
  - Patch B: validate_trial() is a post-hoc acceptance gate on every completed
    trial (duration floor, turnover floor, concurrency-fidelity check against
    EXPECTED_CONCURRENCY, scheduler-pin check against --max-num-seqs).
  - Patch C: true_max_concurrency() computes true peak concurrency by
    millisecond-precision interval sweep over raw per-request timing data,
    instead of trusting vLLM bench serve's own max_concurrent_requests field
    (confirmed during the v3->v4 diagnostic to be a whole-second,
    both-endpoints-inclusive bucket count that over-reports during
    synchronized wave transitions).

v3 changes (retained):
  - C loads BF16 weights + --kv-cache-dtype fp8 + --attention-backend FLASHINFER
    (isolates KV dtype; prior C runs used FP8 weight checkpoint and are invalid)
  - 256K point timeout = 3.5 hours; whole-run backstop = 8 hours from this pass start
  - append mode: top up missing trials without redoing completed ones

Retains: detached-launch-plus-poll, separate warm-up, --save-detailed, FlashInfer gate,
scheduler-pin escalation, SSH retry, no teardown.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Local paths are env-driven so this file can ship without a developer laptop path.
SSH_KEY = os.environ.get("SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519"))
_REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = Path(os.environ.get("SWEEP_LOG_ROOT", _REPO_ROOT / "data"))
REMOTE_LOG_DIR = "/root/mktg3270_logs"
REMOTE_USER = "root"

MODEL_BF16 = "mistralai/Ministral-3-14B-Instruct-2512-BF16"
SERVED_NAME = "mistral-3-14B"
PORT = 8000
OUTPUT_LEN = 256
WARMUP_REQUESTS = 10
TRIALS_PER_POINT = 3
POINT_TIMEOUT_S = 45 * 60
POINT_TIMEOUT_256K_S = int(3.5 * 60 * 60)  # 3.5 hours
BACKSTOP_S = 8 * 60 * 60
FLASHINFER_DIAG_BUDGET_S = 20 * 60
GPU_MEM_UTIL = 0.90
MAX_NUM_SEQS_LADDER = [512, 1024, 2048]
ESCALATION_HEADROOM = 0.88

# Patch A: sizing constants (replace the 15-request probe / 50-2000 floor-ceiling)
TARGET_DURATION_S = 200.0       # sustained measurement window per trial
MIN_TURNOVERS = 8.0             # num_prompts must be >= 8x concurrency
MIN_PROMPTS_FLOOR = 50          # absolute floor, binds only at 128K/256K
ACCEPT_MIN_DURATION_S = 150.0   # gate: reject a trial shorter than this
ACCEPT_MIN_TURNOVERS = 6.0      # gate: reject a trial with less recycling

# random-input-len safety margin. vLLM's own RandomDataset reserves 1 token
# of headroom for the server's special-token (BOS) addition, but its
# decode-then-reencode length correction is documented as best-effort, not
# exact ("imperfect nature of the sampling procedure"). Reproduced live
# against this exact model/tokenizer at ctx=2048, n=2488 (captured full
# 400 response bodies, not status codes alone): 5/2488 requests landed
# exactly at the target instead of 1 below it, and the server then counted
# them as target+1 tokens -- every single observed case drifted by exactly
# 1 token, never more, confirmed via direct HTTP capture (not inferred).
# Checked the client-side prompt_len distribution (no server round-trip
# needed) at all six re-run contexts (2K/4K/8K/16K/32K/64K) at their real
# post-turnover-floor num_prompts (~5,372 requests total): diff from target
# was always in {-1, 0}, never positive, at every context. 8 tokens is 8x
# the largest drift ever observed in this data.
INPUT_LEN_MARGIN = 8

# Patch B: ctx -> int(floor(boot max concurrency)), populated in wait_for_boot()
EXPECTED_CONCURRENCY = {}


def status(heading, *lines):
    print(f"[STATUS] {heading}: {' | '.join(lines)}", flush=True)


def log(msg):
    print(f"[LOG] {msg}", flush=True)


def run_start_epoch():
    p = Path(os.environ.get("SWEEP_START_EPOCH_FILE", _REPO_ROOT / "logs" / "run_start_epoch.txt"))
    if p.exists():
        return int(p.read_text().strip())
    return int(time.time())


def backstop_hit():
    return (time.time() - run_start_epoch()) >= BACKSTOP_S


def point_timeout_for(ctx):
    return POINT_TIMEOUT_256K_S if ctx == 262144 else POINT_TIMEOUT_S


def model_for(fp8_mode):
    # v3: both headline and FP8-KV sensitivity arms use BF16 weights.
    # fp8_mode only layers --kv-cache-dtype fp8 and FlashInfer on top.
    return MODEL_BF16


def ssh(ip, remote_cmd, timeout=60, retries=5):
    delay = 5
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = subprocess.run(
                [
                    "ssh", "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
                    "-i", SSH_KEY, f"{REMOTE_USER}@{ip}", remote_cmd,
                ],
                capture_output=True, text=True, timeout=timeout,
            )
            return r
        except subprocess.TimeoutExpired as e:
            last_err = e
        except Exception as e:
            last_err = e
        if attempt < retries:
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"SSH to {ip} failed after {retries} attempts: {last_err}")


def ssh_bg(ip, remote_cmd):
    return ssh(ip, f"nohup bash -c '{remote_cmd}' > /dev/null 2>&1 &", timeout=20)


def kill_server(ip):
    ssh(
        ip,
        "pkill -9 -f '[v]llm serve' 2>/dev/null; pkill -9 -f '[E]ngineCore' 2>/dev/null; "
        "pkill -9 -f '[b]ench serve' 2>/dev/null; true",
        timeout=20,
    )


def boot_server(ip, label, ctx, max_num_seqs, fp8_mode, model):
    startup_log = f"{REMOTE_LOG_DIR}/vllm_startup_{ctx}.log"
    kill_server(ip)
    time.sleep(2)
    # FIX 3: use --attention-backend FLASHINFER (v0.27.1 does not honor VLLM_ATTENTION_BACKEND)
    attn_flag = "--attention-backend FLASHINFER " if fp8_mode else ""
    kv_flag = "--kv-cache-dtype fp8 " if fp8_mode else ""
    cmd = (
        f"cd /root && source /root/vllm_env/bin/activate && vllm serve {model} "
        f"--served-model-name {SERVED_NAME} --max-model-len {ctx} "
        f"--max-num-seqs {max_num_seqs} --no-enable-prefix-caching {kv_flag}{attn_flag}"
        f"--gpu-memory-utilization {GPU_MEM_UTIL} --port {PORT} "
        f"> {startup_log} 2>&1 &"
    )
    ssh_bg(ip, cmd)


def wait_for_boot(ip, ctx, deadline_ts, boot_timeout_s=360):
    startup_log = f"{REMOTE_LOG_DIR}/vllm_startup_{ctx}.log"
    start = time.time()
    while time.time() - start < boot_timeout_s:
        if time.time() >= deadline_ts:
            return None, "deadline_exceeded"
        r = ssh(
            ip,
            f"grep -E 'Maximum concurrency|Uvicorn running|Application startup complete|Traceback' "
            f"{startup_log} 2>/dev/null || true",
            timeout=20,
        )
        out = r.stdout
        m = re.search(r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)", out)
        if m:
            boot_conc = float(m.group(1))
            # Patch B: populate EXPECTED_CONCURRENCY right where the boot value is
            # parsed, so validate_trial's concurrency-fidelity check has a real
            # value to check against rather than a silently-empty dict.
            EXPECTED_CONCURRENCY[ctx] = max(1, int(boot_conc))
            log(f"ctx={ctx}: EXPECTED_CONCURRENCY populated = {EXPECTED_CONCURRENCY[ctx]} "
                f"(boot reported {boot_conc}x)")
            return boot_conc, None
        if "Traceback" in out and "startup complete" not in out.lower():
            time.sleep(5)
            r2 = ssh(ip, f"tail -40 {startup_log} 2>/dev/null || true", timeout=20)
            return None, f"boot_error: {r2.stdout[-800:]}"
        time.sleep(5)
    return None, "boot_timeout"


def confirm_model_in_startup(ip, ctx, expected_model):
    """FIX 1: refuse to treat a point as valid unless the exact HF repo string is in the log."""
    startup_log = f"{REMOTE_LOG_DIR}/vllm_startup_{ctx}.log"
    # Quote the repo string safely for remote grep -F
    r = ssh(
        ip,
        f"grep -F '{expected_model}' {startup_log} 2>/dev/null | head -3 || true",
        timeout=20,
    )
    if expected_model not in r.stdout:
        r2 = ssh(
            ip,
            f"grep -E 'model=|model ' {startup_log} 2>/dev/null | head -5 || true",
            timeout=20,
        )
        return False, r2.stdout[-500:]
    return True, r.stdout.strip()[:300]


def confirm_flashinfer_active(ip, ctx):
    """FIX 3: require explicit FlashInfer attention-backend confirmation in the startup log.

    v0.27.1 logs: 'Using AttentionBackendEnum.FLASHINFER backend.' (not the older
    'Using FLASHINFER attention backend' phrasing).
    """
    startup_log = f"{REMOTE_LOG_DIR}/vllm_startup_{ctx}.log"
    r = ssh(
        ip,
        f"grep -E 'AttentionBackendEnum|attention backend|FLASHINFER|FLASH_ATTN|FlashInfer resolved' "
        f"{startup_log} 2>/dev/null || true",
        timeout=20,
    )
    out = r.stdout
    ok = bool(
        re.search(r"AttentionBackendEnum\.FLASHINFER", out)
        or re.search(r"Using FLASHINFER attention backend", out, re.IGNORECASE)
    )
    # Reject if FLASH_ATTN was the selected backend
    if re.search(r"AttentionBackendEnum\.FLASH_ATTN\b", out) or re.search(
        r"Using FLASH_ATTN attention backend", out
    ):
        ok = False
    return ok, out[-800:]


def diagnose_flashinfer(ip, label, ctx, diag_deadline):
    """Up to 20 minutes of diagnosis for FIX 3; returns (resolved: bool, notes: str)."""
    notes = []
    r = ssh(ip, "source /root/vllm_env/bin/activate && pip show flashinfer-python 2>&1 | head -8", timeout=30)
    notes.append(f"pip show flashinfer-python:\n{r.stdout.strip()}")
    r = ssh(ip, "source /root/vllm_env/bin/activate && python3 -c 'import flashinfer,vllm; print(\"flashinfer\", flashinfer.__version__); print(\"vllm\", vllm.__version__)' 2>&1", timeout=30)
    notes.append(f"import check:\n{r.stdout.strip()}")
    r = ssh(ip, "nvidia-smi | head -5", timeout=20)
    notes.append(f"nvidia-smi:\n{r.stdout.strip()}")
    # Confirm --attention-backend is what the process cmdline actually used
    r = ssh(
        ip,
        "ps aux | grep -E '[v]llm serve' | head -3; "
        f"grep -E 'Unknown vLLM environment|attention-backend|FLASHINFER|FLASH_ATTN' "
        f"{REMOTE_LOG_DIR}/vllm_startup_{ctx}.log 2>/dev/null | head -20 || true",
        timeout=20,
    )
    notes.append(f"process/log clues:\n{r.stdout.strip()[-1000:]}")

    # Retry once with explicit flag if budget remains
    if time.time() < diag_deadline - 60:
        status(
            f"FlashInfer diagnosis retry boot on {label}/{ctx}",
            "Rebooting with --attention-backend FLASHINFER after collecting install/CUDA evidence.",
        )
        kill_server(ip)
        time.sleep(2)
        boot_server(ip, label, ctx, 512, True, MODEL_BF16)
        conc, err = wait_for_boot(ip, ctx, diag_deadline, boot_timeout_s=300)
        if err is None:
            ok, evidence = confirm_flashinfer_active(ip, ctx)
            notes.append(f"retry boot concurrency={conc} flashinfer_ok={ok}\nevidence:\n{evidence}")
            return ok, "\n---\n".join(notes)
        notes.append(f"retry boot failed: {err}")
    return False, "\n---\n".join(notes)


def _read_completed_failed(ip, result_file):
    r = ssh(
        ip,
        "python3 -c \"import json,sys; d=json.load(open(sys.argv[1])); "
        "print(d.get('completed',0), d.get('failed',0), d.get('num_prompts',0))\" "
        f"{result_file} 2>&1",
        timeout=20,
    )
    parts = r.stdout.strip().split()
    if len(parts) != 3:
        return None, None, None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None, None, None


def _probe_request_rate(ip, probe_out):
    """Read request_throughput from a probe result file. None if unavailable."""
    r = ssh(ip, f"cat {probe_out} 2>/dev/null || true", timeout=20)
    try:
        d = json.loads(r.stdout)
        v = d.get("request_throughput")
        return float(v) if v and float(v) > 0 else None
    except Exception:
        return None


def _has_detailed_per_request(ip, result_file):
    """Confirm --save-detailed actually persisted per-request data (auditable warm-up separation)."""
    r = ssh(
        ip,
        "python3 -c \"import json,sys; d=json.load(open(sys.argv[1])); "
        "keys=sorted(d.keys()); "
        "detail=[k for k in keys if any(x in k.lower() for x in "
        "('ttft','itl','latency','request','per_','detailed','input_len','output_len'))]; "
        "print('KEYS', ','.join(keys)); "
        "print('HAS_LIST', any(isinstance(d.get(k), list) and len(d.get(k) or [])>0 for k in d)); "
        "print('DETAIL_KEYS', ','.join(detail[:20]))\" "
        f"{result_file} 2>&1",
        timeout=30,
    )
    return "HAS_LIST True" in r.stdout, r.stdout.strip()


def run_remote_bg_and_poll(ip, inner_cmd, marker_file, deadline_ts, poll_interval=10, what="command"):
    ssh(ip, f"rm -f {marker_file}", timeout=15)
    ssh_bg(ip, f"{inner_cmd}; echo EXIT=$? > {marker_file}")
    while True:
        if time.time() >= deadline_ts:
            raise RuntimeError(
                f"{what} did not finish before the point deadline (still running remotely, unconfirmed)"
            )
        r = ssh(ip, f"cat {marker_file} 2>/dev/null || true", timeout=15)
        if "EXIT=" in r.stdout:
            return r.stdout.strip()
        time.sleep(poll_interval)


def probe_and_size(ip, ctx, max_concurrency, deadline_ts, model):
    # Concurrency comes from the boot-log ceiling, unclamped. The previous
    # min(..., 64) silently overrode this at 2K/4K/8K and was not documented
    # in the methodology.
    conc = max(1, int(max_concurrency))
    input_len = ctx - OUTPUT_LEN - INPUT_LEN_MARGIN
    probe_out = f"/tmp/probe_{ctx}.json"
    marker = f"/tmp/probe_{ctx}.done"

    # Probe must FILL the batch, or it measures latency-bound throughput and
    # underestimates the sustained rate. One full wave is the minimum that
    # exercises every scheduler slot.
    probe_prompts = conc

    inner = (
        f"cd /root && source /root/vllm_env/bin/activate && vllm bench serve --backend vllm "
        f"--model {SERVED_NAME} --tokenizer {model} "
        f"--host localhost --port {PORT} --dataset-name random "
        f"--random-input-len {input_len} --random-output-len {OUTPUT_LEN} "
        f"--num-prompts {probe_prompts} --request-rate inf --max-concurrency {conc} "
        f"--save-result --result-filename {probe_out} > /tmp/probe_{ctx}_stdout.log 2>&1"
    )
    t0 = time.time()
    exit_line = run_remote_bg_and_poll(ip, inner, marker, deadline_ts, what=f"probe for ctx={ctx}")
    elapsed = max(0.5, time.time() - t0)
    completed, failed, _ = _read_completed_failed(ip, probe_out)
    if "EXIT=0" not in exit_line or not completed:
        r2 = ssh(ip, f"tail -40 /tmp/probe_{ctx}_stdout.log 2>/dev/null || true", timeout=20)
        raise RuntimeError(
            f"probe request failed (exit ok={'EXIT=0' in exit_line}, completed={completed}, "
            f"failed={failed}): {r2.stdout[-800:]}"
        )

    # Prefer the probe's own reported throughput over wall-clock, which
    # includes remote launch and teardown overhead.
    rate = _probe_request_rate(ip, probe_out)
    if rate is None:
        rate = float(completed) / elapsed
    rate = max(rate, 1e-6)

    # Three independent floors, then a deadline-derived ceiling.
    by_duration = int(rate * TARGET_DURATION_S)          # sustained-window target
    by_turnover = int(conc * MIN_TURNOVERS)              # batch must recycle
    num_prompts = max(MIN_PROMPTS_FLOOR, by_duration, by_turnover)

    # Never size a point we cannot finish. Leave 40% of the remaining point
    # budget as headroom for boot, warm-up and the other trials.
    remaining = max(60.0, deadline_ts - time.time())
    affordable = int(rate * remaining * 0.60)
    if affordable < num_prompts:
        log(f"ctx={ctx}: sizing reduced {num_prompts} -> {affordable} "
            f"(deadline-limited, rate={rate:.3f} req/s, remaining={remaining:.0f}s)")
        num_prompts = max(MIN_PROMPTS_FLOOR, affordable)
        deadline_limited = True
    else:
        deadline_limited = False

    turnovers = num_prompts / float(conc)
    log(f"ctx={ctx}: conc={conc} num_prompts={num_prompts} "
        f"turnovers={turnovers:.2f} est_duration={num_prompts / rate:.0f}s "
        f"probe_rate={rate:.3f} req/s deadline_limited={deadline_limited}")

    return num_prompts, conc


def run_warmup(ip, ctx, trial_idx, max_concurrency, model, deadline_ts):
    """FIX 2: separate, explicitly discarded warm-up of exactly 10 requests. Not recorded as data."""
    input_len = ctx - OUTPUT_LEN - INPUT_LEN_MARGIN
    stdout_log = f"{REMOTE_LOG_DIR}/warmup_stdout_{ctx}_trial{trial_idx}.log"
    marker = f"{REMOTE_LOG_DIR}/warmup_{ctx}_trial{trial_idx}.done"
    # Intentionally do NOT --save-result to the trial data path; optional throwaway file only
    throwaway = f"/tmp/warmup_{ctx}_trial{trial_idx}.json"
    inner = (
        f"cd /root && source /root/vllm_env/bin/activate && vllm bench serve --backend vllm "
        f"--model {SERVED_NAME} --tokenizer {model} "
        f"--host localhost --port {PORT} --dataset-name random "
        f"--random-input-len {input_len} --random-output-len {OUTPUT_LEN} "
        f"--num-prompts {WARMUP_REQUESTS} --request-rate inf --max-concurrency {max_concurrency} "
        f"--save-result --result-filename {throwaway} > {stdout_log} 2>&1"
    )
    exit_line = run_remote_bg_and_poll(
        ip, inner, marker, deadline_ts, what=f"warm-up for trial {trial_idx} ctx={ctx}"
    )
    completed, failed, requested = _read_completed_failed(ip, throwaway)
    if "EXIT=0" not in exit_line or not completed:
        r2 = ssh(ip, f"tail -30 {stdout_log} 2>/dev/null || true", timeout=20)
        raise RuntimeError(
            f"warm-up failed (completed={completed}, failed={failed}): {r2.stdout[-500:]}"
        )
    # Leave a one-line audit marker in the remote log dir (not used as trial data)
    ssh(
        ip,
        f"echo 'warmup_trial={trial_idx} completed={completed} failed={failed} "
        f"requested={requested} discarded=yes' > {REMOTE_LOG_DIR}/warmup_{ctx}_trial{trial_idx}.audit",
        timeout=15,
    )
    return completed


def run_trial(ip, ctx, trial_idx, num_prompts, max_concurrency, label, model, deadline_ts):
    input_len = ctx - OUTPUT_LEN - INPUT_LEN_MARGIN
    result_file = f"{REMOTE_LOG_DIR}/bench_{ctx}_trial{trial_idx}.json"
    stdout_log = f"{REMOTE_LOG_DIR}/bench_stdout_{ctx}_trial{trial_idx}.log"
    marker = f"{REMOTE_LOG_DIR}/bench_{ctx}_trial{trial_idx}.done"
    # FIX 2: --save-detailed persists per-request completion data
    inner = (
        f"cd /root && source /root/vllm_env/bin/activate && vllm bench serve --backend vllm "
        f"--model {SERVED_NAME} --tokenizer {model} "
        f"--host localhost --port {PORT} --dataset-name random "
        f"--random-input-len {input_len} --random-output-len {OUTPUT_LEN} "
        f"--num-prompts {num_prompts} --request-rate inf --max-concurrency {max_concurrency} "
        f"--save-result --save-detailed --result-filename {result_file} > {stdout_log} 2>&1"
    )
    exit_line = run_remote_bg_and_poll(
        ip, inner, marker, deadline_ts, what=f"trial {trial_idx} on {label}/{ctx}"
    )
    completed, failed, requested = _read_completed_failed(ip, result_file)
    ok = (
        "EXIT=0" in exit_line
        and completed is not None
        and requested
        and completed >= 0.9 * requested
    )
    if not ok:
        r2 = ssh(ip, f"tail -40 {stdout_log} 2>/dev/null || true", timeout=20)
        raise RuntimeError(
            f"trial {trial_idx} failed or under-completed "
            f"(completed={completed}, failed={failed}, requested={requested}): {r2.stdout[-800:]}"
        )
    has_detail, detail_info = _has_detailed_per_request(ip, result_file)
    if not has_detail:
        raise RuntimeError(
            f"trial {trial_idx} missing per-request detailed data despite --save-detailed: {detail_info}"
        )
    metrics_log = f"{REMOTE_LOG_DIR}/metrics_{ctx}_trial{trial_idx}.log"
    ssh(
        ip,
        f"curl -s localhost:{PORT}/metrics | grep -E 'num_preemptions_total|kv_cache_usage_perc' "
        f"> {metrics_log} || true",
        timeout=30,
    )
    return exit_line


def validate_trial(result_path, ip, ctx, label, trial):
    """
    Post-hoc acceptance gate. Returns (ok, reason).
    Reads what the benchmark actually did, not what we asked for.
    """
    # Patch A's turnover floor can produce multi-MB --save-detailed result files
    # at high-concurrency short contexts (e.g. ~18MB at 2K with conc=311,
    # num_prompts~2488). A 20s SSH timeout was fine for v3's small files but
    # timed out transferring these, turning genuinely-valid trials into false
    # "Trial failure" exceptions. 180s gives real headroom for large payloads.
    r = ssh(ip, f"cat {result_path} 2>/dev/null || true", timeout=180)
    try:
        d = json.loads(r.stdout)
    except Exception as e:
        return False, f"unreadable result json: {e}"

    duration = float(d.get("duration") or 0.0)
    completed = int(d.get("completed") or 0)
    failed = int(d.get("failed") or 0)
    conc_cfg = d.get("max_concurrency")
    conc_obs = int(d.get("max_concurrent_requests") or 0)

    if failed:
        return False, f"{failed} failed requests"
    if not completed:
        return False, "zero completed requests"

    # 1. Sustained window. Floor only. Long context legitimately runs far
    #    past TARGET_DURATION_S and must not be rejected for that.
    if duration < ACCEPT_MIN_DURATION_S:
        return False, (f"duration {duration:.1f}s < {ACCEPT_MIN_DURATION_S:.0f}s floor; "
                       f"measurement is fill-transient dominated, not sustained load")

    # 2. Batch recycling. This is the check that would have caught 2K at
    #    0.60 turnovers and 8K at 1.68.
    if conc_cfg:
        turnovers = completed / float(conc_cfg)
        if turnovers < ACCEPT_MIN_TURNOVERS:
            return False, (f"turnovers {turnovers:.2f} < {ACCEPT_MIN_TURNOVERS:.0f} "
                           f"({completed} completed / {conc_cfg} concurrency)")

    # 3. Concurrency fidelity. The configured value must match the boot-log
    #    ceiling we intended, so no undocumented override can reappear.
    expected = EXPECTED_CONCURRENCY.get(ctx)
    if expected is not None and conc_cfg is not None and int(conc_cfg) != int(expected):
        return False, (f"max_concurrency {conc_cfg} != boot-derived {expected}; "
                       f"an override is in the path")

    # 4. Scheduler-pin check the article promises. --max-num-seqs is 512.
    if conc_obs >= 512:
        return False, f"observed concurrency {conc_obs} reached --max-num-seqs 512; scheduler-limited"

    # Patch C: log vLLM's own (bucketed) peak vs the true interval-sweep peak,
    # alongside the pass/fail decision above. Does not affect the gate itself.
    true_peak = true_max_concurrency(result_path, ip)
    reported_peak = d.get("max_concurrent_requests")
    if true_peak is not None and reported_peak is not None and int(true_peak) != int(reported_peak):
        log(f"{label}/{ctx} trial{trial}: vLLM reported {reported_peak}, true concurrency {true_peak} "
            f"(bucketing artifact, expected)")

    return True, (f"ok: duration={duration:.1f}s completed={completed} "
                  f"conc={conc_cfg} turnovers={completed / float(conc_cfg or 1):.2f}")


def true_max_concurrency(result_path, ip):
    """
    Millisecond-precision interval overlap over raw start_times/ttfts/itls,
    not the whole-second bucket count vLLM's own bench serve reports in
    max_concurrent_requests. Returns the true peak concurrent request count.
    """
    # Same large-file timeout headroom as validate_trial (see note there).
    r = ssh(ip, f"cat {result_path} 2>/dev/null || true", timeout=180)
    d = json.loads(r.stdout)

    starts = d.get("start_times") or []
    ttfts = d.get("ttfts") or []
    itls = d.get("itls") or []
    if not starts or len(starts) != len(ttfts):
        return None  # not enough data to compute; caller should fall back

    events = []
    for i, s in enumerate(starts):
        # end time = start + ttft + sum of inter-token latencies for this request
        total_itl = sum(itls[i]) if i < len(itls) and itls[i] else 0.0
        end = s + ttfts[i] + total_itl
        events.append((s, 1))     # request opens
        events.append((end, -1))  # request closes

    events.sort(key=lambda e: (e[0], e[1]))  # closes before opens at exact ties
    concurrent = 0
    peak = 0
    for _, delta in events:
        concurrent += delta
        peak = max(peak, concurrent)
    return peak


def rsync_logs(ip, local_subdir):
    subprocess.run(
        [
            "rsync", "-avz", "-e",
            f"ssh -o StrictHostKeyChecking=accept-new -i {SSH_KEY}",
            f"{REMOTE_USER}@{ip}:{REMOTE_LOG_DIR}/", str(local_subdir) + "/",
        ],
        capture_output=True, text=True, timeout=120,
    )


def process_point(ip, label, ctx, fp8_mode, local_subdir):
    model = model_for(fp8_mode)
    point_deadline = time.time() + point_timeout_for(ctx)
    if backstop_hit():
        status(
            f"8-HOUR BACKSTOP HIT before {label}/{ctx}",
            f"Droplet {label}, context {ctx}: whole-run backstop reached, stopping this Droplet's remaining work.",
            "Droplets are left running. Delete them from the control panel when you are done.",
        )
        return "backstop"

    concurrency = None
    used_mns = None
    scheduler_note = None
    flashinfer_ok = not fp8_mode  # only required on C

    for i, mns in enumerate(MAX_NUM_SEQS_LADDER):
        if time.time() >= point_deadline:
            status(
                f"_TIMEOUT on {label}/{ctx}",
                f"Droplet {label} context {ctx} exceeded the per-point timeout during boot/escalation "
                f"(budget={point_timeout_for(ctx)//60} min).",
            )
            kill_server(ip)
            return "timeout"
        boot_server(ip, label, ctx, mns, fp8_mode, model)
        conc, err = wait_for_boot(ip, ctx, point_deadline)
        if err == "deadline_exceeded":
            status(f"_TIMEOUT on {label}/{ctx}", "Point timeout hit while waiting for boot.")
            kill_server(ip)
            return "timeout"
        if err is not None:
            status(f"_BOOT_FAILED on {label}/{ctx}", f"Droplet {label} context {ctx} failed to boot cleanly: {err[:300]}")
            kill_server(ip)
            return "boot_failed"

        # FIX 1: confirm exact HF repo string in startup log
        model_ok, model_evidence = confirm_model_in_startup(ip, ctx, model)
        if not model_ok:
            status(
                f"_BOOT_FAILED on {label}/{ctx}",
                f"Expected model repo {model} not found in startup log. Evidence: {model_evidence[:400]}",
            )
            kill_server(ip)
            return "boot_failed"

        # FIX 3: FlashInfer hard gate on C
        if fp8_mode:
            flashinfer_ok, fi_evidence = confirm_flashinfer_active(ip, ctx)
            if not flashinfer_ok:
                status(
                    f"FlashInfer NOT confirmed on {label}/{ctx} — entering diagnosis budget",
                    f"Startup evidence:\n{fi_evidence[:600]}",
                )
                diag_deadline = min(point_deadline, time.time() + FLASHINFER_DIAG_BUDGET_S)
                resolved, diag_notes = diagnose_flashinfer(ip, label, ctx, diag_deadline)
                if not resolved:
                    status(
                        f"_ENVIRONMENT_BLOCKED on {label}/{ctx}",
                        "FlashInfer attention backend not confirmed after up to 20 minutes of diagnosis. "
                        "Stopping further work on this Droplet rather than producing mislabeled results.",
                        diag_notes[-1500:],
                    )
                    kill_server(ip)
                    return "environment_blocked"
                flashinfer_ok = True
                status(
                    f"FlashInfer confirmed after diagnosis on {label}/{ctx}",
                    "Proceeding to load testing.",
                )

        concurrency = conc
        used_mns = mns
        if conc < mns * ESCALATION_HEADROOM or i == len(MAX_NUM_SEQS_LADDER) - 1:
            if i > 0:
                status(
                    f"Scheduler-pin escalation resolved on {label}/{ctx}",
                    f"Escalated max-num-seqs from {MAX_NUM_SEQS_LADDER[0]} to {mns}, reported concurrency now {conc}.",
                )
            break
        status(
            f"Scheduler-pin escalation triggered on {label}/{ctx}",
            f"Reported concurrency {conc} was within headroom threshold of max-num-seqs={mns}; "
            f"escalating to {MAX_NUM_SEQS_LADDER[i+1]} and rechecking.",
        )
        kill_server(ip)
        time.sleep(3)

    if used_mns == MAX_NUM_SEQS_LADDER[-1] and concurrency is not None and concurrency >= used_mns * ESCALATION_HEADROOM:
        scheduler_note = "_SCHEDULER_LIMITED_UNRESOLVED"
        status(
            f"_SCHEDULER_LIMITED_UNRESOLVED on {label}/{ctx}",
            f"Still looks scheduler-bound after 2 escalations (max-num-seqs={used_mns}, reported concurrency={concurrency}).",
        )

    if time.time() >= point_deadline:
        status(f"_TIMEOUT on {label}/{ctx}", "Point timeout hit right after boot, before benching.")
        kill_server(ip)
        return "timeout"

    # Patch B: don't trust EXPECTED_CONCURRENCY was actually filled by wait_for_boot;
    # assert and log it before it's relied on by validate_trial's check 3.
    assert ctx in EXPECTED_CONCURRENCY, (
        f"EXPECTED_CONCURRENCY not populated for ctx={ctx} before first validate_trial call"
    )
    log(f"{label}/{ctx}: EXPECTED_CONCURRENCY[{ctx}] = {EXPECTED_CONCURRENCY[ctx]}")

    try:
        num_prompts, bench_conc = probe_and_size(ip, ctx, concurrency, point_deadline, model)
    except Exception as e:
        status(f"_BOOT_FAILED on {label}/{ctx}", f"Probe request failed: {e}")
        kill_server(ip)
        return "boot_failed"

    trials_ok = 0
    for trial in range(1, TRIALS_PER_POINT + 1):
        if time.time() >= point_deadline:
            status(f"_TIMEOUT on {label}/{ctx}", f"Point timeout hit during trial {trial}/{TRIALS_PER_POINT}.")
            kill_server(ip)
            return "timeout"
        try:
            # FIX 2: separate warm-up, then timed trial with --save-detailed
            wu = run_warmup(ip, ctx, trial, bench_conc, model, point_deadline)
            status(
                f"Warm-up discarded on {label}/{ctx} trial {trial}",
                f"Separate {WARMUP_REQUESTS}-request warm-up completed={wu}; output discarded, not counted as trial data.",
            )
            run_trial(ip, ctx, trial, num_prompts, bench_conc, label, model, point_deadline)
            result_file = f"{REMOTE_LOG_DIR}/bench_{ctx}_trial{trial}.json"
            ok, reason = validate_trial(result_file, ip, ctx, label, trial)
            if not ok:
                status(f"_TRIAL_REJECTED {label}/{ctx} trial{trial}", reason)
                kill_server(ip)
                return "trial_rejected"
            log(f"{label}/{ctx} trial{trial} accepted: {reason}")
            trials_ok += 1
        except Exception as e:
            status(f"Trial failure on {label}/{ctx}", f"Trial {trial} raised: {e}. Continuing to next trial.")

    kill_server(ip)
    rsync_logs(ip, local_subdir)

    if trials_ok == 0:
        status(
            f"_TRIALS_FAILED on {label}/{ctx}",
            f"All {TRIALS_PER_POINT} trials failed to produce a valid detailed result file.",
        )
        return "trials_failed"

    suffix = f" [{scheduler_note}]" if scheduler_note else ""
    if trials_ok < TRIALS_PER_POINT:
        suffix += f" [only {trials_ok}/{TRIALS_PER_POINT} trials produced valid results]"
    if fp8_mode:
        suffix += " [FlashInfer confirmed]"
    status(
        f"Point complete: {label}/{ctx}{suffix}",
        f"model={model}, max-num-seqs={used_mns}, reported concurrency={concurrency}, "
        f"num_prompts per trial={num_prompts}, trials_ok={trials_ok}/{TRIALS_PER_POINT}, "
        f"warm-up=separate {WARMUP_REQUESTS}-request invocation before each trial (discarded), "
        f"--save-detailed=verified on each accepted trial result file.",
    )
    return "ok"


def append_missing_trials(ip, label, ctx, fp8_mode, local_subdir, trial_indices, num_prompts=None, max_concurrency=None):
    """Top up specific trial indices without redoing already-valid trials.

    Boots once, then for each trial_idx in trial_indices: warm-up + timed trial.
    Uses the same gates (BF16 repo confirm, FlashInfer on C) as process_point.
    """
    model = model_for(fp8_mode)
    point_deadline = time.time() + point_timeout_for(ctx)
    if backstop_hit():
        status(f"8-HOUR BACKSTOP HIT before append {label}/{ctx}", "Stopping.")
        return "backstop"

    status(
        f"Append trials starting on {label}/{ctx}",
        f"Will run trial indices {trial_indices} only (not redoing existing). "
        f"model={model}, fp8_kv={fp8_mode}, budget_min={point_timeout_for(ctx)//60}",
    )

    boot_server(ip, label, ctx, 512, fp8_mode, model)
    conc, err = wait_for_boot(ip, ctx, point_deadline)
    if err is not None:
        status(f"_BOOT_FAILED on append {label}/{ctx}", f"{err}")
        kill_server(ip)
        return "boot_failed"

    model_ok, model_evidence = confirm_model_in_startup(ip, ctx, model)
    if not model_ok:
        status(f"_BOOT_FAILED on append {label}/{ctx}", f"Model not confirmed: {model_evidence[:400]}")
        kill_server(ip)
        return "boot_failed"

    if fp8_mode:
        fi_ok, fi_ev = confirm_flashinfer_active(ip, ctx)
        if not fi_ok:
            status(f"_ENVIRONMENT_BLOCKED on append {label}/{ctx}", f"FlashInfer not confirmed:\n{fi_ev[:600]}")
            kill_server(ip)
            return "environment_blocked"

    # Patch B: assert/log EXPECTED_CONCURRENCY here too — append mode is a second
    # entry point into the same trial-running path validate_trial's check 3 guards.
    assert ctx in EXPECTED_CONCURRENCY, (
        f"EXPECTED_CONCURRENCY not populated for ctx={ctx} before first validate_trial call (append)"
    )
    log(f"{label}/{ctx} (append): EXPECTED_CONCURRENCY[{ctx}] = {EXPECTED_CONCURRENCY[ctx]}")

    if max_concurrency is None:
        max_concurrency = max(1, int(conc))
    if num_prompts is None:
        try:
            num_prompts, max_concurrency = probe_and_size(ip, ctx, conc, point_deadline, model)
        except Exception as e:
            status(f"_BOOT_FAILED on append {label}/{ctx}", f"Probe failed: {e}")
            kill_server(ip)
            return "boot_failed"

    trials_ok = 0
    for trial in trial_indices:
        if time.time() >= point_deadline:
            status(f"_TIMEOUT on append {label}/{ctx}", f"Budget exhausted before trial {trial}.")
            break
        try:
            wu = run_warmup(ip, ctx, trial, max_concurrency, model, point_deadline)
            status(
                f"Warm-up discarded on {label}/{ctx} trial {trial} (append)",
                f"Separate {WARMUP_REQUESTS}-request warm-up completed={wu}; discarded.",
            )
            run_trial(ip, ctx, trial, num_prompts, max_concurrency, label, model, point_deadline)
            result_file = f"{REMOTE_LOG_DIR}/bench_{ctx}_trial{trial}.json"
            ok, reason = validate_trial(result_file, ip, ctx, label, trial)
            if not ok:
                status(f"_TRIAL_REJECTED {label}/{ctx} trial{trial} (append)", reason)
                kill_server(ip)
                return "trial_rejected"
            trials_ok += 1
            status(f"Append trial ok: {label}/{ctx} trial {trial}", f"Verified --save-detailed result. {reason}")
        except Exception as e:
            status(f"Append trial failure on {label}/{ctx}", f"Trial {trial}: {e}")

    kill_server(ip)
    rsync_logs(ip, local_subdir)
    status(
        f"Append finished: {label}/{ctx}",
        f"Requested trials {trial_indices}; newly completed this append: {trials_ok}.",
    )
    return "ok" if trials_ok else "trials_failed"


def main():
    # Modes:
    #   normal:  label ip contexts fp8|bf16 local_subdir
    #   append:  append label ip ctx fp8|bf16 local_subdir trial_csv [num_prompts] [max_conc]
    if sys.argv[1] == "append":
        label = sys.argv[2]
        ip = sys.argv[3]
        ctx = int(sys.argv[4])
        fp8_mode = sys.argv[5] == "fp8"
        local_subdir = LOG_ROOT / sys.argv[6]
        trial_indices = [int(x) for x in sys.argv[7].split(",")]
        num_prompts = int(sys.argv[8]) if len(sys.argv) > 8 else None
        max_conc = int(sys.argv[9]) if len(sys.argv) > 9 else None
        append_missing_trials(ip, label, ctx, fp8_mode, local_subdir, trial_indices, num_prompts, max_conc)
        return

    label = sys.argv[1]
    ip = sys.argv[2]
    contexts = [int(x) for x in sys.argv[3].split(",")]
    fp8_mode = sys.argv[4] == "fp8"
    local_subdir = LOG_ROOT / sys.argv[5]
    model = model_for(fp8_mode)

    status(
        f"Droplet {label} sweep starting (v4)",
        f"IP {ip}, contexts {contexts}, fp8_kv_mode={fp8_mode}, model={model}, "
        f"backstop=8h, 256K_timeout=3.5h, other_timeout=45m",
    )

    for ctx in contexts:
        if backstop_hit():
            status(
                f"8-HOUR BACKSTOP HIT before {label}/{ctx}",
                "Stopping this Droplet's remaining work. Droplets left running.",
            )
            break
        try:
            result = process_point(ip, label, ctx, fp8_mode, local_subdir)
        except Exception as e:
            status(
                f"Unhandled error on {label}/{ctx}",
                f"{e}. Attempting to kill server and continue to next context length.",
            )
            try:
                kill_server(ip)
            except Exception:
                pass
            continue
        if result in ("backstop", "environment_blocked"):
            break

    status(f"Droplet {label} sweep finished", f"All assigned contexts processed or skipped: {contexts}")


if __name__ == "__main__":
    main()
