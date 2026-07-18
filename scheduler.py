#!/usr/bin/env python3
"""Autonomous training scheduler for RAP2P on 2x RTX 4090.

Polls every 30s. When a GPU's job completes (checkpoint exists), starts the
next pending job whose dependencies are met. After each job completes, pushes
the checkpoint metadata (history.json) to GitHub for crash recovery.
Logs to /data/lab/scheduler.log.
"""
import os, subprocess, time, sys, json
from pathlib import Path

LAB = Path("/data/lab/rap2p")
CKPT = LAB / "artifacts" / "checkpoints"
SCHED_LOG = Path("/data/lab/scheduler.log")
HF_TOKEN = open("/data/lab/.env").read().split("HF_TOKEN=")[1].split("\n")[0].strip()

ALL_JOBS = [
    ("global_qlora", 1701), ("context_qlora", 1701),
    ("global_qlora", 7), ("context_qlora", 7),
    ("p2p_static", 1701), ("rap2p", 1701),
    ("p2p_static", 7), ("rap2p", 7),
    ("rap2p", 42), ("rap2p_no_graph", 1701),
    ("rap2p_no_history_retrained", 1701),
]
NEEDS_GQ = {"p2p_static", "rap2p", "rap2p_no_graph", "rap2p_no_history_retrained"}

# Target optimizer steps per run (upper end of the config's steps range, which
# is what training.py runs to). A job is DONE only when history.json shows the
# final step reached -- NOT merely when last.pt exists (training.py writes
# last.pt at the FIRST validation, step 200, which is far from finished).
TARGET_STEPS = {
    "global_qlora": 1800, "context_qlora": 1800,
    "p2p_static": 1200, "rap2p": 1200,
    "rap2p_no_graph": 1200, "rap2p_no_history_retrained": 1200,
}

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(SCHED_LOG, "a") as f:
        f.write(line + "\n")

def ckpt_done(run, seed):
    """A run is finished only when its history.json shows the target step
    reached. Merely having last.pt is NOT enough: training.py checkpoints at
    every validation (first at step 200), so last.pt appears long before the
    run converges."""
    d = CKPT / f"{run}_seed{seed}"
    hist = d / "history.json"
    if not hist.exists():
        return False
    try:
        h = json.loads(hist.read_text())
        if not h:
            return False
        target = TARGET_STEPS.get(run, 10**9)
        return int(h[-1].get("step", 0)) >= target
    except Exception:
        return False

def gq_available():
    return ckpt_done("global_qlora", 1701) or ckpt_done("global_qlora", 7)

def gpu_busy(gpu):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except:
        return False

def push_history_to_git(run, seed):
    """Push history.json (small metadata) to GitHub for crash recovery.
    Checkpoint .pt files are too large for git; only history.json is pushed."""
    try:
        d = CKPT / f"{run}_seed{seed}"
        hist = d / "history.json"
        if not hist.exists():
            return
        subprocess.run(["git", "add", "-f", str(hist.relative_to(LAB))], cwd=str(LAB), capture_output=True, timeout=10)
        subprocess.run(["git", "commit", "-m", f"checkpoint: {run} s{seed} history.json"], cwd=str(LAB), capture_output=True, timeout=15)
        # Pull remote first (paper commits, other checkpoint pushes) to avoid
        # non-fast-forward rejection, then push.
        subprocess.run(["git", "pull", "--no-edit", "--no-rebase", "origin", "main"], cwd=str(LAB), capture_output=True, timeout=60)
        r = subprocess.run(["git", "push", "origin", "main"], cwd=str(LAB), capture_output=True, timeout=60, text=True)
        if r.returncode == 0:
            log(f"  pushed {run} s{seed} history.json to GitHub")
        else:
            log(f"  push {run} s{seed} failed: {r.stderr[-200:]}")
    except Exception as e:
        log(f"  push failed for {run} s{seed}: {e}")

def start_job(run, seed, gpu):
    log(f"START {run} s{seed} GPU={gpu}")
    env = os.environ.copy()
    env["HF_TOKEN"] = HF_TOKEN
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, str(LAB / "scripts/train.py"),
           "--config", str(LAB / "configs/mvp.yaml"),
           "--run", run, "--seed", str(seed)]
    logfile = open(f"/data/lab/train_{run}_s{seed}.log", "w")
    proc = subprocess.Popen(cmd, cwd=str(LAB), env=env, stdout=logfile, stderr=subprocess.STDOUT)
    log(f"  PID={proc.pid}")
    return proc

def main():
    log("=== Scheduler started ===")
    completed = set()
    for run, seed in ALL_JOBS:
        if ckpt_done(run, seed):
            completed.add((run, seed))
            log(f"ALREADY DONE: {run} s{seed}")

    running = {}
    last_push = {}

    while len(completed) < len(ALL_JOBS):
        time.sleep(30)

        # Check completions
        for gpu in list(running):
            run, seed, proc = running[gpu]
            done = ckpt_done(run, seed)
            failed = (proc is not None and proc.poll() is not None and not done)
            if done:
                log(f"DONE: {run} s{seed} GPU={gpu}")
                completed.add((run, seed))
                push_history_to_git(run, seed)
                del running[gpu]
            elif failed:
                log(f"FAILED: {run} s{seed} GPU={gpu} exit={proc.returncode}")
                # Log last 5 lines of error
                try:
                    logname = f"/data/lab/train_{run}_s{seed}.log"
                    with open(logname) as f:
                        lines = f.readlines()[-5:]
                    for l in lines:
                        log(f"  {l.rstrip()}")
                except:
                    pass
                del running[gpu]

        # Start new jobs on free GPUs
        for gpu in [0, 1]:
            if gpu in running:
                continue
            for run, seed in ALL_JOBS:
                if (run, seed) in completed:
                    continue
                if any(r == run and s == seed for r, s, _ in running.values()):
                    continue
                if run in NEEDS_GQ and not gq_available():
                    continue
                proc = start_job(run, seed, gpu)
                running[gpu] = (run, seed, proc)
                break

    log(f"=== All {len(completed)} jobs complete ===")

if __name__ == "__main__":
    main()
