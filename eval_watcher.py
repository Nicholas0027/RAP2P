#!/usr/bin/env python3
"""Watch for all 11 training checkpoints to reach their TARGET step, then run
evaluation + figures. Uses step-based completion (NOT just last.pt existence,
since training.py writes last.pt at the first validation, step 200).

Polls every 60s. Logs to /data/lab/eval_watcher.log
"""
import os, subprocess, sys, time, json
from pathlib import Path

LAB = Path("/data/lab/rap2p")
CKPT = LAB / "artifacts" / "checkpoints"
LOG = Path("/data/lab/eval_watcher.log")

REQUIRED = [
    ("global_qlora", 1701, 1800), ("global_qlora", 7, 1800),
    ("context_qlora", 1701, 1800), ("context_qlora", 7, 1800),
    ("p2p_static", 1701, 1200), ("p2p_static", 7, 1200),
    ("rap2p", 1701, 1200), ("rap2p", 7, 1200), ("rap2p", 42, 1200),
    ("rap2p_no_graph", 1701, 1200),
    ("rap2p_no_history_retrained", 1701, 1200),
]

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def is_done(run, seed, target):
    """Done only when history.json shows step >= target. last.pt alone is NOT
    enough (it appears at step 200, far from convergence)."""
    hist = CKPT / f"{run}_seed{seed}" / "history.json"
    if not hist.exists():
        return False
    try:
        h = json.loads(hist.read_text())
        return bool(h) and int(h[-1].get("step", 0)) >= target
    except Exception:
        return False

def all_done():
    return all(is_done(r, s, t) for r, s, t in REQUIRED)

def run(cmd):
    log(f"RUN: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(LAB), capture_output=True, text=True, timeout=14400)
    log(f"EXIT={result.returncode}")
    if result.stdout:
        log(f"STDOUT tail: {result.stdout[-800:]}")
    if result.stderr:
        log(f"STDERR tail: {result.stderr[-800:]}")
    return result.returncode

def main():
    log("=== Eval watcher started (step-based completion) ===")
    while not all_done():
        done_count = sum(1 for r, s, t in REQUIRED if is_done(r, s, t))
        # also report max step per run for visibility
        parts = []
        for r, s, t in REQUIRED:
            hist = CKPT / f"{r}_seed{s}" / "history.json"
            try:
                h = json.loads(hist.read_text())
                parts.append(f"{r.split('_')[0][:6]}s{s}:{h[-1]['step']}/{t}")
            except:
                parts.append(f"{r.split('_')[0][:6]}s{s}:0/{t}")
        log(f"Waiting: {done_count}/{len(REQUIRED)} done | " + " ".join(parts))
        time.sleep(60)

    log("All checkpoints reached target step! Starting evaluation...")
    env = dict(os.environ)
    env["HF_TOKEN"] = open("/data/lab/.env").read().split("HF_TOKEN=")[1].split("\n")[0].strip()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    rc = run([sys.executable, str(LAB / "scripts/evaluate_all.py"), "--config", str(LAB / "configs/mvp.yaml")])
    if rc != 0:
        log("evaluate_all.py FAILED, skipping figures")
        return

    run([sys.executable, str(LAB / "scripts/make_figures.py"), "--config", str(LAB / "configs/mvp.yaml")])
    log("=== Evaluation + figures complete ===")

if __name__ == "__main__":
    main()
