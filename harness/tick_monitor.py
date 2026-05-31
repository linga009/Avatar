#!/usr/bin/env python3
"""Avatar tick health monitor — parses docker logs in real-time.

Tracks: tick number, r, emotion, chi, tau, F_thermo, tick duration.
Alerts on: stuck ticks (>300s), NaN, rising F_thermo, container exit.
Writes CSV to data/tick_health.csv.

Run: python harness/tick_monitor.py
"""
import re
import csv
import time
import subprocess
from datetime import datetime
from pathlib import Path

CSV_PATH = Path("data/tick_health.csv")
STUCK_THRESHOLD = 300  # seconds
F_RISING_THRESHOLD = 30  # consecutive COP reports with rising F

# Regex patterns for log parsing
TICK_RE = re.compile(
    r"Tick\s+(\d+)\s+\|\s+r=\[.*?\]\s+([\d.]+)\s+\|\s+"
    r"(\S+)\s+(\S+)\s+\(i=([\d.]+)\)\s+K=([\d.]+)\s+"
    r"chi=([\d.]+)\s+tau=([\d.]+)\s+U=([\d.]+)/([\d.]+)"
)
QUERY_RE = re.compile(r'q="([^"]+)".*?FE_.*?=([\S]+).*?=([\S]+)')
OVERRUN_RE = re.compile(r"Tick overrun:\s+([\d.]+)s")
COP_RE = re.compile(r"COP:.*?F=([\d.]+)")
DREAM_START_RE = re.compile(r"Entering dream|Dream cycle starting")
DREAM_END_RE = re.compile(r"Reloading physics|Dream cycle complete")
OOM_RE = re.compile(r"OOM|out of memory|exit code 137", re.IGNORECASE)


def init_csv():
    if not CSV_PATH.exists():
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "tick", "timestamp", "r", "emotion", "intensity",
                "K", "chi", "tau", "unity", "gap",
                "F_thermo", "query", "fe_delta", "pred_error", "duration_s",
            ])


def append_csv(row: dict):
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            row.get("tick", ""), row.get("timestamp", ""),
            row.get("r", ""), row.get("emotion", ""),
            row.get("intensity", ""), row.get("K", ""),
            row.get("chi", ""), row.get("tau", ""),
            row.get("unity", ""), row.get("gap", ""),
            row.get("F_thermo", ""), row.get("query", ""),
            row.get("fe_delta", ""), row.get("pred_error", ""),
            row.get("duration_s", ""),
        ])


def monitor():
    init_csv()
    print("[tick_monitor] Watching halo3-train-1 logs...")
    print("[tick_monitor] CSV output: data/tick_health.csv")
    print("[tick_monitor] Press Ctrl+C to stop.\n")

    last_tick_time = time.time()
    f_thermo_history = []
    f_rising_count = 0
    in_dream = False
    f_before_dream = None
    current_row = {}

    proc = subprocess.Popen(
        ["docker", "logs", "-f", "halo3-train-1"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    try:
        for line in proc.stdout:
            line = line.strip()
            now = datetime.now().isoformat(timespec="seconds")

            # Tick line
            m = TICK_RE.search(line)
            if m:
                last_tick_time = time.time()
                current_row = {
                    "tick": m.group(1), "timestamp": now,
                    "r": m.group(2), "emotion": m.group(3),
                    "intensity": m.group(5), "K": m.group(6),
                    "chi": m.group(7), "tau": m.group(8),
                    "unity": m.group(9), "gap": m.group(10),
                }

                # NaN check
                for k, v in current_row.items():
                    if "nan" in str(v).lower():
                        print(f"  [ALERT] NaN in {k} at tick {current_row['tick']}")

            # Query/FE line
            mq = QUERY_RE.search(line)
            if mq:
                current_row["query"] = mq.group(1)
                current_row["fe_delta"] = mq.group(2)
                current_row["pred_error"] = mq.group(3)

            # Tick overrun (duration) — also triggers CSV write
            mo = OVERRUN_RE.search(line)
            if mo:
                current_row["duration_s"] = mo.group(1)
                append_csv(current_row)
                t = current_row.get("tick", "?")
                r = current_row.get("r", "?")
                emo = current_row.get("emotion", "?")
                dur = mo.group(1)
                print(f"  Tick {t:>4s} | r={r} | {emo} | {dur}s")
                current_row = {}

            # COP report with F_thermo
            mc = COP_RE.search(line)
            if mc:
                f_val = float(mc.group(1))
                current_row["F_thermo"] = f_val
                f_thermo_history.append(f_val)

                # Track F rising
                if len(f_thermo_history) >= 2:
                    if f_thermo_history[-1] > f_thermo_history[-2]:
                        f_rising_count += 1
                    else:
                        f_rising_count = 0

                if f_rising_count >= F_RISING_THRESHOLD:
                    print(f"  [ALERT] F_thermo rising for {f_rising_count} reports — thermodynamic divergence")

                trend = "rising" if f_rising_count > 0 else "falling"
                print(f"  [COP] F={f_val:.1f} ({trend})")

            # Dream tracking
            if DREAM_START_RE.search(line):
                in_dream = True
                f_before_dream = f_thermo_history[-1] if f_thermo_history else None
                print("  [DREAM] Dream cycle starting...")

            if DREAM_END_RE.search(line) and in_dream:
                in_dream = False
                if f_before_dream is not None and f_thermo_history:
                    f_after = f_thermo_history[-1]
                    delta = f_after - f_before_dream
                    verdict = "productive" if delta < 0 else "destructive"
                    print(f"  [DREAM] F_before={f_before_dream:.1f} F_after={f_after:.1f} "
                          f"delta={delta:.1f} verdict={verdict}")

            # OOM detection
            if OOM_RE.search(line):
                print("  [ALERT] OOM detected — check WSL2 memory config")

            # Stuck tick detection (checked every line)
            elapsed = time.time() - last_tick_time
            if elapsed > STUCK_THRESHOLD:
                print(f"  [ALERT] No tick for {elapsed:.0f}s — possible stuck")
                last_tick_time = time.time()  # reset to avoid spam

    except KeyboardInterrupt:
        print("\n[tick_monitor] Stopped.")
    finally:
        proc.terminate()


if __name__ == "__main__":
    monitor()
