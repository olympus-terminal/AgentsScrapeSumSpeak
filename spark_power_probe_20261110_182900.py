#!/usr/bin/env python3
"""Measure Spark GPU power while MiMo serves one recorded digest request.

Replays a request that the recording proxy logged (same messages, schema and
sampling settings) straight to the llama.cpp server through its SSH tunnel, so
the measurement does not add a digest call to the ledger. While it runs,
`nvidia-smi` on each Spark streams GB10 GPU power every 500 ms. GB10 reports
GPU power only; CPU, memory, NIC and fan power are not included.

Usage:
    python spark_power_probe_20261110_182900.py \
        --call-id 20261110_112620_xxxxxxxx \
        --hosts spark4 spark3 \
        --url http://127.0.0.1:18001/v1/chat/completions
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

SCRIPT_VERSION = "20261110_182900"
QUERY = "timestamp,power.draw,utilization.gpu,clocks.sm,temperature.gpu"


def start_sampler(host: str) -> subprocess.Popen:
    return subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", host,
         f"nvidia-smi --query-gpu={QUERY} --format=csv,noheader,nounits -lms 500"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )


def read_lines(proc: subprocess.Popen, sink: list) -> None:
    # Stamp each sample with this machine's clock so it lines up with the marks.
    for line in proc.stdout:
        sink.append((time.time(), line))


def parse_samples(lines: list) -> list[dict]:
    rows = []
    for received, line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            rows.append({"unix": round(received, 3), "t": parts[0], "power_w": float(parts[1]),
                         "util_pct": float(parts[2]), "sm_mhz": float(parts[3]),
                         "temp_c": float(parts[4])})
        except ValueError:
            continue
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--ledger", type=Path,
                        default=Path("/media/drn2/External/cluster_agents/ledger"))
    parser.add_argument("--hosts", nargs="+", default=["spark4", "spark3"])
    parser.add_argument("--url", default="http://127.0.0.1:18001/v1/chat/completions")
    parser.add_argument("--idle-seconds", type=float, default=15)
    parser.add_argument("--out", type=Path, default=Path("output/reports"))
    args = parser.parse_args()

    day = args.call_id[:8]
    record_path = args.ledger / "calls" / day / f"{args.call_id}.json.gz"
    with gzip.open(record_path, "rt") as handle:
        record = json.load(handle)
    request = record["request"]

    samplers = {host: start_sampler(host) for host in args.hosts}
    lines = {host: [] for host in args.hosts}
    for host, proc in samplers.items():
        threading.Thread(target=read_lines, args=(proc, lines[host]), daemon=True).start()
    marks = {"idle_start": time.time()}
    time.sleep(args.idle_seconds)
    marks["request_start"] = time.time()
    body = json.dumps(request).encode()
    req = urllib.request.Request(args.url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        response = json.loads(resp.read())
    marks["request_end"] = time.time()
    time.sleep(args.idle_seconds)
    marks["idle_end"] = time.time()

    samples = {}
    for host, proc in samplers.items():
        proc.terminate()
        proc.wait(timeout=10)
        samples[host] = parse_samples(lines[host])

    result = {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "replayed_call": args.call_id,
        "replayed_record": str(record_path),
        "url": args.url,
        "marks_unix": marks,
        "wall_s": round(marks["request_end"] - marks["request_start"], 2),
        "usage": response.get("usage"),
        "timings": response.get("timings"),
        "system_fingerprint": response.get("system_fingerprint"),
        "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
        "note": "GB10 nvidia-smi power.draw is GPU power only.",
        "samples": samples,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out / f"spark_power_{stamp}.json"
    out.write_text(json.dumps(result, indent=1))
    for host, rows in samples.items():
        if rows:
            peak = max(r["power_w"] for r in rows)
            print(f"{host}: {len(rows)} samples, peak {peak:.1f} W")
    print(f"wall {result['wall_s']}s timings {result['timings']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
