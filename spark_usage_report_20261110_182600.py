#!/usr/bin/env python3
"""Measure how much digest work the local MiMo deployment on the DGX Sparks does.

Reads the recording proxy's ledger (one JSONL summary per call plus a gzipped
full record with the request, the response and llama.cpp's timings) and writes
one JSON report of per-call rows and per-day / per-role aggregates. Nothing is
estimated here: every number is copied or summed from the ledger. Cloud-price
equivalents are computed by the report page from these token counts.

Usage:
    python spark_usage_report_20261110_182600.py \
        --ledger /media/drn2/External/cluster_agents/ledger \
        --label mimo-spark34-mxfp4 \
        --out output/reports
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import json
import platform
import re
import statistics
import sys
from pathlib import Path

SCRIPT_VERSION = "20261110_182600"


def role_of(request: dict) -> str:
    messages = request.get("messages") or []
    system = messages[0].get("content", "") if messages else ""
    if not isinstance(system, str):
        system = json.dumps(system)
    if "judge agent" in system:
        return "judge"
    if "writer agent" in system:
        return "writer"
    return "other"


def input_items(request: dict) -> int | None:
    """Number of source items sent in a judge or writer request."""
    messages = request.get("messages") or []
    if len(messages) < 2 or not isinstance(messages[-1].get("content"), str):
        return None
    try:
        payload = json.loads(messages[-1]["content"])
    except json.JSONDecodeError:
        return None
    items = payload.get("items") if isinstance(payload, dict) else None
    return len(items) if isinstance(items, list) else None


def output_stats(content: str | None, role: str) -> dict:
    stats = {"output_chars": len(content or ""), "output_words": None, "output_results": None}
    if not content:
        return stats
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return stats
    if role == "judge" and isinstance(parsed.get("results"), list):
        stats["output_results"] = len(parsed["results"])
        stats["kept"] = sum(1 for r in parsed["results"] if r.get("verdict") == "keep")
    if role == "writer" and isinstance(parsed.get("segment"), str):
        stats["output_words"] = len(parsed["segment"].split())
        stats["covered_items"] = len(parsed.get("covered_item_indices") or [])
    return stats


def error_class(error: str | None) -> str | None:
    if not error:
        return None
    return error.split(":", 1)[0].strip()[:60]


def read_calls(ledger: Path, label: str) -> list[dict]:
    rows = []
    for path in sorted(ledger.glob("ledger_*.jsonl")):
        with path.open() as handle:
            for line in handle:
                summary = json.loads(line)
                if summary.get("label") != label:
                    continue
                row = {
                    "id": summary["id"],
                    "ts": summary["ts"],
                    "day": summary["ts"][:10],
                    "path": summary["path"].split("?")[0],
                    "status": summary.get("status"),
                    "error_class": error_class(summary.get("error")),
                    "prompt_tokens": summary.get("prompt_tokens"),
                    "cached_tokens": summary.get("cached_tokens"),
                    "completion_tokens": summary.get("completion_tokens"),
                    "finish_reason": summary.get("finish_reason"),
                    "wall_s": summary.get("wall_s"),
                    "role": "preflight" if summary["path"].startswith("/v1/models") else None,
                }
                if row["path"] == "/v1/chat/completions":
                    row.update(read_full_record(Path(summary["full_record"])))
                rows.append(row)
    return rows


def read_full_record(path: Path) -> dict:
    extra = {"role": "other"}
    try:
        with gzip.open(path, "rt") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        extra["record_error"] = type(exc).__name__
        return extra
    request = record.get("request") or {}
    role = role_of(request)
    extra["role"] = role
    extra["input_items"] = input_items(request)
    extra["input_chars"] = sum(
        len(m.get("content") or "") for m in request.get("messages") or []
        if isinstance(m.get("content"), str)
    )
    try:
        response = json.loads(record.get("response_raw") or "")
    except json.JSONDecodeError:
        response = {}
    timings = response.get("timings") or {}
    for key in ("cache_n", "prompt_n", "prompt_ms", "prompt_per_second",
                "predicted_n", "predicted_ms", "predicted_per_second"):
        extra[key] = timings.get(key)
    choices = response.get("choices") or []
    content = choices[0].get("message", {}).get("content") if choices else None
    extra.update(output_stats(content, role))
    extra["system_fingerprint"] = response.get("system_fingerprint")
    return extra


def aggregate(rows: list[dict], key) -> dict:
    groups = collections.defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    out = {}
    for name, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ok = [r for r in members if r["status"] == 200]
        gen = [r for r in ok if r.get("predicted_ms")]
        out[str(name)] = {
            "calls": len(members),
            "ok": len(ok),
            "failed": len(members) - len(ok),
            "error_classes": dict(collections.Counter(
                r["error_class"] or f"HTTP {r['status']}" for r in members if r["status"] != 200
            )),
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
            "cached_tokens": sum(r["cached_tokens"] or 0 for r in ok),
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in ok),
            "wall_s_ok": round(sum(r["wall_s"] or 0 for r in ok), 1),
            "prompt_ms": round(sum(r.get("prompt_ms") or 0 for r in gen), 1),
            "predicted_ms": round(sum(r.get("predicted_ms") or 0 for r in gen), 1),
            "input_items": sum(r.get("input_items") or 0 for r in ok),
            "judge_results": sum(r.get("output_results") or 0 for r in ok),
            "judge_kept": sum(r.get("kept") or 0 for r in ok),
            "writer_words": sum(r.get("output_words") or 0 for r in ok),
        }
    return out


# Ledger labels that served MiMo-V2.6-Flash through vLLM (tp4) on Jubail; their
# responses carry system_fingerprint "vllm-0.31.0-tp4-673df2d3".
JUBAIL_FLASH_LABELS = ("mimo", "mimo-jubail-pgp-reread")


def throughput(ledger: Path, label: str) -> dict:
    """Per-call output speed and busiest-hour output for Jubail vs the Sparks."""
    per_call = collections.defaultdict(list)
    hourly = collections.defaultdict(collections.Counter)
    for path in sorted(ledger.glob("ledger_*.jsonl")):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("label") == label:
                    group = "spark"
                elif row.get("label") in JUBAIL_FLASH_LABELS:
                    group = "jubail"
                else:
                    continue
                tokens, wall = row.get("completion_tokens"), row.get("wall_s")
                if row.get("status") != 200 or not tokens or not wall:
                    continue
                hourly[group][row["ts"][:13]] += tokens
                # Short replies are dominated by prompt time; streams by client reads.
                if tokens >= 300 and not row.get("stream"):
                    per_call[group].append(tokens / wall)
    out = {}
    for group, rates in per_call.items():
        rates.sort()
        busiest = hourly[group].most_common(1)[0]
        out[group] = {
            "calls": len(rates),
            "median_tok_s": round(statistics.median(rates), 1),
            "p10_tok_s": round(rates[len(rates) // 10], 1),
            "p90_tok_s": round(rates[9 * len(rates) // 10], 1),
            "rates": [round(r, 1) for r in rates],
            "busiest_hour": busiest[0],
            "busiest_hour_tokens": busiest[1],
            "busiest_hour_tok_s": round(busiest[1] / 3600, 1),
        }
    return out


def digest_history(log_path: Path) -> list[dict]:
    """Backend, judging and script totals for each digest cycle in agent.log.

    agent.log lines carry only a time of day, so each cycle is dated from the
    script file name it reports.
    """
    backend = None
    pending: dict = {}
    cycles = []
    patterns = {
        "codex": re.compile(r"Codex preflight passed"),
        "mimo": re.compile(r"Model preflight passed: (\S+)"),
        "judged": re.compile(r"Judging complete: (\d+)/(\d+) kept"),
        "written": re.compile(r"Digest script written: \S+digest_(\d{8})_\d{6}\.txt "
                              r"\((\d+) words"),
    }
    with log_path.open(errors="replace") as handle:
        for line in handle:
            if patterns["codex"].search(line):
                backend = "codex"
            elif (m := patterns["mimo"].search(line)):
                backend = m.group(1)
            elif (m := patterns["judged"].search(line)):
                pending = {"kept": int(m.group(1)), "judged": int(m.group(2))}
            elif (m := patterns["written"].search(line)):
                cycles.append({"date": m.group(1), "backend": backend,
                               "words": int(m.group(2)), **pending})
                pending = {}
    return cycles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ledger", type=Path,
                        default=Path("/media/drn2/External/cluster_agents/ledger"))
    parser.add_argument("--label", default="mimo-spark34-mxfp4")
    parser.add_argument("--agent-log", type=Path, default=Path("agent.log"))
    parser.add_argument("--out", type=Path, default=Path("output/reports"))
    args = parser.parse_args()

    rows = read_calls(args.ledger, args.label)
    chat = [r for r in rows if r["path"] == "/v1/chat/completions"]
    report = {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "python": platform.python_version(),
        "ledger": str(args.ledger),
        "label": args.label,
        "first_call": rows[0]["ts"] if rows else None,
        "last_call": rows[-1]["ts"] if rows else None,
        "by_day": aggregate(chat, lambda r: r["day"]),
        "by_role": aggregate(chat, lambda r: r["role"]),
        "by_day_role": aggregate(chat, lambda r: f"{r['day']} {r['role']}"),
        "preflight": aggregate([r for r in rows if r["role"] == "preflight"],
                               lambda r: r["day"]),
        "throughput": throughput(args.ledger, args.label),
        "digest_history": digest_history(args.agent_log) if args.agent_log.exists() else [],
        "calls": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out / f"spark_usage_{stamp}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"{len(rows)} ledger rows ({len(chat)} chat calls) -> {out}")
    for name, agg in report["by_day_role"].items():
        print(f"{name:22s} calls={agg['calls']:4d} ok={agg['ok']:3d} "
              f"in={agg['prompt_tokens']:7d} out={agg['completion_tokens']:6d} "
              f"wall={agg['wall_s_ok']:7.0f}s items={agg['input_items']:3d} "
              f"kept={agg['judge_kept']:3d} words={agg['writer_words']:6d} "
              f"errors={agg['error_classes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
