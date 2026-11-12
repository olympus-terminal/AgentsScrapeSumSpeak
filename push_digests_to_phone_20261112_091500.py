#!/usr/bin/env python3
"""Copy digest MP3s a phone does not have to its Music folder and record every transfer.

Each verified file gets one row in a tab-separated record (default
output/phone_transfers.tsv, git-ignored so it never reaches the public repo):
when it was checked, which phone, the file, its size and SHA-256, and whether it
was copied now or was already on the phone. A file counts as on the phone only
when the phone's copy has the same size and SHA-256 as the local file.

Phones (one MTP client at a time; never run two copies at once):
  honor   HONOR LLY-NX1 via its GVfs mount. Its Music folder is never listed,
          because listing has hung gvfsd-mtp on this phone; files already in
          the record for the honor are skipped, the rest are copied with
          `cp --update=none`, so a same-named file already there is kept and
          checked instead of overwritten.
  huawei  HUAWEI EVR-L29 via its GVfs mount. Music is listed; same-named files
          there are checked, missing ones are copied.

Usage:
    python push_digests_to_phone_20261112_091500.py --device huawei --dry-run
    python push_digests_to_phone_20261112_091500.py --device huawei
    python push_digests_to_phone_20261112_091500.py --backfill-log LOG --device honor
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_VERSION = "20261112_091500"
# GVfs MTP host ids ("<MODEL>_<SERIAL>") contain each phone's serial number, so they come from
# the git-ignored .env (PHONE_MTP_HONOR, PHONE_MTP_HUAWEI), never from this public file.
load_dotenv(Path(__file__).parent / ".env")
DEVICES = {name: os.environ.get(f"PHONE_MTP_{name.upper()}", "") for name in ("honor", "huawei")}
STORAGE_ROOTS = ("Internal storage", "Internal shared storage", "Phone")
FIELDS = ["checked_at", "device", "device_id", "phone_path", "file", "bytes",
          "sha256", "action", "verified", "script", "note"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def read_record(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, delimiter="\t")
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})


def music_dir(device: str) -> Path:
    if not DEVICES[device]:
        sys.exit(f"ERROR: set PHONE_MTP_{device.upper()} in .env (the GVfs mtp:host id of the {device}).")
    mount = Path(f"/run/user/{os.getuid()}/gvfs/mtp:host={DEVICES[device]}")
    if not mount.is_dir():
        sys.exit(f"ERROR: {device} is not mounted at {mount}. Plug it in, unlock it and "
                 "select File Transfer.")
    if device == "honor":
        return mount / "Internal storage" / "Music"  # known path; do not probe
    for root in STORAGE_ROOTS:
        if (mount / root / "Music").is_dir():
            return mount / root / "Music"
    sys.exit(f"ERROR: no Music folder under {mount} ({', '.join(STORAGE_ROOTS)}).")


def verify(local: Path, remote: Path) -> tuple[bool, int, str]:
    size, digest = local.stat().st_size, sha256(local)
    try:
        ok = remote.stat().st_size == size and sha256(remote) == digest
    except OSError:
        ok = False
    return ok, size, digest


def push(args) -> int:
    music = music_dir(args.device)
    local_files = sorted(args.source.glob("ai_research_digest_*.mp3"), reverse=True)
    recorded = {r["file"] for r in read_record(args.record)
                if r["device"] == args.device and r["verified"] == "size+sha256"}
    on_phone = None if args.device == "honor" else set(os.listdir(music))
    todo = [f for f in local_files if f.name not in recorded]
    print(f"{args.device}: {len(local_files)} local digests, {len(recorded)} already recorded, "
          f"{len(todo)} to check/copy -> {music}")
    problems = 0
    for local in todo:
        remote = music / local.name
        present = on_phone is not None and local.name in on_phone
        if args.dry_run:
            print(f"  would {'check' if present else 'copy'} {local.name}")
            continue
        if not present:
            # --update=none keeps a same-named file already on the phone; verify decides.
            result = subprocess.run(["cp", "-v", "--update=none", str(local), str(remote)],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  FAIL cp {local.name}: {result.stderr.strip()}")
                problems += 1
                continue
            action = "copied" if result.stdout.strip() else "already_present"
        else:
            action = "already_present"
        ok, size, digest = verify(local, remote)
        append_row(args.record, {
            "checked_at": now(), "device": args.device, "device_id": DEVICES[args.device],
            "phone_path": str(remote.relative_to(music.parent.parent)), "file": local.name,
            "bytes": size, "sha256": digest, "action": action,
            "verified": "size+sha256" if ok else "MISMATCH", "script": Path(__file__).name,
        })
        print(f"  {'OK' if ok else 'MISMATCH'} {action} {local.name} {size}")
        problems += not ok
    print(f"done, problems={problems}, record: {args.record}")
    return 1 if problems else 0


LOG_LINE = re.compile(r"^(\d\d:\d\d:\d\d) OK (copied|existing) (\S+\.mp3) (\d+)$")


def backfill(args) -> int:
    """Record a finished push from its timestamped OK lines (each verified size+SHA-256)."""
    recorded = {r["file"] for r in read_record(args.record) if r["device"] == args.device}
    added = 0
    for line in args.backfill_log.read_text().splitlines():
        match = LOG_LINE.match(line.strip())
        if not match or match.group(3) in recorded:
            continue
        clock, status, name, size = match.groups()
        local = args.source / name
        if local.stat().st_size != int(size):
            sys.exit(f"ERROR: {name} size changed since the push")
        note = args.note.get(name, "")
        append_row(args.record, {
            "checked_at": f"{args.log_date}T{clock}+04:00", "device": args.device,
            "device_id": DEVICES[args.device], "phone_path": f"Internal storage/Music/{name}",
            "file": name, "bytes": size, "sha256": sha256(local),
            "action": "copied" if status == "copied" else "already_present",
            "verified": "size+sha256", "script": f"backfill from {args.backfill_log.name}",
            "note": note,
        })
        added += 1
    print(f"backfilled {added} rows -> {args.record}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", choices=sorted(DEVICES), required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--record", type=Path,
                        default=Path(__file__).resolve().parent / "output" / "phone_transfers.tsv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backfill-log", type=Path,
                        help="record an earlier push from lines 'HH:MM:SS OK copied|existing FILE BYTES'")
    parser.add_argument("--log-date", help="date of the backfilled push, YYYY-MM-DD")
    parser.add_argument("--note", action="append", default=[], metavar="FILE=TEXT",
                        help="note for one backfilled file")
    args = parser.parse_args()
    args.note = dict(n.split("=", 1) for n in args.note)
    if args.backfill_log:
        if not args.log_date:
            parser.error("--backfill-log needs --log-date")
        return backfill(args)
    return push(args)


if __name__ == "__main__":
    sys.exit(main())
