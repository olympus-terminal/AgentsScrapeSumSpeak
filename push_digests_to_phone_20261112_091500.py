#!/usr/bin/env python3
"""Copy digest MP3s a phone does not have to its digest folder and record every transfer.

Each verified file gets one row in a tab-separated record (default
output/phone_transfers.tsv, git-ignored so it never reaches the public repo):
when it was checked, which phone, the file, its size and SHA-256, and whether it
was copied now or was already on the phone. A file counts as on the phone only
when the phone's copy has the same size and SHA-256 as the local file.

Phones (one MTP client at a time; never run two copies at once):
Digests go to "AI Research Digests" beside Music. --organize moves digests
already in Music there (a rename on the phone, checked by object id, size and
SHA-256) and records each move.

  honor   HONOR LLY-NX1 via its GVfs mount. Its folders are never listed,
          because listing has hung gvfsd-mtp on this phone; files already in
          the record for the honor are skipped, the rest are copied with
          `cp --update=none`, so a same-named file already there is kept and
          checked instead of overwritten.
  huawei  HUAWEI EVR-L29 via its GVfs mount. The digest folder is listed;
          same-named files there are checked, missing ones are copied.

Usage:
    python push_digests_to_phone_20261112_091500.py --device huawei --dry-run
    python push_digests_to_phone_20261112_091500.py --device huawei
    python push_digests_to_phone_20261112_091500.py --device huawei --organize
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
DIGEST_FOLDER = "AI Research Digests"  # beside Music on each phone
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


def digest_dir(device: str) -> Path:
    """The digest folder beside Music; created on first use."""
    folder = music_dir(device).parent / DIGEST_FOLDER
    if not folder.is_dir():
        folder.mkdir()
    return folder


def object_id(path: Path) -> str | None:
    """The phone's MTP object id for a path (same id before and after = moved, not copied)."""
    result = subprocess.run(["gio", "info", "-a", "id::file", str(path)],
                            capture_output=True, text=True, timeout=60)
    match = re.search(r"id::file: \S+:(\d+)", result.stdout)
    return match.group(1) if match else None


def organize(args) -> int:
    """Move digests already on the phone from Music into the digest folder.

    Files are renamed on the phone, never deleted. After a move GVfs may still
    show the old Music path from its cache; that entry is the same phone object
    as the moved file, so it must not be removed. It disappears on reconnect.
    """
    music, folder = music_dir(args.device), digest_dir(args.device)
    rows = read_record(args.record)
    latest = {}
    for row in rows:
        if row["device"] == args.device and row["verified"] == "size+sha256":
            latest[row["file"]] = row
    if args.device == "honor":
        # Music is never listed on the honor: move what the record says is in Music.
        names = sorted(n for n, r in latest.items() if "/Music/" in r["phone_path"])
    else:
        names = sorted(n for n in os.listdir(music)
                       if n.startswith("ai_research_digest_") and n.endswith(".mp3"))
    print(f"{args.device}: {len(names)} digests to move {music} -> {folder}")
    problems = 0
    for name in reversed(names):
        src, dst, local = music / name, folder / name, args.source / name
        if args.dry_run:
            print(f"  would move {name}")
            continue
        before = object_id(src)
        if not dst.exists():
            result = subprocess.run(["mv", "-n", str(src), str(dst)], capture_output=True,
                                    text=True)
            if result.returncode != 0:
                print(f"  FAIL mv {name}: {result.stderr.strip()}")
                problems += 1
                continue
        after = object_id(dst)
        if local.exists():
            ok, size, digest = verify(local, dst)
        else:  # a digest no longer kept on this computer: record what the phone has
            ok, size, digest = None, dst.stat().st_size, sha256(dst)
        moved = before is not None and before == after
        verified = "size+sha256" if ok else ("no local copy" if ok is None else "MISMATCH")
        append_row(args.record, {
            "checked_at": now(), "device": args.device, "device_id": DEVICES[args.device],
            "phone_path": str(dst.relative_to(music.parent.parent)), "file": name,
            "bytes": size, "sha256": digest, "action": "moved" if moved else "move_unconfirmed",
            "verified": verified, "script": Path(__file__).name,
            "note": f"from Music; object id {before} -> {after}",
        })
        print(f"  {'OK' if ok is not False and moved else 'CHECK'} "
              f"{'moved' if moved else 'move_unconfirmed'} {name} {verified}")
        problems += not (moved and ok is not False)
    print(f"done, problems={problems}, record: {args.record}")
    return 1 if problems else 0


def verify(local: Path, remote: Path) -> tuple[bool, int, str]:
    size, digest = local.stat().st_size, sha256(local)
    try:
        ok = remote.stat().st_size == size and sha256(remote) == digest
    except OSError:
        ok = False
    return ok, size, digest


def push(args) -> int:
    music = digest_dir(args.device)
    local_files = sorted(args.source.glob("ai_research_digest_*.mp3"), reverse=True)
    latest = {}
    for row in read_record(args.record):
        if row["device"] == args.device:
            latest[row["file"]] = row
    on_device = {n: r for n, r in latest.items()
                 if r["verified"] == "size+sha256" and r["action"] != "removed_duplicate"}
    recorded = set(on_device)
    by_hash = {r["sha256"]: n for n, r in on_device.items()}
    on_phone = None if args.device == "honor" else set(os.listdir(music))
    todo = []
    for local in (f for f in local_files if f.name not in recorded):
        # A digest renamed here (e.g. _REVIEW_NEEDED dropped) is already on the phone.
        twin = by_hash.get(sha256(local))
        if twin is None:
            todo.append(local)
        elif not args.dry_run:
            append_row(args.record, {
                "checked_at": now(), "device": args.device, "device_id": DEVICES[args.device],
                "phone_path": on_device[twin]["phone_path"], "file": local.name,
                "bytes": local.stat().st_size, "sha256": on_device[twin]["sha256"],
                "action": "same_as_phone_file", "verified": "size+sha256",
                "script": Path(__file__).name, "note": f"renamed locally; phone keeps {twin}",
            })
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
    parser.add_argument("--organize", action="store_true",
                        help=f"move digests from Music into '{DIGEST_FOLDER}'")
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
    return organize(args) if args.organize else push(args)


if __name__ == "__main__":
    sys.exit(main())
