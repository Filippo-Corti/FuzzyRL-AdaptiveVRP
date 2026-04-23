from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path


CHECKPOINT_PATTERN = re.compile(r"^(?P<prefix>fuzzy|transformer)-(?P<body>[0-9-]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize checkpoint names and consolidate metrics files in a checkpoints directory."
        )
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Path to checkpoints folder (default: checkpoints).",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help=(
            "Optional destination for full backup copy. "
            "Default: checkpoints_backup_YYYYMMDD_HHMMSS next to checkpoints."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned operations without changing files.",
    )
    return parser.parse_args()


def ensure_backup(checkpoints_dir: Path, backup_dir: Path | None, dry_run: bool) -> Path:
    if backup_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = checkpoints_dir.parent / f"{checkpoints_dir.name}_backup_{ts}"

    if backup_dir.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_dir}")

    print(f"[backup] {checkpoints_dir} -> {backup_dir}")
    if not dry_run:
        shutil.copytree(checkpoints_dir, backup_dir)

    return backup_dir


def extract_last_step(stem: str) -> tuple[str, int] | None:
    m = CHECKPOINT_PATTERN.match(stem)
    if m is None:
        return None

    prefix = m.group("prefix")
    numbers = [int(x) for x in m.group("body").split("-") if x]
    if not numbers:
        return None

    return prefix, numbers[-1]


def normalize_checkpoints(checkpoints_dir: Path, dry_run: bool) -> None:
    candidates = [
        p
        for p in checkpoints_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".pt", ".pkl"}
    ]

    keep_for_target: dict[Path, Path] = {}
    for src in candidates:
        parsed = extract_last_step(src.stem)
        if parsed is None:
            continue
        prefix, step = parsed
        target = checkpoints_dir / f"{prefix}-{step}{src.suffix.lower()}"

        current = keep_for_target.get(target)
        if current is None:
            keep_for_target[target] = src
            continue

        # Prefer canonical short names if present; otherwise keep newest by mtime.
        if current.name == target.name:
            continue
        if src.name == target.name:
            keep_for_target[target] = src
            continue
        if src.stat().st_mtime > current.stat().st_mtime:
            keep_for_target[target] = src

    to_delete: list[Path] = []
    moved = 0

    for target, chosen_src in keep_for_target.items():
        if chosen_src == target:
            continue

        if target.exists() and target != chosen_src:
            print(f"[delete] duplicate target will be replaced: {target.name}")
            to_delete.append(target)

        print(f"[rename] {chosen_src.name} -> {target.name}")
        if not dry_run:
            if target.exists() and target != chosen_src:
                target.unlink()
            chosen_src.rename(target)
        moved += 1

    name_to_path: dict[str, Path] = {
        p.name: p for p in checkpoints_dir.iterdir() if p.is_file()
    }

    for src in candidates:
        if src.name not in name_to_path:
            continue
        current_path = name_to_path[src.name]
        parsed = extract_last_step(current_path.stem)
        if parsed is None:
            continue
        prefix, step = parsed
        canonical_name = f"{prefix}-{step}{current_path.suffix.lower()}"
        if current_path.name != canonical_name:
            print(f"[delete] redundant alias: {current_path.name}")
            to_delete.append(current_path)

    unique_delete = []
    seen = set()
    for p in to_delete:
        if p not in seen and p.exists():
            unique_delete.append(p)
            seen.add(p)

    for p in unique_delete:
        if not dry_run:
            p.unlink()

    print(f"[summary] normalized checkpoint names; renamed {moved} files")
    if unique_delete:
        print(f"[summary] removed {len(unique_delete)} duplicate aliases")


def read_metrics_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
        return reader.fieldnames, rows


def consolidate_metrics(checkpoints_dir: Path, prefix: str, dry_run: bool) -> None:
    metrics_files = sorted(checkpoints_dir.glob(f"{prefix}*-metrics.csv"))
    if not metrics_files:
        return

    headers: list[str] | None = None
    by_episode: dict[int, dict[str, str]] = {}

    for path in metrics_files:
        current_headers, rows = read_metrics_rows(path)
        if headers is None:
            headers = current_headers
        elif current_headers != headers:
            raise ValueError(
                f"Metrics schema mismatch in {path.name}; expected {headers}, got {current_headers}"
            )

        for row in rows:
            episode_raw = (row.get("episode") or "").strip()
            if episode_raw == "":
                continue
            episode = int(float(episode_raw))
            by_episode[episode] = row

    if headers is None:
        return

    output_path = checkpoints_dir / f"{prefix}-metrics.csv"
    print(
        f"[metrics] merge {len(metrics_files)} files -> {output_path.name} "
        f"({len(by_episode)} episodes)"
    )

    if not dry_run:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for episode in sorted(by_episode):
                writer.writerow(by_episode[episode])

        for path in metrics_files:
            if path == output_path:
                continue
            path.unlink()


def main() -> None:
    args = parse_args()
    checkpoints_dir = args.checkpoints_dir.resolve()

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    if not checkpoints_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {checkpoints_dir}")

    backup_dir = ensure_backup(checkpoints_dir, args.backup_dir, args.dry_run)
    normalize_checkpoints(checkpoints_dir, args.dry_run)
    consolidate_metrics(checkpoints_dir, "fuzzy", args.dry_run)
    consolidate_metrics(checkpoints_dir, "transformer", args.dry_run)

    print(f"[done] backup at: {backup_dir}")


if __name__ == "__main__":
    main()
