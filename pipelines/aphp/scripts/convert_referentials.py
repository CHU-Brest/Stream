"""One-shot conversion of recode-scenario referentials into Stream format.

Reads the original referentials/ directory from the AP-HP recode-scenario repo
and produces a Stream-friendly copy under pipelines/aphp/referentials/:

- ``.xlsx`` / ``.xls``  →  ``.parquet`` (polars-native, smaller, faster to load)
- ``.csv`` / ``.txt`` / ``.tsv`` / ``.yml`` / no-extension  →  copied verbatim
- ``.jpg`` / ``.pdf``  →  skipped (documentation, not data)

Usage::

    python pipelines/aphp/scripts/convert_referentials.py \\
        --src ~/repo/recode-scenario/referentials \\
        --dst pipelines/aphp/referentials

Requires ``python-calamine`` (Excel reader). Installed only for this one-shot
conversion — not a runtime dependency of Stream.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import polars as pl

SKIP_SUFFIXES = {".jpg", ".jpeg", ".png", ".pdf"}
EXCEL_SUFFIXES = {".xlsx", ".xls"}


def convert_excel(src: Path, dst_dir: Path) -> Path:
    """Read an Excel workbook and dump every sheet as a Parquet file.

    Single-sheet workbooks become ``<stem>.parquet``; multi-sheet workbooks
    become ``<stem>__<sheet>.parquet`` so nothing collides.
    """
    sheets = pl.read_excel(src, sheet_id=0, engine="calamine")
    if not isinstance(sheets, dict):
        sheets = {src.stem: sheets}

    written: list[Path] = []
    for sheet_name, df in sheets.items():
        if len(sheets) == 1:
            out = dst_dir / f"{src.stem}.parquet"
        else:
            safe = sheet_name.replace("/", "_").replace(" ", "_")
            out = dst_dir / f"{src.stem}__{safe}.parquet"
        df.write_parquet(out)
        written.append(out)
    return written[0] if len(written) == 1 else dst_dir / src.stem


def copy_tree(src: Path, dst: Path) -> tuple[int, int, int]:
    """Walk *src*, mirror it under *dst*, converting Excel files on the fly.

    Returns ``(n_copied, n_converted, n_skipped)``.
    """
    n_copied = n_converted = n_skipped = 0

    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue

        rel = path.relative_to(src)
        target_dir = dst / rel.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        suffix = path.suffix.lower()

        if suffix in SKIP_SUFFIXES:
            print(f"  skip       {rel}")
            n_skipped += 1
            continue

        if suffix in EXCEL_SUFFIXES:
            try:
                out = convert_excel(path, target_dir)
                print(f"  parquet ←  {rel}  →  {out.relative_to(dst)}")
                n_converted += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED     {rel}: {exc}")
                n_skipped += 1
            continue

        shutil.copy2(path, target_dir / path.name)
        print(f"  copy       {rel}")
        n_copied += 1

    return n_copied, n_converted, n_skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="recode-scenario referentials/ directory")
    parser.add_argument("--dst", type=Path, required=True, help="destination directory inside Stream")
    args = parser.parse_args()

    src: Path = args.src.expanduser().resolve()
    dst: Path = args.dst.expanduser().resolve()

    if not src.is_dir():
        raise SystemExit(f"source directory not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)
    print(f"Converting {src} → {dst}")
    n_copied, n_converted, n_skipped = copy_tree(src, dst)
    print(
        f"\nDone: {n_copied} copied, {n_converted} converted to parquet, "
        f"{n_skipped} skipped."
    )


if __name__ == "__main__":
    main()
