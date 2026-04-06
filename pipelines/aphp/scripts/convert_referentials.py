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

# Workbooks whose first row is data, not column names. Maps file stem to the
# desired column names (used by the upstream `pandas.read_excel(header=None,
# names=...)` calls in recode-scenario's utils_v2.py).
HEADERLESS = {
    "Affections chroniques": ["code", "chronic", "libelle"],
}

# Pipe-delimited latin-1 text files that should be transcoded to a UTF-8
# Parquet (polars' scan_csv only supports UTF-8). Maps relative path to the
# column names of the resulting Parquet.
LATIN1_PIPE_TO_PARQUET: dict[str, list[str]] = {
    "CIM_ATIH_2025/LIBCIM10MULTI.TXT": [
        "icd_code",
        "aut_mco",
        "pos",
        "aut_ssr",
        "icd_code_description_short",
        "icd_code_description",
    ],
}


def convert_excel(src: Path, dst_dir: Path) -> Path:
    """Read an Excel workbook and dump every sheet as a Parquet file.

    Single-sheet workbooks become ``<stem>.parquet``; multi-sheet workbooks
    become ``<stem>__<sheet>.parquet`` so nothing collides.
    """
    if src.stem in HEADERLESS:
        sheets = pl.read_excel(
            src,
            sheet_id=0,
            engine="calamine",
            read_options={"header_row": None, "column_names": HEADERLESS[src.stem]},
        )
    else:
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
        rel_posix = rel.as_posix()

        if suffix in SKIP_SUFFIXES:
            print(f"  skip       {rel}")
            n_skipped += 1
            continue

        if rel_posix in LATIN1_PIPE_TO_PARQUET:
            cols = LATIN1_PIPE_TO_PARQUET[rel_posix]
            # polars' read_csv only supports UTF-8; transcode in memory.
            utf8_bytes = path.read_bytes().decode("latin-1").encode("utf-8")
            df = pl.read_csv(
                utf8_bytes,
                separator="|",
                has_header=False,
                new_columns=cols,
            ).with_columns(pl.col("icd_code").str.strip_chars())
            out = target_dir / f"{path.stem}.parquet"
            df.write_parquet(out)
            print(f"  parquet ←  {rel}  →  {out.relative_to(dst)}")
            n_converted += 1
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
