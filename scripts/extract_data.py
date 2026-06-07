"""Extract the OptionsDX ``.7z`` archives in ``data/`` into per-symbol folders.

Each archive (e.g. ``spy_eod_2010-bndqqt.7z``) holds monthly EOD chain ``.txt``
files. We route by the ``spy_``/``qqq_`` filename prefix into
``data/raw/spy/`` and ``data/raw/qqq/`` so the existing loaders can glob them:

    python scripts/extract_data.py
    python scripts/run_real_data.py --data "data/raw/spy/spy_eod_*.txt" ...

The step is idempotent: an archive whose entries are already present in the
target folder is skipped.
"""

from __future__ import annotations

import pathlib as _pl
import sys as _sys

import py7zr

# The repo lives under a path with non-Latin characters (OneDrive "桌面"); make
# stdout UTF-8 so progress printing never hits a cp1252 encode error on Windows.
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA = _pl.Path(__file__).resolve().parents[1] / "data"
SYMBOLS = ("spy", "qqq")


def target_dir(archive_name: str) -> _pl.Path | None:
    for sym in SYMBOLS:
        if archive_name.lower().startswith(f"{sym}_"):
            return DATA / "raw" / sym
    return None


def main() -> None:
    archives = sorted(DATA.glob("*.7z"))
    if not archives:
        print(f"no .7z archives in {DATA}")
        return
    for arc in archives:
        out = target_dir(arc.name)
        if out is None:
            print(f"skip (unknown symbol): {arc.name}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(arc, "r") as z:
            names = z.getnames()
            missing = [n for n in names if not (out / _pl.Path(n).name).exists()]
            if not missing:
                print(f"skip (already extracted): {arc.name} ({len(names)} files)")
                continue
            print(f"extracting {arc.name} -> {out}/ ({len(missing)}/{len(names)} files)")
            z.extract(path=out, targets=missing)
    # Report
    for sym in SYMBOLS:
        d = DATA / "raw" / sym
        n = len(list(d.glob("*.txt"))) if d.exists() else 0
        print(f"  {sym}: {n} .txt files in {d}")


if __name__ == "__main__":
    main()
