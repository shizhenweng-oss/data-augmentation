"""Stage example layouts from Real_Data/ into data/layouts/ for the pipeline.

By default, copies the battery_cells mask (the variable element to be augmented).
Edit SOURCE_MASKS below to include other masks as additional examples.
"""

from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL_DATA = HERE.parent / "Real_Data" / "Type_C" / "masks"
DST_DIR = HERE / "data" / "layouts"

# Masks to stage as augmentation example(s).
# battery_cells is the typical augmentation target (cell placement varies).
# Add e.g. "design_space" if you want the model to see additional constraint masks.
SOURCE_MASKS = ["battery_cells"]


def main() -> None:
    missing = [m for m in SOURCE_MASKS if not (REAL_DATA / f"{m}.npy").exists()]
    if missing:
        raise SystemExit(f"Missing source masks in {REAL_DATA}: {missing}")

    DST_DIR.mkdir(parents=True, exist_ok=True)

    for name in SOURCE_MASKS:
        src = REAL_DATA / f"{name}.npy"
        dst = DST_DIR / f"{name}.npy"
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")

    print(f"\nDone. Add more .npy layouts to {DST_DIR} as needed.")


if __name__ == "__main__":
    main()
