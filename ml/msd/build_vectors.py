from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ml.msd.featurize import FEATURE_DIM, featurize
from ml.msd.read import read_track


LOGGER = logging.getLogger(__name__)
PROGRESS_EVERY = 10_000


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MSD id list and feature vectors from .h5 files.")
    parser.add_argument("--msd_root", required=True, help="Root directory containing MSD .h5 files.")
    parser.add_argument("--out_ids", required=True, help="Output JSON file for track ids.")
    parser.add_argument("--out_vecs", required=True, help="Output NPY file for feature vectors.")
    return parser.parse_args(argv)


def _iter_h5_files(msd_root: Path) -> list[Path]:
    return sorted(msd_root.rglob("*.h5"))


def _prepare_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)

    msd_root = Path(args.msd_root).expanduser().resolve()
    out_ids = Path(args.out_ids).expanduser().resolve()
    out_vecs = Path(args.out_vecs).expanduser().resolve()

    if not msd_root.exists() or not msd_root.is_dir():
        LOGGER.error("MSD root does not exist or is not a directory: %s", msd_root)
        return 1

    _prepare_output_path(out_ids)
    _prepare_output_path(out_vecs)

    h5_files = _iter_h5_files(msd_root)
    LOGGER.info("Discovered %d .h5 files under %s", len(h5_files), msd_root)

    seen_track_ids: set[str] = set()
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    skipped = 0
    duplicates = 0

    for idx, h5_path in enumerate(h5_files, start=1):
        try:
            track = read_track(h5_path)
            track_id = str(track["track_id"])
            if not track_id:
                raise ValueError("empty track_id")

            if track_id in seen_track_ids:
                duplicates += 1
                LOGGER.warning("Skipping duplicate track_id from %s: %s", h5_path, track_id)
                continue

            vec = featurize(track)
            if vec.shape != (FEATURE_DIM,):
                raise ValueError(f"unexpected vector shape {vec.shape}, expected ({FEATURE_DIM},)")

            seen_track_ids.add(track_id)
            ids.append(track_id)
            vectors.append(np.asarray(vec, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            LOGGER.warning("Skipping unreadable/bad file %s: %s", h5_path, exc)

        if idx % PROGRESS_EVERY == 0:
            LOGGER.info(
                "Processed %d files (kept=%d skipped=%d duplicates=%d)",
                idx,
                len(ids),
                skipped,
                duplicates,
            )

    if not ids:
        LOGGER.error("No valid feature vectors were produced.")
        return 1

    matrix = np.stack(vectors).astype(np.float32, copy=False)
    with out_ids.open("w", encoding="utf-8") as f:
        json.dump(ids, f)
    with out_vecs.open("wb") as f:
        np.save(f, matrix)

    LOGGER.info(
        "Finished: scanned=%d kept=%d skipped=%d duplicates=%d out_ids=%s out_vecs=%s",
        len(h5_files),
        len(ids),
        skipped,
        duplicates,
        out_ids,
        out_vecs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
