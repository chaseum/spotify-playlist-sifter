from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

from ml.msd.read import read_track


LOGGER = logging.getLogger(__name__)
PROGRESS_EVERY = 10_000


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MSD metadata SQLite database from .h5 files.")
    parser.add_argument("--msd_root", required=True, help="Root directory containing MSD .h5 files.")
    parser.add_argument("--out", required=True, help="Output SQLite path.")
    return parser.parse_args(argv)


def _iter_h5_files(msd_root: Path) -> list[Path]:
    return sorted(msd_root.rglob("*.h5"))


def _prepare_database(out_path: Path) -> sqlite3.Connection:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn: sqlite3.Connection | None = None
    if out_path.exists():
        try:
            conn = sqlite3.connect(out_path)
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("DROP INDEX IF EXISTS idx_msd_meta_title_artist")
            conn.execute("DROP INDEX IF EXISTS idx_msd_meta_artist")
            conn.execute("DROP TABLE IF EXISTS msd_meta")
        except sqlite3.DatabaseError:
            if conn is not None:
                conn.close()
            out_path.unlink()
            conn = sqlite3.connect(out_path)
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
    else:
        conn = sqlite3.connect(out_path)
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")

    conn.execute(
        """
        CREATE TABLE msd_meta (
            track_id TEXT PRIMARY KEY,
            title TEXT,
            artist TEXT,
            duration REAL,
            year INTEGER
        )
        """
    )
    conn.execute("CREATE INDEX idx_msd_meta_title_artist ON msd_meta(title, artist)")
    conn.execute("CREATE INDEX idx_msd_meta_artist ON msd_meta(artist)")
    return conn


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)

    msd_root = Path(args.msd_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not msd_root.exists() or not msd_root.is_dir():
        LOGGER.error("MSD root does not exist or is not a directory: %s", msd_root)
        return 1

    h5_files = _iter_h5_files(msd_root)
    LOGGER.info("Discovered %d .h5 files under %s", len(h5_files), msd_root)

    inserted = 0
    skipped = 0
    duplicates = 0

    conn = _prepare_database(out_path)
    try:
        for idx, h5_path in enumerate(h5_files, start=1):
            try:
                track = read_track(h5_path)
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO msd_meta(track_id, title, artist, duration, year)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (
                        str(track["track_id"]),
                        str(track["title"]),
                        str(track["artist_name"]),
                        float(track["duration"]),
                        int(track["year"]),
                    ),
                )
                if cursor.rowcount == 1:
                    inserted += 1
                else:
                    duplicates += 1
                    LOGGER.warning("Skipping duplicate track_id from %s: %s", h5_path, track.get("track_id"))
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                LOGGER.warning("Skipping unreadable/bad file %s: %s", h5_path, exc)

            if idx % PROGRESS_EVERY == 0:
                LOGGER.info(
                    "Processed %d files (inserted=%d skipped=%d duplicates=%d)",
                    idx,
                    inserted,
                    skipped,
                    duplicates,
                )

        conn.commit()
    finally:
        conn.close()

    LOGGER.info(
        "Finished: scanned=%d inserted=%d skipped=%d duplicates=%d out=%s",
        len(h5_files),
        inserted,
        skipped,
        duplicates,
        out_path,
    )
    if inserted == 0:
        LOGGER.error("No valid tracks were inserted.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
