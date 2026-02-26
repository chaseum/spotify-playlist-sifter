from __future__ import annotations

import argparse
import logging
from pathlib import Path

import faiss
import numpy as np

from ml.msd.featurize import featurize
from ml.msd.read import read_track


LOGGER = logging.getLogger(__name__)
DEFAULT_INDEX_PATH = Path("data/msd.faiss")


def l2_normalize(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm > 0.0:
            out = arr / norm
        else:
            out = np.zeros_like(arr, dtype=np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    if arr.ndim == 2:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        out = arr / safe_norms
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    raise ValueError(f"Expected 1D or 2D array, got ndim={arr.ndim}")


def build(vecs_path: str | Path, out_path: str | Path) -> int:
    vecs = Path(vecs_path).expanduser().resolve()
    out = Path(out_path).expanduser().resolve()

    if not vecs.exists():
        LOGGER.error("Vector file does not exist: %s", vecs)
        return 1

    try:
        matrix = np.load(vecs)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to load vectors from %s: %s", vecs, exc)
        return 1

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        LOGGER.error("Expected non-empty 2D matrix, got shape=%s", matrix.shape)
        return 1

    normalized = l2_normalize(matrix)
    dim = int(normalized.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(normalized)

    out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))
    LOGGER.info("Wrote FAISS index: path=%s dim=%d ntotal=%d", out, dim, index.ntotal)
    return 0


def query_vec(vec: np.ndarray, index: faiss.Index, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if index.ntotal == 0:
        raise ValueError("index is empty")

    query = np.asarray(vec, dtype=np.float32).reshape(-1)
    search_k = min(int(k), int(index.ntotal))
    normalized_query = l2_normalize(query).reshape(1, -1)
    scores, row_ids = index.search(normalized_query, search_k)
    return row_ids[0], scores[0]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query a cosine FAISS index for MSD vectors.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build FAISS index from msd_vecs.npy.")
    build_parser.add_argument("--vecs", required=True, help="Input NPY matrix (N, D).")
    build_parser.add_argument("--out", required=True, help="Output FAISS index path.")

    query_parser = subparsers.add_parser("query", help="Query FAISS index from an MSD .h5 track.")
    query_parser.add_argument("--h5", required=True, help="Path to MSD .h5 file.")
    query_parser.add_argument("--k", type=int, default=5, help="Number of neighbors to return.")
    query_parser.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_PATH),
        help=f"Path to FAISS index (default: {DEFAULT_INDEX_PATH.as_posix()}).",
    )

    return parser.parse_args(argv)


def _run_query(h5_path: str | Path, index_path: str | Path, k: int) -> int:
    resolved_index_path = Path(index_path).expanduser().resolve()
    if not resolved_index_path.exists():
        LOGGER.error("Index file does not exist: %s", resolved_index_path)
        return 1

    try:
        index = faiss.read_index(str(resolved_index_path))
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to load index from %s: %s", resolved_index_path, exc)
        return 1

    try:
        track = read_track(h5_path)
        vec = featurize(track)
        row_ids, scores = query_vec(vec, index, k)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Query failed for h5=%s: %s", h5_path, exc)
        return 1

    for rank, (row_id, score) in enumerate(zip(row_ids, scores), start=1):
        print(f"rank={rank} row_id={int(row_id)} score={float(score):.6f}")

    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)

    if args.command == "build":
        return build(args.vecs, args.out)
    if args.command == "query":
        return _run_query(args.h5, args.index, args.k)

    LOGGER.error("Unknown command: %s", args.command)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
