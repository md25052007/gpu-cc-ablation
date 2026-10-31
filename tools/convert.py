#!/usr/bin/env python3
"""Convert a downloaded graph to .egr (symmetric, no self-loops, no duplicates).

Supported inputs (optionally .gz):
  *.mtx          Matrix Market coordinate (SuiteSparse). 1-based ids, kept as-is.
  *.txt / *.el   SNAP-style edge list ("u v" per line, '#' comments).
                 Ids are relabelled to 0..n-1 (SNAP ids are often not contiguous).

Usage:
  python3 tools/convert.py data/raw/road_usa/road_usa.mtx data/road_usa.egr
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from egr import csr_from_edges, write_egr  # noqa: E402


def read_pairs(path, comment, skiprows):
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=r"\s+", comment=comment, header=None, usecols=[0, 1],
                         skiprows=skiprows, dtype=np.int64, engine="c")
        return df[0].to_numpy(), df[1].to_numpy()
    except ImportError:
        a = np.loadtxt(path, comments=comment, usecols=(0, 1), dtype=np.int64, skiprows=skiprows)
        return a[:, 0], a[:, 1]


def open_text(path):
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt")
    return open(path, "r")


def load_mtx(path):
    with open_text(path) as f:
        banner = f.readline()
        if not banner.startswith("%%MatrixMarket"):
            raise ValueError("not a Matrix Market file")
        if "coordinate" not in banner.lower().split():
            raise ValueError("only coordinate (sparse) Matrix Market files are supported")
        for line in f:
            if line.startswith("%") or not line.strip():
                continue
            rows, cols, nnz = map(int, line.split()[:3])
            break
    if rows != cols:
        raise ValueError(f"matrix is {rows}x{cols}; a graph needs a square matrix")
    # Every header line except the size line starts with '%', so reading with
    # comment='%' and NO skiprows yields: size line, then the entries. (pandas
    # versions disagree on how skiprows interacts with comment lines, which
    # silently dropped a few edges in an earlier version of this script.)
    u, v = read_pairs(path, "%", 0)
    if len(u) == 0 or (u[0], v[0]) != (rows, cols):
        raise ValueError("could not locate the size line")
    u, v = u[1:], v[1:]
    if len(u) != nnz:
        raise ValueError(f"header says {nnz} entries but read {len(u)} — file is damaged or truncated")
    return rows, u - 1, v - 1


def load_snap(path):
    u, v = read_pairs(path, "#", 0)
    ids, inv = np.unique(np.concatenate([u, v]), return_inverse=True)
    n = len(ids)
    return n, inv[: len(u)], inv[len(u):]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    args = ap.parse_args()

    t = time.time()
    name = args.input.lower().removesuffix(".gz")
    if name.endswith(".mtx"):
        n, u, v = load_mtx(args.input)
    else:
        n, u, v = load_snap(args.input)
    print(f"read {len(u):,} entries, n={n:,} ({time.time() - t:.1f}s)")
    if len(u) and (u.min() < 0 or v.min() < 0 or u.max() >= n or v.max() >= n):
        raise ValueError("vertex id out of range")

    t = time.time()
    nindex, nlist = csr_from_edges(n, u, v, symmetrize=True, dedupe=True)
    del u, v
    write_egr(args.output, n, nindex, nlist)
    print(f"wrote {args.output}: |V|={n:,} undirected |E|={len(nlist) // 2:,} "
          f"avg degree={len(nlist) / n:.2f} ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
