#!/usr/bin/env python3
"""Generate small correctness-test graphs with deliberately awkward shapes.

Each one targets a specific way the GPU code could go wrong:
  path_ordered / path_shuffled  very deep trees (Multi-Jump must flatten them)
  star_low / star_high          one hub of degree 100k with the hub at the lowest /
                                highest id (forward degree 100k vs 0) -> block bucket,
                                heavy CAS contention on one root
  isolated                      mostly degree-0 vertices
  cliques                       many dense components -> warp / block buckets
  messy_onedir                  edges stored one direction only + duplicates + self-loops
                                (written raw, NOT cleaned, to test the loader)
  grid                          2D mesh, large diameter, uniform degree
  er_random                     Erdos-Renyi style uniform random graph
  powerlaw                      Chung-Lu power-law graph (hubs + long tail)
  single_vertex / two_vertices  edge cases
Usage: python3 tools/gen_tests.py data/tests
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from egr import csr_from_edges, write_egr  # noqa: E402

rng = np.random.default_rng(12345)


def save(outdir, name, n, u, v, clean=True):
    path = os.path.join(outdir, name + ".egr")
    if clean:
        nindex, nlist = csr_from_edges(n, u, v)
    else:  # raw: one direction, duplicates and self-loops kept
        u = np.asarray(u, dtype=np.int64)
        v = np.asarray(v, dtype=np.int64)
        order = np.argsort(u, kind="stable")
        u, v = u[order], v[order]
        nindex = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(u, minlength=n), out=nindex[1:])
        nlist = v
    write_egr(path, n, nindex, nlist)
    print(f"{name:16s} n={n:>8,}  stored entries={len(nlist):>9,}")


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "data/tests"
    os.makedirs(outdir, exist_ok=True)

    n = 200_000
    i = np.arange(n - 1)
    save(outdir, "path_ordered", n, i, i + 1)
    perm = rng.permutation(n)
    save(outdir, "path_shuffled", n, perm[i], perm[i + 1])

    n = 100_001
    leaves = np.arange(1, n)
    save(outdir, "star_low", n, np.zeros(n - 1, dtype=np.int64), leaves)
    save(outdir, "star_high", n, np.full(n - 1, n - 1), np.arange(n - 1))

    n = 50_000
    save(outdir, "isolated", n, rng.integers(0, n, 1000), rng.integers(0, n, 1000))

    k, size = 200, 40
    us, vs = [], []
    for c in range(k):
        base = c * size
        a, b = np.triu_indices(size, 1)
        us.append(a + base); vs.append(b + base)
    perm = rng.permutation(k * size)  # scatter clique members across the id space
    save(outdir, "cliques", k * size, perm[np.concatenate(us)], perm[np.concatenate(vs)])

    n = 20_000
    u = rng.integers(0, n, 100_000); v = rng.integers(0, n, 100_000)
    u = np.concatenate([u, u[:5000], np.arange(500)])      # duplicates + self-loops
    v = np.concatenate([v, v[:5000], np.arange(500)])
    save(outdir, "messy_onedir", n, u, v, clean=False)

    w = 300
    idx = np.arange(w * w).reshape(w, w)
    u = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    v = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    save(outdir, "grid", w * w, u, v)

    n, m = 200_000, 600_000
    save(outdir, "er_random", n, rng.integers(0, n, m), rng.integers(0, n, m))

    n, m = 100_000, 1_000_000
    wts = 1.0 / np.arange(1, n + 1) ** 0.8
    wts /= wts.sum()
    perm = rng.permutation(n)
    save(outdir, "powerlaw", n, perm[rng.choice(n, m, p=wts)], perm[rng.choice(n, m, p=wts)])

    save(outdir, "single_vertex", 1, [], [])
    save(outdir, "two_vertices", 2, [0], [1])


if __name__ == "__main__":
    main()
