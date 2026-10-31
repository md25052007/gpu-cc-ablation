"""Read/write ECL-CC binary CSR (.egr): int32 nodes, int32 edges,
int32 nindex[nodes+1], int32 nlist[edges]."""
import numpy as np


def write_egr(path, n, nindex, nlist):
    nindex = np.asarray(nindex, dtype=np.int64)
    nlist = np.asarray(nlist, dtype=np.int64)
    if len(nindex) != n + 1 or nindex[0] != 0 or nindex[-1] != len(nlist):
        raise ValueError("inconsistent CSR")
    if len(nlist) >= 2**31 or n >= 2**31:
        raise ValueError("graph too large for 32-bit .egr")
    with open(path, "wb") as f:
        np.array([n, len(nlist)], dtype=np.int32).tofile(f)
        nindex.astype(np.int32).tofile(f)
        nlist.astype(np.int32).tofile(f)


def read_egr(path):
    with open(path, "rb") as f:
        n, m = np.fromfile(f, dtype=np.int32, count=2)
        nindex = np.fromfile(f, dtype=np.int32, count=n + 1)
        nlist = np.fromfile(f, dtype=np.int32, count=m)
    return int(n), nindex, nlist


def csr_from_edges(n, u, v, symmetrize=True, dedupe=True):
    """Edge arrays -> CSR. Drops self-loops; optionally adds reverse edges
    and removes duplicates. Rows come out sorted."""
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    keep = u != v
    u, v = u[keep], v[keep]
    if symmetrize:
        u, v = np.concatenate([u, v]), np.concatenate([v, u])
    key = u * n + v
    del u, v
    key = np.unique(key) if dedupe else np.sort(key)
    src = key // n
    dst = key - src * n
    del key
    counts = np.bincount(src, minlength=n)
    nindex = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=nindex[1:])
    return nindex, dst
