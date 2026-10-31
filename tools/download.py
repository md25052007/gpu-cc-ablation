#!/usr/bin/env python3
"""Download the benchmark graphs and convert them to .egr.

Sources (tried in this order for each graph):
  1. a copy you already downloaded yourself (data/raw/ or your Downloads folder)
  2. SNAP graphs: straight from snap.stanford.edu (edge-list .txt.gz)
  3. SuiteSparse graphs: the University of Florida mirror of the collection, then
     sparse.tamu.edu and its herokuapp host (Matrix Market .tar.gz)
If every source fails, the script prints the page to download from by hand.

Usage:
  python3 tools/download.py                 # the 7 core graphs
  python3 tools/download.py --all           # core + optional
  python3 tools/download.py web-Google rgg_n_2_21_s0
  python3 tools/download.py --list
Files land in data/raw/ (archives) and data/<name>.egr.
"""
import argparse
import os
import subprocess
import sys
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# Tried in order. The servers reject Python's default User-Agent (HTTP 403),
# so we send a browser-like one and prefer curl/wget when available.
URLS = [
    "https://www.cise.ufl.edu/research/sparse/MM/{group}/{name}.tar.gz",
    "https://sparse.tamu.edu/MM/{group}/{name}.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz",
]
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def _try(url, dest):
    import shutil
    part = dest + ".part"
    if os.path.exists(part):
        os.remove(part)
    if shutil.which("curl"):
        cmd = ["curl", "-fL", "--retry", "3", "-A", UA, "-o", part, url]
    elif shutil.which("wget"):
        cmd = ["wget", "-q", "--show-progress", "--tries=3", f"--user-agent={UA}", "-O", part, url]
    else:
        cmd = None
    try:
        if cmd:
            subprocess.run(cmd, check=True)
        else:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        with open(part, "rb") as f:          # make sure we got a gzip, not an HTML error page
            if f.read(2) != b"\x1f\x8b":
                raise ValueError("server did not return a .tar.gz")
        os.replace(part, dest)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"       failed: {e}")
        if os.path.exists(part):
            os.remove(part)
        return False


# SNAP-group graphs can come straight from SNAP (Stanford), which allows scripted downloads.
SNAP_DIRECT = {
    "web-Google": "https://snap.stanford.edu/data/web-Google.txt.gz",
    "soc-LiveJournal1": "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz",
    "com-Orkut": "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz",
}


def find_in_downloads(fname):
    """Look for a browser-downloaded file in the Windows (WSL) or Linux Downloads folder."""
    import glob
    pats = [f"/mnt/c/Users/*/Downloads/{fname}", os.path.expanduser(f"~/Downloads/{fname}")]
    for p in pats:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None


def download(group, name, dest):
    for tmpl in URLS:
        url = tmpl.format(group=group, name=name)
        print(f"[get ] {url}")
        if _try(url, dest):
            return True
    return False


def manual_hint(group, name):
    if name in SNAP_DIRECT:
        return (f"  -> download {SNAP_DIRECT[name]} in your browser, put the file in data/raw/ "
                f"(or leave it in your Downloads folder), re-run.")
    return (f"  -> open https://www.cise.ufl.edu/research/sparse/matrices/{group}/{name}.html "
            f"(or https://sparse.tamu.edu/{group}/{name}), download the Matrix Market (MM) .tar.gz, "
            f"put {name}.tar.gz in data/raw/ (or leave it in your Downloads folder), re-run.")

def _try_snap(url, dest):
    print(f"[get ] {url}")
    return _try(url, dest)


# name: (group, category, core?)
DATASETS = {
    "road_usa":          ("DIMACS10", "road",      True),
    "europe_osm":        ("DIMACS10", "road",      False),  # large: ~51M vertices
    "soc-LiveJournal1":  ("SNAP",     "social",    True),
    "com-Orkut":         ("SNAP",     "social",    True),
    "kron_g500-logn21":  ("DIMACS10", "synthetic", True),
    "rgg_n_2_21_s0":     ("DIMACS10", "random",    True),
    "web-Google":        ("SNAP",     "web",       True),
    "indochina-2004":    ("LAW",      "web",       False),
    "delaunay_n22":      ("DIMACS10", "mesh",      True),
}


def fetch(name, data_dir):
    group = DATASETS[name][0]
    out = os.path.join(data_dir, name + ".egr")
    if os.path.exists(out):
        print(f"[skip] {out} exists")
        return out
    raw = os.path.join(data_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    tgz = os.path.join(raw, name + ".tar.gz")
    if not os.path.exists(tgz):
        found = find_in_downloads(name + ".tar.gz")
        if found:
            import shutil
            print(f"[copy] {found}")
            shutil.copy(found, tgz)
    if not os.path.exists(tgz) and name in SNAP_DIRECT:
        txt = os.path.join(raw, os.path.basename(SNAP_DIRECT[name]))
        if not os.path.exists(txt):
            found = find_in_downloads(os.path.basename(txt))
            if found:
                import shutil
                print(f"[copy] {found}")
                shutil.copy(found, txt)
        if os.path.exists(txt) or _try_snap(SNAP_DIRECT[name], txt):
            print(f"[conv] {txt}")
            subprocess.run([sys.executable, os.path.join(HERE, "convert.py"), txt, out], check=True)
            return out
    if not os.path.exists(tgz) and not download(group, name, tgz):
        print(f"[FAIL] could not download {name}")
        print(manual_hint(group, name))
        return None
    mtx = os.path.join(raw, name, name + ".mtx")
    if not os.path.exists(mtx):
        print(f"[untar] {tgz}")
        with tarfile.open(tgz) as t:
            member = next(m for m in t.getmembers() if m.name.endswith(f"{name}/{name}.mtx"))
            try:
                t.extract(member, raw, filter="data")
            except TypeError:  # Python < 3.12 without the extraction-filter backport
                t.extract(member, raw)
    print(f"[conv] {mtx}")
    subprocess.run([sys.executable, os.path.join(HERE, "convert.py"), mtx, out], check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    if a.list:
        for n, (g, c, core) in DATASETS.items():
            print(f"{n:20s} {g:9s} {c:10s} {'core' if core else 'optional'}")
        return
    names = a.names or [n for n, v in DATASETS.items() if a.all or v[2]]
    failed = []
    for n in names:
        if n not in DATASETS:
            sys.exit(f"unknown dataset {n}; see --list")
        if fetch(n, a.data) is None:
            failed.append(n)
    if failed:
        print("\nStill missing (download these in the browser, then re-run the same command):")
        for n in failed:
            print(manual_hint(DATASETS[n][0], n))


if __name__ == "__main__":
    main()
