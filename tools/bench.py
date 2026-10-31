#!/usr/bin/env python3
"""Run the full experiment over a set of .egr graphs.

Modes (default: main + counters + stats):
  main      timing build, all versions, auto segments  -> results/timing.csv
  counters  counter build, all versions, 1 run          -> results/counters.csv, results/segments.csv
  stats     graph metadata table                        -> results/stats.csv
  sweep-s   V1, V3, V4 over s in {1,2,4,...,256}         -> results/sweep_s.csv
  sweep-t   V2, V3, V4 over the (t1, t2) grid             -> results/sweep_t.csv

Usage:
  python3 tools/bench.py data/*.egr
  python3 tools/bench.py data/road_usa.egr --modes sweep-s --runs 3
The timing CSV is appended to, so re-running adds rows; delete results/ to start fresh.
"""
import argparse
import csv
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode == 3:
        print("!! WRONG PARTITION reported above — investigate before using these numbers", flush=True)
    elif r.returncode != 0:
        sys.exit(f"command failed ({r.returncode})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graphs", nargs="+")
    ap.add_argument("--modes", default="main,counters,stats")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--t1", type=int, default=4)
    ap.add_argument("--t2", type=int, default=32)
    a = ap.parse_args()

    exe = ".exe" if os.name == "nt" else ""
    cc = os.path.join(ROOT, "bin", "cc" + exe)
    ccc = os.path.join(ROOT, "bin", "cc_counters" + exe)
    if not os.path.exists(cc):
        sys.exit(f"{cc} missing — build first (make / build.bat)")
    os.makedirs(a.out, exist_ok=True)
    modes = set(a.modes.split(","))
    tag = ["--tag", a.tag] if a.tag else []
    th = ["--t1", str(a.t1), "--t2", str(a.t2)]
    out = lambda f: os.path.join(a.out, f)  # noqa: E731

    if "stats" in modes:
        rows = []
        for g in a.graphs:
            r = subprocess.run([cc, g, "--stats"], capture_output=True, text=True, check=True)
            rows.append(r.stdout.strip().splitlines())
        with open(out("stats.csv"), "w", newline="") as f:
            f.write(rows[0][0] + "\n")
            for r in rows:
                f.write(r[1] + "\n")
        print(f"wrote {out('stats.csv')}")

    for g in a.graphs:
        if "main" in modes:
            run([cc, g, "--ver", "all", "--runs", str(a.runs), "--warmup", str(a.warmup),
                 "--csv", out("timing.csv")] + th + tag)
        if "counters" in modes:
            if not os.path.exists(ccc):
                sys.exit("bin/cc_counters missing — run `make counters` (Linux) or build.bat (Windows)")
            run([ccc, g, "--ver", "all", "--runs", "1", "--warmup", "0",
                 "--csv", out("counters.csv"), "--seg-csv", out("segments.csv")] + th + tag)
        if "sweep-s" in modes:
            for s in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                run([cc, g, "--ver", "V1,V3,V4", "--segments", str(s), "--runs", str(a.runs),
                     "--warmup", str(a.warmup), "--csv", out("sweep_s.csv")] + th + tag)
        if "sweep-t" in modes:
            for t1 in [2, 4, 8]:
                for t2 in [16, 32, 64, 128]:
                    run([cc, g, "--ver", "V2,V3,V4", "--t1", str(t1), "--t2", str(t2),
                         "--runs", str(a.runs), "--warmup", str(a.warmup), "--csv", out("sweep_t.csv")] + tag)


if __name__ == "__main__":
    main()
