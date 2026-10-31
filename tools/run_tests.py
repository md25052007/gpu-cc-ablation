#!/usr/bin/env python3
"""Correctness tests (same checks as `make tests`, also works on Windows).

1. Generates the 12 small test graphs (tools/gen_tests.py, fixed random seed).
2. Checks each graph's size and component count against tests/expected_test_stats.csv,
   so you know you are testing the same graphs we did.
3. Runs every version (V0-V4) on every graph; the program itself compares each
   run's labelling with a sequential CPU union-find reference and fails on any mismatch.

Usage: python3 tools/run_tests.py             (real GPU)
       python3 tools/run_tests.py --emulate   (kernel logic run on the CPU, no GPU needed)
"""
import csv
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cc = os.path.join(ROOT, "bin", "cc" + (".exe" if os.name == "nt" else ""))
if not os.path.exists(cc):
    sys.exit(f"{cc} missing - build first (make / build.bat)")
tests = os.path.join(ROOT, "data", "tests")
subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gen_tests.py"), tests], check=True)

expected_file = os.path.join(ROOT, "tests", "expected_test_stats.csv")
expected = {}
if os.path.exists(expected_file):
    with open(expected_file) as f:
        expected = {r["graph"]: r for r in csv.DictReader(f)}

extra = ["--emulate"] if "--emulate" in sys.argv else []
failed = []
for g in sorted(glob.glob(os.path.join(tests, "*.egr"))):
    name = os.path.splitext(os.path.basename(g))[0]
    if name in expected:
        out = subprocess.run([cc, g, "--stats"], capture_output=True, text=True, check=True).stdout
        got = dict(zip(*[line.split(",") for line in out.strip().splitlines()[:2]]))
        for key in ("n", "m", "components", "largest_cc"):
            if got[key] != expected[name][key]:
                print(f"[stats mismatch] {name}: {key} = {got[key]}, expected {expected[name][key]}")
                failed.append(name)
                break
    r = subprocess.run([cc, g, "--runs", "3", "--warmup", "1"] + extra)
    if r.returncode != 0:
        failed.append(name)
if failed:
    sys.exit("FAILED: " + ", ".join(sorted(set(failed))))
print("ALL TESTS PASSED" + (" (CPU emulation)" if extra else ""))
