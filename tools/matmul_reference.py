#!/usr/bin/env python3
"""Generates A, B, and C=A@B for the F2 phase gate."""
import argparse
import os
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="build/matmul_data")
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--n", type=int, default=64)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    lo, hi = -9841, 9841
    A = rng.integers(lo, hi + 1, size=(a.n, a.n), dtype=np.int32)
    B = rng.integers(lo, hi + 1, size=(a.n, a.n), dtype=np.int32)
    C = A.astype(np.int64) @ B.astype(np.int64)
    A.tofile(os.path.join(a.out_dir, "A.bin"))
    B.tofile(os.path.join(a.out_dir, "B.bin"))
    C.tofile(os.path.join(a.out_dir, "C.bin"))
    print("wrote A,B,C to", a.out_dir)

if __name__ == "__main__":
    main()
