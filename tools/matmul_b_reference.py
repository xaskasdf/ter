#!/usr/bin/env python3
"""Generates random float matrices A (M,K) and B (K,N) plus C = A@B for the format-B matmul test."""
import argparse
import os
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="build/matmul_b_data")
    p.add_argument("--seed", type=int, default=0xBEE)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--k", type=int, default=27)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    A = rng.standard_normal((a.m, a.k)).astype(np.float32)
    B = rng.standard_normal((a.k, a.n)).astype(np.float32)
    C = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
    A.tofile(os.path.join(a.out_dir, "A.bin"))
    B.tofile(os.path.join(a.out_dir, "B.bin"))
    C.tofile(os.path.join(a.out_dir, "C.bin"))
    with open(os.path.join(a.out_dir, "shape.txt"), "w") as f:
        f.write(f"{a.m} {a.n} {a.k}\n")
    print("wrote A,B,C,shape to", a.out_dir)

if __name__ == "__main__":
    main()
