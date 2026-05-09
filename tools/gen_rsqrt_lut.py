#!/usr/bin/env python3
"""Generates a 256-entry int32 LUT for rsqrt, used by tk_rmsnorm."""
import argparse
import os
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="build/lut_data")
    p.add_argument("--n-entries", type=int, default=256)
    p.add_argument("--out-scale", type=int, default=9841)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    idx = np.arange(a.n_entries, dtype=np.float64)
    vals = (idx + 1.0) / float(a.n_entries)
    rsq = 1.0 / np.sqrt(vals)
    lut = np.round(rsq * a.out_scale / rsq.max()).astype(np.int32)
    out_path = os.path.join(a.out_dir, "rsqrt_lut.bin")
    lut.tofile(out_path)
    with open(os.path.join(a.out_dir, "rsqrt_lut.meta"), "w") as f:
        f.write(f"n_entries={a.n_entries}\nout_scale={a.out_scale}\nrsq_max={rsq.max()}\n")
    print(f"wrote {a.n_entries}-entry rsqrt LUT to {out_path}")
    print(f"rsq_max = {rsq.max():.6f}")

if __name__ == "__main__":
    main()
