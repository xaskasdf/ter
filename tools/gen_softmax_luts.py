#!/usr/bin/env python3
"""Generates exp + reciprocal LUTs for tk_softmax."""
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
    x = (idx - (a.n_entries - 1) / 2.0) / 32.0
    e = np.exp(x)
    exp_max = e.max()
    exp_lut = np.round(e * a.out_scale / exp_max).astype(np.int32)
    exp_path = os.path.join(a.out_dir, "exp_lut.bin")
    exp_lut.tofile(exp_path)

    val = (idx + 1.0) / float(a.n_entries)
    r = 1.0 / val
    rcp_max = r.max()
    rcp_lut = np.round(r * a.out_scale / rcp_max).astype(np.int32)
    rcp_path = os.path.join(a.out_dir, "rcp_lut.bin")
    rcp_lut.tofile(rcp_path)

    with open(os.path.join(a.out_dir, "softmax_lut.meta"), "w") as f:
        f.write(f"n_entries={a.n_entries}\n")
        f.write(f"out_scale={a.out_scale}\n")
        f.write(f"exp_max={exp_max}\n")
        f.write(f"rcp_max={rcp_max}\n")
        f.write(f"x_step=0.03125\n")
        f.write(f"x_offset_idx={(a.n_entries - 1) / 2.0}\n")
    print(f"wrote exp_lut.bin and rcp_lut.bin to {a.out_dir}")
    print(f"exp_max = {exp_max:.4f}, rcp_max = {rcp_max:.4f}")

if __name__ == "__main__":
    main()
