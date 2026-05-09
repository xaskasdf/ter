#!/usr/bin/env python3
"""Generates a 256-entry int32 sigmoid LUT for tk_silu."""
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
    s = 1.0 / (1.0 + np.exp(-x))
    lut = np.round(s * a.out_scale).astype(np.int32)
    out_path = os.path.join(a.out_dir, "sigmoid_lut.bin")
    lut.tofile(out_path)
    with open(os.path.join(a.out_dir, "sigmoid_lut.meta"), "w") as f:
        f.write(f"n_entries={a.n_entries}\n")
        f.write(f"out_scale={a.out_scale}\n")
        f.write(f"x_step=0.03125\n")
        f.write(f"x_offset_idx={(a.n_entries - 1) / 2.0}\n")
    print(f"wrote {a.n_entries}-entry sigmoid LUT to {out_path}")
    print(f"sigmoid range: x in [{x[0]:.2f}, {x[-1]:.2f}], y in [{s[0]:.4f}, {s[-1]:.4f}]")

if __name__ == "__main__":
    main()
