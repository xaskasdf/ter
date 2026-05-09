# ter — Number Formats

## Format B — fixed-point ternary + per-tensor scale (MVP)

Each tensor stores:
- `dtype = TritFP_B`
- `n_trits_per_elem` (default 9)
- `scale: float32` (per tensor)
- `shape`
- `payload`: one Word27 per element, lower `n_trits_per_elem` trits valid

### Conversion bf16/float → trit

```
mti      = (3^n_trits_per_elem - 1) / 2     # max representable trit-int
scale    = max(|tensor|) / mti              # 0 if tensor is all-zero
trit_int = round(value / scale)             # clamped to [-mti, +mti]
trits    = balanced_ternary_digits(trit_int, n_trits_per_elem)
```

### Quality budget

With `n_trits_per_elem = 9`:
- Effective bits ≈ log2(3^9) = 14.27 — between int10 and int15.
- Quantization noise variance ≤ scale²/12 per element.
- Round-trip MSE on uniform `[-1, 1]` test data is bounded by `4·scale²/12` (see `test_numfmt_roundtrip.cpp`).

### Matmul under format B

For `Y = X · W^T`:
1. `acc_int = sum_k X.payload[i,k].to_int() * W.payload[k,j].to_int()` — pure integer ternary, computed by `tvmac` chunks.
2. `Y[i,j] = acc_int * X.scale * W.scale` — single float multiply per output, on the host.

Inside the simulator there are zero floating-point operations. The float scale lives on the host, applied once per output tile. This is the operational embodiment of the project's thesis: matmul reduces to ternary additions and a single per-tile float scale.

Validation: `tests/test_kernel_matmul_b_32.cpp` — random standard-normal `A (8,27)` and `B (27,8)`, quantized to format B (9 trits), computed via the `tk_matmul_b_9t` kernel through host-orchestrated K-tiling, dequantized with the product of scales, and compared element-by-element to numpy reference. Measured `max_rel_err < 1e-2`.

## Format A — `tfloat` (deferred)

Native ternary float (1-trit sign · 5-trit exponent · 9-trit mantissa). Documented for the optional F7 phase.
