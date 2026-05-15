# Ternary activations: substrate-aligned X path

Status: design proposal, not yet implemented (2026-05-15)
Related: H3 (substrate-data alignment for BitNet), Phase B (ADD-only kernel)

## The current state

The `ter` CUDA kernel toolkit treats activations as **int8** and weights as
**packed ternary** (4 trits/byte, 1.58 bits/elem). Matmul invokes `__dp4a`
(`acc = x[k] * w[k] + acc`) where `x` is signed 8-bit and `w` is `{-1, 0, +1}`.

For BitNet b1.58 weights, the multiplication is already degenerate: the
Phase B `tk_matmul_bitnet` kernel demonstrates TVMAC = 0 (zero `__dp4a`
invocations) by replacing multiply with conditional add/sub/skip. But X
remains int8: each accumulate operation is still an int8 ADD, not free.

## The proposal: also quantize X to ternary

Map activations to `{-1, 0, +1}` via a per-tensor scale + threshold:

```
x_ternary[k] = sign(x[k]) if |x[k]| > thresh else 0
x_scale     = mean(|x[k]| for k where x_ternary[k] != 0)
```

Then matmul becomes:

```
sum_k (x_t[k] * w_t[k]) where both are in {-1, 0, +1}
```

The product `x_t * w_t` is itself in `{-1, 0, +1}`. The accumulator counts
matches and mismatches:

```
acc = #{k : (x_t[k]=+1 and w_t[k]=+1) or (x_t[k]=-1 and w_t[k]=-1)}
    - #{k : (x_t[k]=+1 and w_t[k]=-1) or (x_t[k]=-1 and w_t[k]=+1)}
```

On packed-trit storage (2 bits per elem for both X and W):

```
pos_x = (x_t == +1)  // bit mask
neg_x = (x_t == -1)  // bit mask
pos_w = (w_t == +1)
neg_w = (w_t == -1)

pos_match = popcount((pos_x & pos_w) | (neg_x & neg_w))
neg_match = popcount((pos_x & neg_w) | (neg_x & pos_w))
acc       = pos_match - neg_match
```

CUDA `__popc` is 1-cycle on Ampere; processing 32 K-elements per popcount
pair gives a 4-bit-budget compute throughput equivalent to billions of
binary ops/second. The kernel structure is what BitNet papers call
"bit-serial popcount" and what BNN papers call "XNOR-net" (with sign
mapping).

## Why we didn't do it in round 2

In round 2 we considered "Phase D = quantize activations to ternary" as
an option but skipped it because:

1. **Accuracy risk:** Llama 3.2 1B was not trained for ternary activations.
   Naive post-training quantization of activations to `{-1, 0, +1}` would
   likely break model output. BitNet b1.58 2B-4T was trained with
   activations in **int8**, not ternary; their substrate-data alignment
   only spans the W axis.
2. **Two free wins were closer to hand:** Phase B (BitNet ADD-only,
   architectural TVMAC=0) and Phase E (INT4 TC + hybrid dispatch) both
   gave measurable wins without retraining. Ternary X requires either
   retraining or carefully designed activation calibration.
3. **The existing Phase B kernel already eliminates the multiply** for
   BitNet weights — the additional gain from ternary X is the elimination
   of the int8 ADD too, which is a smaller marginal win than the
   multiply elimination.

## When this becomes valuable

- **Models trained with ternary activations.** If a future BitNet variant
  (BitNet b1.58 with ternary X, or 1-bit-per-activation transformers)
  ships GGUF weights, the substrate is uniquely positioned to run them
  with full XNOR+popcount throughput.
- **Edge / mobile deployment.** A quality-vs-speed knob: ternary X
  trades quality for substantial speed and memory advantage, useful in
  bandwidth-bound regimes (mobile NPUs, embedded inference).
- **Custom silicon.** A ternary-native ALU has no intermediate types;
  ternary X aligns with what the hardware can natively do, eliminating
  an intermediate transformation step that exists only because GPUs
  expose int8 as the smallest practical operand type.
- **Speculative decoding draft model.** A ternary-X draft of a Q8 main
  model could give very high speculation throughput at the cost of more
  rejections.

## Implementation sketch

If/when implemented, the kernel structure would be:

```cuda
__global__ void mm_packed_xnor(
    const uint8_t* X_packed,  // K/4 bytes per row (4 trits/byte)
    const uint8_t* W_packed,  // K/4 bytes per col
    int K, int N, float scale, __half* out)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    int K_bytes = K / 4;
    int j_byte = j / 4;
    int sub = (j & 3) * 2;

    int pos_match = 0, neg_match = 0;

    // Process 16 K-elements per iter (4 bytes × 4 trits/byte)
    for (int kb = 0; kb < K_bytes; kb += 4) {
        uint32_t x4 = *reinterpret_cast<const uint32_t*>(X_packed + kb);
        uint32_t w4_packed = ...;  // gather W column from W_packed[kb*N_bytes + j_byte]

        // For each of the 16 trit positions, extract (pos, neg) bits
        // and accumulate via __popc
        // Implementation depends on the exact 2-bit code mapping
    }

    int acc = pos_match - neg_match;
    out[j] = __float2half(acc * scale);
}
```

The trick is the bit-twiddling to extract `pos_x_mask | pos_w_mask` from
2-bit-per-elem packed format efficiently. Several encodings are possible:

- Encoding A: `00=0, 01=+1, 10=-1` — direct, but XNOR doesn't directly map
- Encoding B: separate `pos` and `neg` bit-planes (2 separate bit arrays)
- Encoding C: packed sign + zero-mask (sign bit + zero bit per trit)

Encoding B (separate bit-planes) is the cleanest for popcount-based math
but doubles the metadata storage to manage. Encoding A is the current
storage but needs more bit manipulation per access.

## Decision

Document and defer. Phase E + the forward sync fixes give us enough
runway against the production baseline (1.25× behind ggml mmvq at gen
M=1). Ternary X is the next architectural lever if/when retraining is
on the table or the substrate moves to custom silicon.
