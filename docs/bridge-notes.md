# `ter` Bridge Notes

Captured during F5.1 — lifting `~/ntransformer` infrastructure into `vendor/ntransformer/`.

## What we lifted

- `core/types.{h,cpp}` — `DType` enum with `DType::TERNARY = 9` added.
- `core/device.h` — `Device::CPU` only (CUDA stripped).
- `core/tensor.{h,cpp}` — CUDA paths stripped, plain malloc backing.
- `model/config.{h,cpp}` — drop-in (no CUDA in source).
- `model/loader.{h,cpp}` — mmap GGUF parsing, already CUDA-free.
- `inference/tokenizer.{h,cpp}` — drop-in (BPE, no CUDA).
- `inference/sampler.{h,cpp}` — drop-in (RNG-only).

## What we did NOT lift

- `model/transformer.{h,cpp}`, `model/attention.{h,cpp}`, `model/ffn.{h,cpp}`, `model/norm.{h,cpp}` — these hardcode `cuda::launch_*()` calls; we will write our own thin replacements that call our 5 ternary kernels (F5.2).
- `inference/engine.{h,cpp}` — entangled with the streaming pipeline; we will write a minimal generation loop (F5.3).
- `src/cuda/*` — replaced entirely by our kernels and `Sim::call_kernel`.
- `src/memory/streamer.*` (NVMe streaming) — out of scope.

## Why this split

Exploration found no backend abstraction in ntransformer. Patching attention.cpp / ffn.cpp / norm.cpp with `if (use_ternary_sim_)` branches at every call site would be invasive and ugly. Writing fresh transformer logic that calls our kernels directly is cleaner and gives us a clean Llama-on-ternary front-end without inheriting CUDA orchestration.

## Bridge contract (F5.2)

Our transformer code (in `src/host/ter_transformer.cpp`, F5.2) will:
- Take an `nt::ModelConfig` and `nt::GGUFLoader` for setup.
- Use `nt::Tensor` for activations and weights.
- Convert weights to `DType::TERNARY` at load time (per-tensor scale stored alongside).
- Call our 5 kernels (`tk_matmul_b_9t`, `tk_rmsnorm`, `tk_softmax`, `tk_silu`, `tk_rope`) for arithmetic.
- Use `nt::Tokenizer` for I/O and `nt::Sampler` for token selection.

This keeps the bridge surface small: only the transformer/attention/ffn/norm logic is new code; everything around it is the user's tested infrastructure.

## Useful API anchors

- `nt::DType::TERNARY` = 9 (in `vendor/ntransformer/core/types.h`).
- `nt::Tensor::from_ptr(data, shape, dtype, Device::CPU)` — zero-copy wrap around a pointer (good for sim memory).
- `nt::GGUFLoader::load(path)` returns bool; `.config()` returns `ModelConfig`; `.get_tensor(name)` returns a CPU `Tensor`.
- `nt::Tokenizer::init(vocab, bos, eos)`, `.encode(text)`, `.decode(tokens)`.
- `nt::Sampler::init(SamplerConfig)`, `.sample(logits, n)` returns `int`.
