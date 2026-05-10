// ter_cuda_forward.cu -- end-to-end Llama 1B forward in CUDA, INT8 tensor core
// matmuls + custom kernels for RMSNorm/RoPE/attention/SiLU/argmax. Random
// weights (throughput skeleton, not real inference). Times N tokens end-to-end
// for honest comparison vs llama.cpp Q8_0.
//
// Build:
//   nvcc -O3 -arch=sm_86 -std=c++17 ter_cuda_forward.cu -lcublas -o ter_cuda_forward

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>

#define CK(call) do { cudaError_t e=(call); if(e){std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);} } while(0)
#define CB(call) do { cublasStatus_t s=(call); if(s){std::fprintf(stderr,"cuBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s);std::exit(1);} } while(0)

// -------- Llama 3.2 1B config --------
struct Cfg {
    int H        = 2048;        // hidden_size
    int F        = 8192;        // ffn intermediate
    int Nl       = 16;          // n_layers
    int Nh       = 32;          // n_heads
    int Nkv      = 8;           // n_kv_heads
    int Hd       = 64;          // head_dim = H / Nh
    int Hkv;                    // n_kv_heads * head_dim = 512
    int V        = 128256;      // vocab
    int Smax     = 64;          // max context for KV cache
    float eps    = 1e-5f;
    float rope_theta = 500000.0f;
};

// -------- Custom CUDA kernels --------

// rmsnorm: y = x * rsqrt(mean(x^2) + eps) * w. One block per row; H<=8192 fits in 1 block (max 1024 threads, vectorized).
__global__ void rmsnorm_k(const __half* x, const __half* w, __half* y, int H, float eps) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float local = 0.f;
    for (int i = tid; i < H; i += blockDim.x) {
        float v = __half2float(x[i]);
        local += v * v;
    }
    smem[tid] = local;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    float rms_inv = rsqrtf(smem[0] / (float)H + eps);
    for (int i = tid; i < H; i += blockDim.x) {
        float v = __half2float(x[i]) * rms_inv * __half2float(w[i]);
        y[i] = __float2half(v);
    }
}

// quantize fp16 -> int8 with abs-max per-tensor scale (computed inline).
// Single block; output_scale written to scale_out.
__global__ void quant_int8_k(const __half* x, int8_t* y, float* scale_out, int N) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float local = 0.f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = fabsf(__half2float(x[i]));
        if (v > local) local = v;
    }
    smem[tid] = local;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s && smem[tid+s] > smem[tid]) smem[tid] = smem[tid+s];
        __syncthreads();
    }
    float amax = smem[0] + 1e-9f;
    float scale = 127.0f / amax;
    if (tid == 0) *scale_out = 1.0f / scale;  // dequant scale
    for (int i = tid; i < N; i += blockDim.x) {
        float v = __half2float(x[i]) * scale;
        int q = __float2int_rn(v);
        if (q > 127) q = 127; if (q < -128) q = -128;
        y[i] = (int8_t)q;
    }
}

// dequantize int32 -> fp16 with scalar scale broadcast
__global__ void dequant_int32_k(const int32_t* x, __half* y, float scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    y[i] = __float2half((float)x[i] * scale);
}

// RoPE interleaved (Llama 3 style). v has shape (n_heads, head_dim) where head_dim is even.
// rotates pairs (v[2k], v[2k+1]) by angle = pos / theta^(2k/head_dim).
__global__ void rope_k(__half* v, int n_heads, int head_dim, int pos, float theta) {
    int h = blockIdx.x;
    int k = threadIdx.x;
    if (k >= head_dim/2) return;
    float freq = 1.0f / powf(theta, (2.0f * k) / (float)head_dim);
    float angle = (float)pos * freq;
    float c = cosf(angle), s = sinf(angle);
    int idx = h * head_dim + 2*k;
    float v0 = __half2float(v[idx]);
    float v1 = __half2float(v[idx+1]);
    v[idx]   = __float2half(v0 * c - v1 * s);
    v[idx+1] = __float2half(v0 * s + v1 * c);
}

// silu_mul: out = silu(gate) * up. silu(x) = x * sigmoid(x).
__global__ void silu_mul_k(const __half* gate, const __half* up, __half* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    float s = g / (1.f + expf(-g));   // silu
    out[i] = __float2half(s * u);
}

// add_residual: x += y
__global__ void add_k(__half* x, const __half* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    x[i] = __float2half(__half2float(x[i]) + __half2float(y[i]));
}

// scaled-dot attention scoring: scores[h, s] = (Q[h] . K_cache[s, kv_h]) / sqrt(head_dim)
// Then softmax over s in [0, pos]; finally out[h, d] = sum_s scores[h, s] * V_cache[s, kv_h, d].
// One block per query head; threads parallelize over seq positions then head_dim.
__global__ void attention_k(
    const __half* Q,            // (n_heads, head_dim)
    const __half* K_cache,      // (max_seq, n_kv_heads, head_dim)
    const __half* V_cache,      // (max_seq, n_kv_heads, head_dim)
    __half* out,                // (n_heads, head_dim)
    int n_heads, int n_kv_heads, int head_dim, int seq_len, int max_seq)
{
    int h = blockIdx.x;
    int kv_h = h * n_kv_heads / n_heads;
    int tid = threadIdx.x;

    extern __shared__ float smem[];     // size: seq_len floats for scores + head_dim floats for out
    float* scores = smem;
    float* outv   = smem + seq_len;

    float inv_sqrt = rsqrtf((float)head_dim);
    // Compute Q . K[s] for each s
    for (int s = tid; s < seq_len; s += blockDim.x) {
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            float qv = __half2float(Q[h * head_dim + d]);
            float kv = __half2float(K_cache[(s * n_kv_heads + kv_h) * head_dim + d]);
            dot += qv * kv;
        }
        scores[s] = dot * inv_sqrt;
    }
    __syncthreads();

    // Softmax: max + exp + normalize
    if (tid == 0) {
        float mx = scores[0];
        for (int s = 1; s < seq_len; ++s) if (scores[s] > mx) mx = scores[s];
        float sum = 0.f;
        for (int s = 0; s < seq_len; ++s) { scores[s] = expf(scores[s] - mx); sum += scores[s]; }
        float inv = 1.f / sum;
        for (int s = 0; s < seq_len; ++s) scores[s] *= inv;
    }
    __syncthreads();

    // Weighted sum of V: out[h, d] = sum_s scores[s] * V[s, kv_h, d]
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < seq_len; ++s) {
            acc += scores[s] * __half2float(V_cache[(s * n_kv_heads + kv_h) * head_dim + d]);
        }
        outv[d] = acc;
    }
    __syncthreads();
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out[h * head_dim + d] = __float2half(outv[d]);
    }
}

// argmax over fp16 logits. Single block, one element per thread atomically.
__global__ void argmax_k(const __half* logits, int* out_idx, int N) {
    extern __shared__ float smem[];
    int* sidx = (int*)(smem + blockDim.x);
    int tid = threadIdx.x;
    float best = -INFINITY; int bi = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = __half2float(logits[i]);
        if (v > best) { best = v; bi = i; }
    }
    smem[tid] = best; sidx[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s && smem[tid+s] > smem[tid]) { smem[tid] = smem[tid+s]; sidx[tid] = sidx[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) *out_idx = sidx[0];
}

// -------- Helpers --------

static void cublas_int8_matmul(cublasHandle_t h,
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K)
{
    // Compute C(N,M) = B(N,K) x A(K,M).  In our forward A=X (K,M), B=W (N,K) row-major
    // = (K,N) col-major; cuBLAS expects column-major args.
    const int32_t alpha = 1, beta = 0;
    CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &alpha, B, CUDA_R_8I, N, A, CUDA_R_8I, K,
        &beta,  C, CUDA_R_32I, N,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// matmul + dequantize-to-fp16 wrapper. activations are quantized (X int8 + scale_x);
// weights pre-quantized once (W int8 + scale_w). Output fp16 = scale_x * scale_w * INT32 acc.
struct WI8 { int8_t* data; float scale; };
static void mm_int8_to_fp16(cublasHandle_t cublas,
    const int8_t* X_i8, float scale_x, const WI8& W,
    int32_t* tmp_i32, __half* out_fp16,
    int M, int K, int N)
{
    cublas_int8_matmul(cublas, X_i8, W.data, tmp_i32, M, N, K);
    float comb_scale = scale_x * W.scale;
    int n = M * N;
    int blocks = (n + 255) / 256;
    dequant_int32_k<<<blocks, 256>>>(tmp_i32, out_fp16, comb_scale, n);
}

// -------- Random init --------

static void fill_random_int8(int8_t* d, size_t n, std::mt19937& rng) {
    std::vector<int8_t> h(n);
    std::uniform_int_distribution<int> d3(-1, 1);   // BitNet-style ternary
    for (auto& x : h) x = (int8_t)d3(rng);
    CK(cudaMemcpy(d, h.data(), n * sizeof(int8_t), cudaMemcpyHostToDevice));
}

static void fill_random_half(__half* d, size_t n, std::mt19937& rng) {
    std::vector<__half> h(n);
    std::uniform_real_distribution<float> df(-1.f, 1.f);
    for (auto& x : h) x = __float2half(df(rng));
    CK(cudaMemcpy(d, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
}

// -------- Forward --------

struct Layer {
    WI8 Wq, Wk, Wv, Wo, Wgate, Wup, Wdown;
    __half *attn_norm, *ffn_norm;
};
struct Model {
    Cfg cfg;
    __half* token_embd;     // (V, H) fp16
    Layer* layers;          // Nl
    __half* output_norm;    // H
    WI8 lm_head;            // (V, H) int8 = same shape as token_embd (tied)
};

struct Scratch {
    __half *hidden, *x_norm;
    __half *q, *k, *v, *attn_out;
    __half *gate, *up, *ff;
    __half *logits;
    int8_t *x_q, *ff_q;
    int32_t *tmp_i32;
    float *scale_buf;
    __half *K_cache, *V_cache;
    int *next_token;
};

void alloc_model(Model& m, const Cfg& c, std::mt19937& rng) {
    m.cfg = c;
    c.Hkv;  // unused warn fix
    int Hkv = c.Nkv * c.Hd;
    auto alloc_w = [&](WI8& w, size_t n) {
        CK(cudaMalloc(&w.data, n));
        fill_random_int8(w.data, n, rng);
        w.scale = 0.05f;  // arbitrary; throughput, not accuracy
    };
    auto alloc_h = [&](__half** p, size_t n) {
        CK(cudaMalloc(p, n * sizeof(__half)));
        fill_random_half(*p, n, rng);
    };

    alloc_h(&m.token_embd, (size_t)c.V * c.H);
    alloc_h(&m.output_norm, c.H);
    m.layers = new Layer[c.Nl];
    for (int L = 0; L < c.Nl; ++L) {
        Layer& l = m.layers[L];
        alloc_w(l.Wq,    (size_t)c.H * c.H);
        alloc_w(l.Wk,    (size_t)c.H * Hkv);
        alloc_w(l.Wv,    (size_t)c.H * Hkv);
        alloc_w(l.Wo,    (size_t)c.H * c.H);
        alloc_w(l.Wgate, (size_t)c.H * c.F);
        alloc_w(l.Wup,   (size_t)c.H * c.F);
        alloc_w(l.Wdown, (size_t)c.F * c.H);
        alloc_h(&l.attn_norm, c.H);
        alloc_h(&l.ffn_norm,  c.H);
    }
    alloc_w(m.lm_head, (size_t)c.V * c.H);
}

void alloc_scratch(Scratch& s, const Cfg& c) {
    int Hkv = c.Nkv * c.Hd;
    auto allh = [&](__half** p, size_t n) { CK(cudaMalloc(p, n * sizeof(__half))); };
    allh(&s.hidden,   c.H);
    allh(&s.x_norm,   c.H);
    allh(&s.q,        c.H);
    allh(&s.k,        Hkv);
    allh(&s.v,        Hkv);
    allh(&s.attn_out, c.H);
    allh(&s.gate,     c.F);
    allh(&s.up,       c.F);
    allh(&s.ff,       c.F);
    allh(&s.logits,   c.V);
    CK(cudaMalloc(&s.x_q,       std::max(c.H, c.F)));
    CK(cudaMalloc(&s.ff_q,      c.F));
    CK(cudaMalloc(&s.tmp_i32,   std::max((size_t)c.V, (size_t)c.F * 1) * sizeof(int32_t)));
    CK(cudaMalloc(&s.scale_buf, sizeof(float)));
    CK(cudaMalloc(&s.K_cache,   (size_t)c.Nl * c.Smax * Hkv * sizeof(__half)));
    CK(cudaMalloc(&s.V_cache,   (size_t)c.Nl * c.Smax * Hkv * sizeof(__half)));
    CK(cudaMalloc(&s.next_token, sizeof(int)));
}

int forward_token(cublasHandle_t cublas, Model& m, Scratch& s, int token_id, int pos)
{
    const Cfg& c = m.cfg;
    int Hkv = c.Nkv * c.Hd;

    // 1) Embedding lookup: hidden = token_embd[token_id, :]
    CK(cudaMemcpy(s.hidden, m.token_embd + (size_t)token_id * c.H,
                  c.H * sizeof(__half), cudaMemcpyDeviceToDevice));

    for (int L = 0; L < c.Nl; ++L) {
        Layer& l = m.layers[L];

        // 2) attn_norm
        rmsnorm_k<<<1, 256, 256*sizeof(float)>>>(s.hidden, l.attn_norm, s.x_norm, c.H, c.eps);

        // 3) Quantize x_norm -> int8
        float scale_x;  // we'll read from device after async, but simpler: inline write to scratch
        quant_int8_k<<<1, 256, 256*sizeof(float)>>>(s.x_norm, s.x_q, s.scale_buf, c.H);
        CK(cudaMemcpy(&scale_x, s.scale_buf, sizeof(float), cudaMemcpyDeviceToHost));

        // 4) Q,K,V projections (INT8 TC)
        mm_int8_to_fp16(cublas, s.x_q, scale_x, l.Wq, s.tmp_i32, s.q, 1, c.H, c.H);
        mm_int8_to_fp16(cublas, s.x_q, scale_x, l.Wk, s.tmp_i32, s.k, 1, c.H, Hkv);
        mm_int8_to_fp16(cublas, s.x_q, scale_x, l.Wv, s.tmp_i32, s.v, 1, c.H, Hkv);

        // 5) RoPE on Q (per query head) and K (per kv head)
        rope_k<<<c.Nh,  c.Hd/2>>>(s.q, c.Nh,  c.Hd, pos, c.rope_theta);
        rope_k<<<c.Nkv, c.Hd/2>>>(s.k, c.Nkv, c.Hd, pos, c.rope_theta);

        // 6) Write K, V to cache
        size_t kv_off = ((size_t)L * c.Smax + pos) * Hkv;
        CK(cudaMemcpy(s.K_cache + kv_off, s.k, Hkv * sizeof(__half), cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(s.V_cache + kv_off, s.v, Hkv * sizeof(__half), cudaMemcpyDeviceToDevice));

        // 7) Attention: scores = Q @ K[0..pos] ; softmax ; attn = scores @ V
        size_t cache_off = (size_t)L * c.Smax * Hkv;
        int seq_len = pos + 1;
        size_t shmem = (seq_len + c.Hd) * sizeof(float);
        attention_k<<<c.Nh, 64, shmem>>>(
            s.q, s.K_cache + cache_off, s.V_cache + cache_off,
            s.attn_out, c.Nh, c.Nkv, c.Hd, seq_len, c.Smax);

        // 8) Wo projection
        quant_int8_k<<<1, 256, 256*sizeof(float)>>>(s.attn_out, s.x_q, s.scale_buf, c.H);
        CK(cudaMemcpy(&scale_x, s.scale_buf, sizeof(float), cudaMemcpyDeviceToHost));
        mm_int8_to_fp16(cublas, s.x_q, scale_x, l.Wo, s.tmp_i32, s.attn_out, 1, c.H, c.H);

        // 9) Residual: hidden += attn_out
        int blocks = (c.H + 255) / 256;
        add_k<<<blocks, 256>>>(s.hidden, s.attn_out, c.H);

        // 10) ffn_norm
        rmsnorm_k<<<1, 256, 256*sizeof(float)>>>(s.hidden, l.ffn_norm, s.x_norm, c.H, c.eps);

        // 11) FFN: gate, up, then silu_mul, then down
        quant_int8_k<<<1, 256, 256*sizeof(float)>>>(s.x_norm, s.x_q, s.scale_buf, c.H);
        CK(cudaMemcpy(&scale_x, s.scale_buf, sizeof(float), cudaMemcpyDeviceToHost));
        mm_int8_to_fp16(cublas, s.x_q, scale_x, l.Wgate, s.tmp_i32, s.gate, 1, c.H, c.F);
        mm_int8_to_fp16(cublas, s.x_q, scale_x, l.Wup,   s.tmp_i32, s.up,   1, c.H, c.F);

        int fblocks = (c.F + 255) / 256;
        silu_mul_k<<<fblocks, 256>>>(s.gate, s.up, s.ff, c.F);

        quant_int8_k<<<1, 1024, 1024*sizeof(float)>>>(s.ff, s.ff_q, s.scale_buf, c.F);
        CK(cudaMemcpy(&scale_x, s.scale_buf, sizeof(float), cudaMemcpyDeviceToHost));
        mm_int8_to_fp16(cublas, s.ff_q, scale_x, l.Wdown, s.tmp_i32, s.attn_out, 1, c.F, c.H);

        // 12) Residual: hidden += ff_out
        add_k<<<blocks, 256>>>(s.hidden, s.attn_out, c.H);
    }

    // 13) Final RMSNorm
    rmsnorm_k<<<1, 256, 256*sizeof(float)>>>(s.hidden, m.output_norm, s.x_norm, c.H, c.eps);

    // 14) lm_head logits
    float scale_x;
    quant_int8_k<<<1, 256, 256*sizeof(float)>>>(s.x_norm, s.x_q, s.scale_buf, c.H);
    CK(cudaMemcpy(&scale_x, s.scale_buf, sizeof(float), cudaMemcpyDeviceToHost));
    mm_int8_to_fp16(cublas, s.x_q, scale_x, m.lm_head, s.tmp_i32, s.logits, 1, c.H, c.V);

    // 15) Argmax
    argmax_k<<<1, 256, 256*(sizeof(float)+sizeof(int))>>>(s.logits, s.next_token, c.V);
    int next = 0;
    CK(cudaMemcpy(&next, s.next_token, sizeof(int), cudaMemcpyDeviceToHost));
    return next;
}

int main(int argc, char** argv)
{
    int n_gen = (argc > 1) ? std::atoi(argv[1]) : 32;
    int n_warmup = 4;

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s, SM %d.%d, %d SMs, %.1f GiB\n",
        prop.name, prop.major, prop.minor, prop.multiProcessorCount,
        prop.totalGlobalMem / (1024.0*1024.0*1024.0));

    cublasHandle_t cublas; CB(cublasCreate(&cublas));

    Cfg c;
    c.Hkv = c.Nkv * c.Hd;

    std::printf("Llama 1B config: H=%d F=%d L=%d Nh=%d Nkv=%d Hd=%d V=%d Smax=%d\n",
        c.H, c.F, c.Nl, c.Nh, c.Nkv, c.Hd, c.V, c.Smax);

    std::mt19937 rng(42);
    Model m; alloc_model(m, c, rng);
    Scratch s; alloc_scratch(s, c);

    // Warmup
    for (int t = 0; t < n_warmup; ++t) {
        forward_token(cublas, m, s, 1, t);
    }
    CK(cudaDeviceSynchronize());

    // Time n_gen tokens
    auto t0 = std::chrono::high_resolution_clock::now();
    int tok = 1;
    for (int t = 0; t < n_gen; ++t) {
        tok = forward_token(cublas, m, s, tok, n_warmup + t);
    }
    CK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_tok = ms / n_gen;
    double tps = 1000.0 / per_tok;
    std::printf("\n=== ter sim end-to-end forward (INT8 TC, full pipeline) ===\n");
    std::printf("n_gen tokens : %d\n", n_gen);
    std::printf("total ms     : %.3f\n", ms);
    std::printf("ms / token   : %.4f\n", per_tok);
    std::printf("tokens/s     : %.1f\n", tps);
    std::printf("Compare: llama.cpp Q8_0 RTX 3090 tg128 ~ 130 t/s = 7.7 ms/token\n");

    return 0;
}
