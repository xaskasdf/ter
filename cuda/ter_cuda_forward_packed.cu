// ter_cuda_forward_packed.cu -- end-to-end Llama 1B forward with packed
// 1.58-bit ternary weights (4 trits/byte) for ALL projections. Activations
// stay int8 for matmul; norms/attn/silu in fp16. Custom packed-trit kernel
// (v4_wide variant) replaces the cublasGemmEx INT8 path.
//
// Compares end-to-end wall-clock vs the int8 baseline (ter_cuda_forward.cu).
//
// Build:
//   nvcc -O3 -arch=sm_86 -std=c++17 ter_cuda_forward_packed.cu -o ter_cuda_forward_packed

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>

#define CK(call) do { cudaError_t e=(call); if(e){std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);} } while(0)

struct Cfg {
    int H = 2048, F = 8192, Nl = 16, Nh = 32, Nkv = 8, Hd = 64, V = 128256, Smax = 64;
    float eps = 1e-5f, rope_theta = 500000.0f;
};

// ---------- Same support kernels as ter_cuda_forward.cu ----------
__global__ void rmsnorm_k(const __half* x, const __half* w, __half* y, int H, float eps) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float local = 0.f;
    for (int i = tid; i < H; i += blockDim.x) { float v = __half2float(x[i]); local += v*v; }
    smem[tid] = local; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid<s) smem[tid]+=smem[tid+s]; __syncthreads(); }
    float rms_inv = rsqrtf(smem[0] / (float)H + eps);
    for (int i = tid; i < H; i += blockDim.x)
        y[i] = __float2half(__half2float(x[i]) * rms_inv * __half2float(w[i]));
}

__global__ void quant_int8_k(const __half* x, int8_t* y, float* scale_out, int N) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float local = 0.f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = fabsf(__half2float(x[i]));
        if (v > local) local = v;
    }
    smem[tid] = local; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid<s && smem[tid+s]>smem[tid]) smem[tid]=smem[tid+s];
        __syncthreads();
    }
    float amax = smem[0] + 1e-9f;
    float scale = 127.0f / amax;
    if (tid == 0) *scale_out = 1.0f / scale;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = __half2float(x[i]) * scale;
        int q = __float2int_rn(v);
        if (q>127) q=127; if (q<-128) q=-128;
        y[i] = (int8_t)q;
    }
}

__global__ void rope_k(__half* v, int n_heads, int head_dim, int pos, float theta) {
    int h = blockIdx.x;
    int k = threadIdx.x;
    if (k >= head_dim/2) return;
    float freq = 1.0f / powf(theta, (2.0f * k) / (float)head_dim);
    float a = (float)pos * freq;
    float c = cosf(a), s = sinf(a);
    int idx = h * head_dim + 2*k;
    float v0 = __half2float(v[idx]), v1 = __half2float(v[idx+1]);
    v[idx]   = __float2half(v0*c - v1*s);
    v[idx+1] = __float2half(v0*s + v1*c);
}

__global__ void silu_mul_k(const __half* g, const __half* u, __half* o, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=N) return;
    float gv = __half2float(g[i]), uv = __half2float(u[i]);
    float s = gv / (1.f + expf(-gv));
    o[i] = __float2half(s * uv);
}

__global__ void add_k(__half* x, const __half* y, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=N) return;
    x[i] = __float2half(__half2float(x[i]) + __half2float(y[i]));
}

__global__ void attention_k(
    const __half* Q, const __half* K, const __half* V, __half* out,
    int n_heads, int n_kv_heads, int head_dim, int seq_len, int max_seq)
{
    int h = blockIdx.x;
    int kv_h = h * n_kv_heads / n_heads;
    int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* scores = smem;
    float* outv = smem + seq_len;
    float inv_sqrt = rsqrtf((float)head_dim);
    for (int s = tid; s < seq_len; s += blockDim.x) {
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += __half2float(Q[h*head_dim+d]) * __half2float(K[(s*n_kv_heads+kv_h)*head_dim+d]);
        }
        scores[s] = dot * inv_sqrt;
    }
    __syncthreads();
    if (tid == 0) {
        float mx = scores[0];
        for (int s = 1; s < seq_len; ++s) if (scores[s]>mx) mx = scores[s];
        float sum = 0.f;
        for (int s = 0; s < seq_len; ++s) { scores[s] = expf(scores[s]-mx); sum += scores[s]; }
        float inv = 1.f/sum;
        for (int s = 0; s < seq_len; ++s) scores[s] *= inv;
    }
    __syncthreads();
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < seq_len; ++s)
            acc += scores[s] * __half2float(V[(s*n_kv_heads+kv_h)*head_dim+d]);
        outv[d] = acc;
    }
    __syncthreads();
    for (int d = tid; d < head_dim; d += blockDim.x)
        out[h*head_dim+d] = __float2half(outv[d]);
}

__global__ void argmax_k(const __half* logits, int* out_idx, int N) {
    extern __shared__ float smem[];
    int* sidx = (int*)(smem + blockDim.x);
    int tid = threadIdx.x;
    float best = -1e30f; int bi = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = __half2float(logits[i]);
        if (v > best) { best = v; bi = i; }
    }
    smem[tid] = best; sidx[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid<s && smem[tid+s]>smem[tid]) { smem[tid]=smem[tid+s]; sidx[tid]=sidx[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) *out_idx = sidx[0];
}

// ---------- The packed matmul: int8 X * packed-trit W -> fp16 out ----------
// Replaces cublasGemmEx INT8 + dequant. Per-tensor scale folded in here.
__global__ void mm_packed_v4(
    const int8_t*  X, const uint8_t* W, int K, int N, float scale, __half* out)
{
    constexpr int K_TILE = 512;
    __shared__ int8_t X_smem[K_TILE];
    int tid = threadIdx.x;
    int j = blockIdx.x * blockDim.x + tid;
    if (j >= N) return;
    int N_bytes = N / 4;
    int j_byte = j / 4;
    int sub_bit = (j & 3) * 2;
    int acc = 0;
    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        for (int i = tid; i < K_TILE && k0 + i < K; i += blockDim.x) X_smem[i] = X[k0 + i];
        __syncthreads();
        int kmax = min(K_TILE, K - k0);
        for (int kk = 0; kk < kmax; ++kk) {
            int xv = X_smem[kk];
            if (xv == 0) continue;
            uint8_t b = W[(k0 + kk) * N_bytes + j_byte];
            int t = (b >> sub_bit) & 3;
            acc += xv * ((t == 1) - (t == 2));
        }
        __syncthreads();
    }
    out[j] = __float2half((float)acc * scale);
}

// ---------- Model + forward ----------
struct WPK { uint8_t* data; float scale; size_t bytes; };
struct Layer {
    WPK Wq, Wk, Wv, Wo, Wgate, Wup, Wdown;
    __half *attn_norm, *ffn_norm;
};
struct Model {
    Cfg cfg;
    __half* token_embd;
    Layer* layers;
    __half* output_norm;
    WPK lm_head;
};
struct Scratch {
    __half *hidden, *x_norm, *q, *k, *v, *attn_out, *gate, *up, *ff, *logits;
    int8_t *x_q, *ff_q;
    float *scale_buf;
    __half *K_cache, *V_cache;
    int *next_token;
};

static void fill_random_packed(uint8_t* d, size_t bytes, std::mt19937& rng) {
    std::vector<uint8_t> h(bytes);
    std::uniform_int_distribution<int> d3(0, 2);  // 0=zero, 1=+1, 2=-1
    for (auto& b : h) {
        int t0 = d3(rng), t1 = d3(rng), t2 = d3(rng), t3 = d3(rng);
        b = t0 | (t1 << 2) | (t2 << 4) | (t3 << 6);
    }
    CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
}
static void fill_random_half(__half* d, size_t n, std::mt19937& rng) {
    std::vector<__half> h(n);
    std::uniform_real_distribution<float> df(-1.f, 1.f);
    for (auto& x : h) x = __float2half(df(rng));
    CK(cudaMemcpy(d, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
}

void alloc_packed(WPK& w, int K, int N, std::mt19937& rng) {
    w.bytes = (size_t)K * (N / 4);
    CK(cudaMalloc(&w.data, w.bytes));
    fill_random_packed(w.data, w.bytes, rng);
    w.scale = 0.05f;
}

void alloc_model(Model& m, const Cfg& c, std::mt19937& rng, size_t* total_bytes) {
    m.cfg = c;
    int Hkv = c.Nkv * c.Hd;
    auto allh = [&](__half** p, size_t n) {
        CK(cudaMalloc(p, n * sizeof(__half)));
        fill_random_half(*p, n, rng);
        *total_bytes += n * sizeof(__half);
    };

    allh(&m.token_embd, (size_t)c.V * c.H);
    allh(&m.output_norm, c.H);
    m.layers = new Layer[c.Nl];
    for (int L = 0; L < c.Nl; ++L) {
        Layer& l = m.layers[L];
        alloc_packed(l.Wq,    c.H, c.H,   rng); *total_bytes += l.Wq.bytes;
        alloc_packed(l.Wk,    c.H, Hkv,   rng); *total_bytes += l.Wk.bytes;
        alloc_packed(l.Wv,    c.H, Hkv,   rng); *total_bytes += l.Wv.bytes;
        alloc_packed(l.Wo,    c.H, c.H,   rng); *total_bytes += l.Wo.bytes;
        alloc_packed(l.Wgate, c.H, c.F,   rng); *total_bytes += l.Wgate.bytes;
        alloc_packed(l.Wup,   c.H, c.F,   rng); *total_bytes += l.Wup.bytes;
        alloc_packed(l.Wdown, c.F, c.H,   rng); *total_bytes += l.Wdown.bytes;
        allh(&l.attn_norm, c.H);
        allh(&l.ffn_norm,  c.H);
    }
    alloc_packed(m.lm_head, c.H, c.V, rng);
    *total_bytes += m.lm_head.bytes;
}

void alloc_scratch(Scratch& s, const Cfg& c) {
    int Hkv = c.Nkv * c.Hd;
    auto allh = [&](__half** p, size_t n) { CK(cudaMalloc(p, n * sizeof(__half))); };
    allh(&s.hidden, c.H); allh(&s.x_norm, c.H);
    allh(&s.q, c.H); allh(&s.k, Hkv); allh(&s.v, Hkv); allh(&s.attn_out, c.H);
    allh(&s.gate, c.F); allh(&s.up, c.F); allh(&s.ff, c.F);
    allh(&s.logits, c.V);
    CK(cudaMalloc(&s.x_q, std::max(c.H, c.F)));
    CK(cudaMalloc(&s.ff_q, c.F));
    CK(cudaMalloc(&s.scale_buf, sizeof(float)));
    CK(cudaMalloc(&s.K_cache, (size_t)c.Nl * c.Smax * Hkv * sizeof(__half)));
    CK(cudaMalloc(&s.V_cache, (size_t)c.Nl * c.Smax * Hkv * sizeof(__half)));
    CK(cudaMalloc(&s.next_token, sizeof(int)));
}

static void mm_packed_dispatch(const int8_t* X, float scale_x, const WPK& W,
                               __half* out, int K, int N)
{
    int t = 256, blocks = (N + t - 1) / t;
    mm_packed_v4<<<blocks, t>>>(X, W.data, K, N, scale_x * W.scale, out);
}

int forward_token(Model& m, Scratch& s, int token_id, int pos)
{
    const Cfg& c = m.cfg;
    int Hkv = c.Nkv * c.Hd;
    CK(cudaMemcpy(s.hidden, m.token_embd + (size_t)token_id * c.H, c.H*sizeof(__half), cudaMemcpyDeviceToDevice));

    for (int L = 0; L < c.Nl; ++L) {
        Layer& l = m.layers[L];
        rmsnorm_k<<<1,256,256*sizeof(float)>>>(s.hidden, l.attn_norm, s.x_norm, c.H, c.eps);
        float scale_x;
        quant_int8_k<<<1,256,256*sizeof(float)>>>(s.x_norm, s.x_q, s.scale_buf, c.H);
        CK(cudaMemcpy(&scale_x, s.scale_buf, sizeof(float), cudaMemcpyDeviceToHost));
        mm_packed_dispatch(s.x_q, scale_x, l.Wq, s.q, c.H, c.H);
        mm_packed_dispatch(s.x_q, scale_x, l.Wk, s.k, c.H, Hkv);
        mm_packed_dispatch(s.x_q, scale_x, l.Wv, s.v, c.H, Hkv);
        rope_k<<<c.Nh,  c.Hd/2>>>(s.q, c.Nh,  c.Hd, pos, c.rope_theta);
        rope_k<<<c.Nkv, c.Hd/2>>>(s.k, c.Nkv, c.Hd, pos, c.rope_theta);
        size_t kv_off = ((size_t)L * c.Smax + pos) * Hkv;
        CK(cudaMemcpy(s.K_cache + kv_off, s.k, Hkv*sizeof(__half), cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(s.V_cache + kv_off, s.v, Hkv*sizeof(__half), cudaMemcpyDeviceToDevice));
        size_t cache_off = (size_t)L * c.Smax * Hkv;
        int seq_len = pos + 1;
        size_t shmem = (seq_len + c.Hd) * sizeof(float);
        attention_k<<<c.Nh, 64, shmem>>>(s.q, s.K_cache+cache_off, s.V_cache+cache_off,
            s.attn_out, c.Nh, c.Nkv, c.Hd, seq_len, c.Smax);
        quant_int8_k<<<1,256,256*sizeof(float)>>>(s.attn_out, s.x_q, s.scale_buf, c.H);
        CK(cudaMemcpy(&scale_x, s.scale_buf, sizeof(float), cudaMemcpyDeviceToHost));
        mm_packed_dispatch(s.x_q, scale_x, l.Wo, s.attn_out, c.H, c.H);
        int blocks = (c.H+255)/256;
        add_k<<<blocks,256>>>(s.hidden, s.attn_out, c.H);

        rmsnorm_k<<<1,256,256*sizeof(float)>>>(s.hidden, l.ffn_norm, s.x_norm, c.H, c.eps);
        quant_int8_k<<<1,256,256*sizeof(float)>>>(s.x_norm, s.x_q, s.scale_buf, c.H);
        CK(cudaMemcpy(&scale_x, s.scale_buf, sizeof(float), cudaMemcpyDeviceToHost));
        mm_packed_dispatch(s.x_q, scale_x, l.Wgate, s.gate, c.H, c.F);
        mm_packed_dispatch(s.x_q, scale_x, l.Wup,   s.up,   c.H, c.F);
        int fblocks = (c.F+255)/256;
        silu_mul_k<<<fblocks,256>>>(s.gate, s.up, s.ff, c.F);
        quant_int8_k<<<1,1024,1024*sizeof(float)>>>(s.ff, s.ff_q, s.scale_buf, c.F);
        CK(cudaMemcpy(&scale_x, s.scale_buf, sizeof(float), cudaMemcpyDeviceToHost));
        mm_packed_dispatch(s.ff_q, scale_x, l.Wdown, s.attn_out, c.F, c.H);
        add_k<<<blocks,256>>>(s.hidden, s.attn_out, c.H);
    }

    rmsnorm_k<<<1,256,256*sizeof(float)>>>(s.hidden, m.output_norm, s.x_norm, c.H, c.eps);
    float scale_x;
    quant_int8_k<<<1,256,256*sizeof(float)>>>(s.x_norm, s.x_q, s.scale_buf, c.H);
    CK(cudaMemcpy(&scale_x, s.scale_buf, sizeof(float), cudaMemcpyDeviceToHost));
    mm_packed_dispatch(s.x_q, scale_x, m.lm_head, s.logits, c.H, c.V);
    argmax_k<<<1,256,256*(sizeof(float)+sizeof(int))>>>(s.logits, s.next_token, c.V);
    int next = 0;
    CK(cudaMemcpy(&next, s.next_token, sizeof(int), cudaMemcpyDeviceToHost));
    return next;
}

int main(int argc, char** argv) {
    int n_gen = (argc > 1) ? std::atoi(argv[1]) : 32;
    int n_warmup = 4;

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n", prop.name);

    Cfg c;
    std::printf("Llama 1B: H=%d F=%d L=%d Nh=%d Nkv=%d Hd=%d V=%d Smax=%d\n",
        c.H, c.F, c.Nl, c.Nh, c.Nkv, c.Hd, c.V, c.Smax);
    std::printf("Weights: 1.58 bits/elem packed (4 trits/byte)\n");

    std::mt19937 rng(42);
    Model m;
    size_t total_bytes = 0;
    alloc_model(m, c, rng, &total_bytes);
    Scratch s; alloc_scratch(s, c);

    std::printf("Total VRAM for weights: %.2f MB (vs INT8 baseline ~1180 MB)\n",
        total_bytes / (1024.0*1024.0));

    for (int t = 0; t < n_warmup; ++t) forward_token(m, s, 1, t);
    CK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    int tok = 1;
    for (int t = 0; t < n_gen; ++t) tok = forward_token(m, s, tok, n_warmup + t);
    CK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_tok = ms / n_gen;
    std::printf("\n=== ter sim packed-trit forward (1.58b weights, end-to-end) ===\n");
    std::printf("n_gen tokens : %d\n", n_gen);
    std::printf("total ms     : %.3f\n", ms);
    std::printf("ms / token   : %.4f\n", per_tok);
    std::printf("tokens/s     : %.1f\n", 1000.0 / per_tok);
    std::printf("\nReference:\n");
    std::printf("  ter sim INT8 forward (random weights, 1180 MB) : 14.7 t/s = 68.2 ms/token\n");
    std::printf("  llama.cpp Q8_0 fp16-TC                          : 130 t/s = 7.7 ms/token\n");

    return 0;
}
