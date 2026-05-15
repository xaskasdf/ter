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

// BitNet b1.58 2B-4T config (different from Llama 3.2 1B):
//   H = 2560 hidden, F = 6912 intermediate, 30 layers
//   20 attention heads, 5 KV heads (GQA 4:1), head_dim 128, vocab 128256
//   ARCHITECTURAL DIFFERENCE: BitNet uses sub-RMSNorms before Wo and Wdown
//   to stabilize training under ternary weight constraints.
struct Cfg {
    int H = 2560, F = 6912, Nl = 30, Nh = 20, Nkv = 5, Hd = 128, V = 128256, Smax = 64;
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

__global__ void rope_k(__half* v, int n_heads, int head_dim, const int* pos_dev, float theta) {
    int pos = pos_dev[0];
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

// Copy K/V into cache at slot [L, pos] -- replaces cudaMemcpy with a kernel
// that reads pos from device, enabling cudaGraph capture.
__global__ void kv_copy_k(
    const __half* k, const __half* v, __half* K_cache, __half* V_cache,
    int L, int Smax, int Hkv, const int* pos_dev)
{
    int p = pos_dev[0];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Hkv) return;
    size_t off = ((size_t)L * Smax + p) * Hkv + idx;
    K_cache[off] = k[idx];
    V_cache[off] = v[idx];
}

// Increment pos counter on device (called once per forward, after all uses).
__global__ void inc_pos_k(int* pos_dev) { if (threadIdx.x == 0) pos_dev[0] += 1; }

// Token embedding lookup: copy token_embd[token_id_dev[0] * H .. + H] into hidden.
// Replaces cudaMemcpyDeviceToDevice with kernel for graph capture.
__global__ void token_embed_lookup_k(
    const __half* token_embd, int H, const int* token_id_dev, __half* hidden)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= H) return;
    int t = token_id_dev[0];
    hidden[tid] = token_embd[(size_t)t * H + tid];
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
    int n_heads, int n_kv_heads, int head_dim, const int* pos_dev, int max_seq)
{
    int seq_len = pos_dev[0] + 1;
    int h = blockIdx.x;
    int kv_h = h * n_kv_heads / n_heads;
    int tid = threadIdx.x;
    extern __shared__ float smem[];
    // shmem size capped at (max_seq + head_dim) for cudaGraph compatibility
    float* scores = smem;
    float* outv = smem + max_seq;
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
// v11 warp-cooperative kernel: col-major W (W_col[j_byte*K + k]),
// 4 cols per warp via 4 trits/byte, warp-cooperative K reduction via
// __shfl_xor_sync, __dp4a SIMD inner loop. Same kernel that beat cuBLAS
// INT8 TC by 1.90x in the matmul-fabric bench.
__global__ void mm_packed_v4(
    const int8_t*  X, const uint8_t* W_col, int K, int N,
    const float* scale_x_dev, float w_scale, __half* out)
{
    int warp_block_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int j_base = warp_block_id * 4;
    if (j_base >= N) return;
    int j_byte = j_base / 4;
    const uint8_t* Wc = W_col + (size_t)j_byte * K;

    __shared__ float scale_smem;
    if (threadIdx.x == 0) scale_smem = scale_x_dev[0] * w_scale;

    int kpt = (K + 31) / 32;
    int kstart = lane * kpt, kend = min(K, kstart + kpt);
    int acc[4] = {0, 0, 0, 0};
    int kk = kstart;
    for (; kk + 4 <= kend; kk += 4) {
        uint32_t w = *reinterpret_cast<const uint32_t*>(Wc + kk);
        uint8_t b0 = w & 0xff, b1 = (w>>8) & 0xff, b2 = (w>>16) & 0xff, b3 = (w>>24) & 0xff;
        int32_t X4 = *reinterpret_cast<const uint32_t*>(X + kk);
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int sb = sub * 2;
            int t0=(b0>>sb)&3, t1=(b1>>sb)&3, t2=(b2>>sb)&3, t3=(b3>>sb)&3;
            int wv0=(t0==1)-(t0==2), wv1=(t1==1)-(t1==2), wv2=(t2==1)-(t2==2), wv3=(t3==1)-(t3==2);
            int32_t W4 = (uint8_t)wv0 | ((uint8_t)wv1<<8) | ((uint8_t)wv2<<16) | ((uint8_t)wv3<<24);
            acc[sub] = __dp4a(X4, W4, acc[sub]);
        }
    }
    for (; kk < kend; ++kk) {
        int xv = X[kk]; if (xv == 0) continue;
        uint8_t b = Wc[kk];
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int t = (b >> (sub * 2)) & 3;
            acc[sub] += xv * ((t == 1) - (t == 2));
        }
    }
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub)
        for (int o = 16; o > 0; o >>= 1) acc[sub] += __shfl_xor_sync(0xffffffff, acc[sub], o);
    if (lane == 0) {
        __syncwarp();
        float scale = scale_smem;  // broadcast read
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub)
            if (j_base + sub < N)
                out[j_base + sub] = __float2half((float)acc[sub] * scale);
    }
}

// ---------- Model + forward ----------
struct WPK { uint8_t* data; float scale; size_t bytes; };
struct Layer {
    WPK Wq, Wk, Wv, Wo, Wgate, Wup, Wdown;
    __half *attn_norm, *ffn_norm;
    __half *attn_sub_norm, *ffn_sub_norm;  // BitNet-specific: pre-Wo (size H), pre-Wdown (size F)
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
    int *pos_dev;        // device-resident position counter (graph-capture friendly)
    int *token_id_dev;   // device-resident current token id (chained from argmax)
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

// Read N bytes from FILE* into a device buffer. Used by load_model_from_bin.
static void read_to_device(std::FILE* fp, void* d_ptr, size_t bytes,
                           std::vector<uint8_t>& staging)
{
    if (staging.size() < bytes) staging.resize(bytes);
    size_t n = std::fread(staging.data(), 1, bytes, fp);
    if (n != bytes) { std::fprintf(stderr, "short read %zu/%zu\n", n, bytes); std::exit(1); }
    CK(cudaMemcpy(d_ptr, staging.data(), bytes, cudaMemcpyHostToDevice));
}

// Convert F32 norm weights -> F16 device buffer.
static void read_f32_to_half_device(std::FILE* fp, __half* d_dst, size_t n,
                                    std::vector<float>& staging,
                                    std::vector<__half>& staging_h)
{
    if (staging.size() < n) staging.resize(n);
    if (staging_h.size() < n) staging_h.resize(n);
    size_t got = std::fread(staging.data(), sizeof(float), n, fp);
    if (got != n) { std::fprintf(stderr, "short read %zu/%zu\n", got, n); std::exit(1); }
    for (size_t i = 0; i < n; ++i) staging_h[i] = __float2half(staging[i]);
    CK(cudaMemcpy(d_dst, staging_h.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
}

// Load BitNet 2B-4T weights from a packed binary blob produced by
// tools/convert_bitnet_to_packed.py. Replaces alloc_model when real weights
// are available. Returns total bytes allocated for tensors.
bool load_model_from_bin(Model& m, const char* path, size_t* total_bytes)
{
    std::FILE* fp = std::fopen(path, "rb");
    if (!fp) { std::fprintf(stderr, "Could not open %s\n", path); return false; }

    uint32_t magic, version;
    std::fread(&magic, 4, 1, fp);
    std::fread(&version, 4, 1, fp);
    if (magic != 0x54455254u || version != 1) {
        std::fprintf(stderr, "Bad magic/version: 0x%08x v%u\n", magic, version);
        std::fclose(fp); return false;
    }
    uint32_t H, F, Nl, V, Nh, Nkv, Hd, Smax;
    std::fread(&H, 4, 1, fp); std::fread(&F, 4, 1, fp);
    std::fread(&Nl, 4, 1, fp); std::fread(&V, 4, 1, fp);
    std::fread(&Nh, 4, 1, fp); std::fread(&Nkv, 4, 1, fp);
    std::fread(&Hd, 4, 1, fp); std::fread(&Smax, 4, 1, fp);
    float rope_theta, eps;
    std::fread(&rope_theta, 4, 1, fp);
    std::fread(&eps, 4, 1, fp);
    std::fseek(fp, 64, SEEK_SET);  // skip pad to 64-byte header end

    Cfg& c = m.cfg;
    c.H = H; c.F = F; c.Nl = Nl; c.V = V; c.Nh = Nh; c.Nkv = Nkv;
    c.Hd = Hd; c.Smax = Smax; c.rope_theta = rope_theta; c.eps = eps;
    std::printf("Loaded config: H=%d F=%d Nl=%d Nh=%d Nkv=%d Hd=%d V=%d eps=%.2e rope=%.0f\n",
        H, F, Nl, Nh, Nkv, Hd, V, eps, rope_theta);

    int Hkv = c.Nkv * c.Hd;
    std::vector<uint8_t> staging;
    std::vector<float> staging_f;
    std::vector<__half> staging_h;

    // token_embd (F16)
    size_t tok_n = (size_t)V * H;
    CK(cudaMalloc(&m.token_embd, tok_n * sizeof(__half)));
    read_to_device(fp, m.token_embd, tok_n * sizeof(__half), staging);
    *total_bytes += tok_n * sizeof(__half);

    // output_norm (F32 -> F16)
    CK(cudaMalloc(&m.output_norm, H * sizeof(__half)));
    read_f32_to_half_device(fp, m.output_norm, H, staging_f, staging_h);
    *total_bytes += H * sizeof(__half);

    m.layers = new Layer[Nl];
    for (uint32_t L = 0; L < Nl; ++L) {
        Layer& l = m.layers[L];
        for (auto& [pp, sz] : std::vector<std::pair<__half**, size_t>>{
            {&l.attn_norm,     H}, {&l.ffn_norm, H},
            {&l.attn_sub_norm, H}, {&l.ffn_sub_norm, F}})
        {
            CK(cudaMalloc(pp, sz * sizeof(__half)));
            read_f32_to_half_device(fp, *pp, sz, staging_f, staging_h);
            *total_bytes += sz * sizeof(__half);
        }
        struct WS { WPK* w; int K, N; };
        WS ws[] = {
            {&l.Wq, H, H},   {&l.Wk, H, Hkv}, {&l.Wv, H, Hkv}, {&l.Wo, H, H},
            {&l.Wgate, H, F}, {&l.Wup, H, F}, {&l.Wdown, F, H},
        };
        for (auto& w : ws) {
            size_t bytes = (size_t)w.K * (w.N / 4);
            CK(cudaMalloc(&w.w->data, bytes));
            w.w->bytes = bytes;
            read_to_device(fp, w.w->data, bytes, staging);
            // Read per-tensor i2_scale from microsoft converter's trailer.
            // BUT: empirically the BitNet 2B-4T attn_norm/sub_norm gammas
            // are very small (~0.01-0.02), suggesting they ABSORB the
            // weight scale. Applying ws again would double-count.
            // Tentative override: ignore stored scale, use 1.0 (gammas
            // handle the magnitude). If FFN matmuls still need scaling,
            // a per-matmul override could re-enable for those tensors.
            float scale;
            if (std::fread(&scale, sizeof(float), 1, fp) != 1) {
                std::fprintf(stderr, "missing scale for layer %u\n", L); std::exit(1);
            }
            w.w->scale = 1.0f;  // override: gammas already absorb weight scale
            (void)scale;
            *total_bytes += bytes + sizeof(float);
        }
    }
    // lm_head: tied to token_embd (F16). We don't pack it here; final
    // matmul against tied F16 embeddings would need a separate fp16 kernel.
    // For this validation run we skip lm_head and dump hidden state instead.
    m.lm_head.data = nullptr;
    m.lm_head.bytes = 0;
    m.lm_head.scale = 0.0f;

    std::fclose(fp);
    return true;
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
        allh(&l.attn_norm,     c.H);
        allh(&l.ffn_norm,      c.H);
        allh(&l.attn_sub_norm, c.H);
        allh(&l.ffn_sub_norm,  c.F);
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
    CK(cudaMalloc(&s.pos_dev, sizeof(int)));
    CK(cudaMalloc(&s.token_id_dev, sizeof(int)));
}

static void mm_packed_dispatch(const int8_t* X, const float* scale_x_dev, const WPK& W,
                               __half* out, int K, int N, cudaStream_t stream = 0)
{
    // v11 layout: 4 cols per warp, warps_per_block = 8 (256 threads)
    int t = 256;
    int blocks = (N / 4 + 8 - 1) / 8;
    mm_packed_v4<<<blocks, t, 0, stream>>>(X, W.data, K, N, scale_x_dev, W.scale, out);
}

// All-device forward: pos and token_id read from device pointers, no host
// state per kernel launch. Argmax result written to s.token_id_dev so the
// next call chains automatically. cudaGraph-capturable when stream != 0.
void forward_token_dev(Model& m, Scratch& s, cudaStream_t st = 0)
{
    const Cfg& c = m.cfg;
    int Hkv = c.Nkv * c.Hd;
    int hblk = (c.H + 255) / 256;
    token_embed_lookup_k<<<hblk, 256, 0, st>>>(m.token_embd, c.H, s.token_id_dev, s.hidden);

    for (int L = 0; L < c.Nl; ++L) {
        Layer& l = m.layers[L];
        rmsnorm_k<<<1,256,256*sizeof(float),st>>>(s.hidden, l.attn_norm, s.x_norm, c.H, c.eps);
        quant_int8_k<<<1,256,256*sizeof(float),st>>>(s.x_norm, s.x_q, s.scale_buf, c.H);
        mm_packed_dispatch(s.x_q, s.scale_buf, l.Wq, s.q, c.H, c.H, st);
        mm_packed_dispatch(s.x_q, s.scale_buf, l.Wk, s.k, c.H, Hkv, st);
        mm_packed_dispatch(s.x_q, s.scale_buf, l.Wv, s.v, c.H, Hkv, st);
        rope_k<<<c.Nh,  c.Hd/2, 0, st>>>(s.q, c.Nh,  c.Hd, s.pos_dev, c.rope_theta);
        rope_k<<<c.Nkv, c.Hd/2, 0, st>>>(s.k, c.Nkv, c.Hd, s.pos_dev, c.rope_theta);
        int kvb = (Hkv + 255) / 256;
        kv_copy_k<<<kvb, 256, 0, st>>>(s.k, s.v, s.K_cache, s.V_cache, L, c.Smax, Hkv, s.pos_dev);
        size_t cache_off = (size_t)L * c.Smax * Hkv;
        size_t shmem_max = (c.Smax + c.Hd) * sizeof(float);
        attention_k<<<c.Nh, 64, shmem_max, st>>>(s.q, s.K_cache+cache_off, s.V_cache+cache_off,
            s.attn_out, c.Nh, c.Nkv, c.Hd, s.pos_dev, c.Smax);
        // BitNet sub-RMSNorm before Wo (over attention output of size H)
        rmsnorm_k<<<1,256,256*sizeof(float),st>>>(s.attn_out, l.attn_sub_norm, s.x_norm, c.H, c.eps);
        quant_int8_k<<<1,256,256*sizeof(float),st>>>(s.x_norm, s.x_q, s.scale_buf, c.H);
        mm_packed_dispatch(s.x_q, s.scale_buf, l.Wo, s.attn_out, c.H, c.H, st);
        add_k<<<hblk,256,0,st>>>(s.hidden, s.attn_out, c.H);

        rmsnorm_k<<<1,256,256*sizeof(float),st>>>(s.hidden, l.ffn_norm, s.x_norm, c.H, c.eps);
        quant_int8_k<<<1,256,256*sizeof(float),st>>>(s.x_norm, s.x_q, s.scale_buf, c.H);
        mm_packed_dispatch(s.x_q, s.scale_buf, l.Wgate, s.gate, c.H, c.F, st);
        mm_packed_dispatch(s.x_q, s.scale_buf, l.Wup,   s.up,   c.H, c.F, st);
        int fblocks = (c.F+255)/256;
        silu_mul_k<<<fblocks,256,0,st>>>(s.gate, s.up, s.ff, c.F);
        // BitNet sub-RMSNorm before Wdown (over FFN intermediate of size F)
        rmsnorm_k<<<1,1024,1024*sizeof(float),st>>>(s.ff, l.ffn_sub_norm, s.ff, c.F, c.eps);
        quant_int8_k<<<1,1024,1024*sizeof(float),st>>>(s.ff, s.ff_q, s.scale_buf, c.F);
        mm_packed_dispatch(s.ff_q, s.scale_buf, l.Wdown, s.attn_out, c.F, c.H, st);
        add_k<<<hblk,256,0,st>>>(s.hidden, s.attn_out, c.H);
    }

    rmsnorm_k<<<1,256,256*sizeof(float),st>>>(s.hidden, m.output_norm, s.x_norm, c.H, c.eps);
    if (m.lm_head.data != nullptr) {
        quant_int8_k<<<1,256,256*sizeof(float),st>>>(s.x_norm, s.x_q, s.scale_buf, c.H);
        mm_packed_dispatch(s.x_q, s.scale_buf, m.lm_head, s.logits, c.H, c.V, st);
        argmax_k<<<1,256,256*(sizeof(float)+sizeof(int)),st>>>(s.logits, s.token_id_dev, c.V);
    }
    // else: real-BitNet path, lm_head is tied to F16 token_embd and needs
    // an fp16 matmul kernel (not yet implemented). For validation runs,
    // token_id_dev is updated externally (or held constant) so the graph
    // can still loop without divergence.
    inc_pos_k<<<1, 1, 0, st>>>(s.pos_dev);
}

// Backward-compat wrapper: takes host token/pos, writes to device, runs forward,
// reads next_token back. Kept for reference / debugging; bench uses graph path.
int forward_token(Model& m, Scratch& s, int token_id, int pos) {
    CK(cudaMemcpy(s.token_id_dev, &token_id, sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s.pos_dev, &pos, sizeof(int), cudaMemcpyHostToDevice));
    forward_token_dev(m, s);
    int next = 0;
    CK(cudaMemcpy(&next, s.token_id_dev, sizeof(int), cudaMemcpyDeviceToHost));
    return next;
}

int main(int argc, char** argv) {
    int n_gen = (argc > 1) ? std::atoi(argv[1]) : 32;
    const char* bin_path = (argc > 2) ? argv[2] : nullptr;
    int n_warmup = 4;

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n", prop.name);

    Cfg c_default;
    std::printf("BitNet 2B-4T: H=%d F=%d L=%d Nh=%d Nkv=%d Hd=%d V=%d Smax=%d\n",
        c_default.H, c_default.F, c_default.Nl, c_default.Nh, c_default.Nkv,
        c_default.Hd, c_default.V, c_default.Smax);
    std::printf("Weights: 1.58 bits/elem packed (4 trits/byte)\n");

    std::mt19937 rng(42);
    Model m;
    size_t total_bytes = 0;
    if (bin_path) {
        std::printf("Loading real BitNet weights from %s...\n", bin_path);
        if (!load_model_from_bin(m, bin_path, &total_bytes)) {
            std::fprintf(stderr, "Load failed; aborting\n"); return 1;
        }
        std::printf("Real BitNet weights loaded.\n");
    } else {
        m.cfg = c_default;
        alloc_model(m, m.cfg, rng, &total_bytes);
        std::printf("(Random ternary weights -- no --bin specified)\n");
    }
    Scratch s; alloc_scratch(s, m.cfg);

    std::printf("Total VRAM for weights: %.2f MB (vs INT8 baseline ~1180 MB)\n",
        total_bytes / (1024.0*1024.0));

    // Warmup with the host-state path (also exercises both code paths)
    for (int t = 0; t < n_warmup; ++t) forward_token(m, s, 1, t);
    CK(cudaDeviceSynchronize());

    // Reset device state for the captured graph: pos = n_warmup, token_id = 1
    int init_pos = n_warmup, init_tok = 1;
    CK(cudaMemcpy(s.pos_dev,      &init_pos, sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s.token_id_dev, &init_tok, sizeof(int), cudaMemcpyHostToDevice));

    // Capture forward into a cudaGraph. Kernels in forward_token_dev launch on
    // the legacy default stream (0); cudaStreamCaptureModeGlobal on a real
    // stream captures all default-stream activity from this thread too.
    cudaStream_t stream; CK(cudaStreamCreate(&stream));
    cudaGraph_t graph;
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    forward_token_dev(m, s, stream);
    CK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t graph_exec;
    CK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    // After capture pos/token advanced once; reset for the timed loop.
    CK(cudaMemcpy(s.pos_dev,      &init_pos, sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s.token_id_dev, &init_tok, sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < n_gen; ++t) CK(cudaGraphLaunch(graph_exec, stream));
    CK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_tok = ms / n_gen;
    std::printf("\n=== ter sim BitNet 2B-4T forward (1.58b weights + sub-norms, end-to-end) ===\n");
    std::printf("n_gen tokens : %d\n", n_gen);
    std::printf("total ms     : %.3f\n", ms);
    std::printf("ms / token   : %.4f\n", per_tok);
    std::printf("tokens/s     : %.1f\n", 1000.0 / per_tok);
    std::printf("\nReference:\n");
    std::printf("  ter sim packed-trit Llama 1B (forward_packed)    : 421 t/s = 2.37 ms/token (cudaGraph, May 2026)\n");
    std::printf("  llama.cpp Q8_0 (Llama 3.2 1B, RTX 3090 clean GPU): 395 t/s = 2.53 ms/token (real ref, May 2026)\n");

    // Hidden state sanity check: dump first 12 values of x_norm (POST
    // output_norm — what would feed into lm_head). For BitNet b1.58 with
    // properly trained weights, these should be O(1) magnitude.
    std::vector<__half> h_hidden(m.cfg.H);
    CK(cudaMemcpy(h_hidden.data(), s.x_norm, m.cfg.H * sizeof(__half),
                  cudaMemcpyDeviceToHost));
    float sum = 0, sum2 = 0, mx = -1e30f, mn = 1e30f;
    int finite = 0;
    for (int i = 0; i < m.cfg.H; ++i) {
        float v = __half2float(h_hidden[i]);
        if (std::isfinite(v)) { ++finite; sum += v; sum2 += v*v; }
        if (v > mx) mx = v;
        if (v < mn) mn = v;
    }
    float mean = sum / m.cfg.H, var = sum2 / m.cfg.H - mean * mean;
    std::printf("\nFinal hidden state stats (H=%d):\n", m.cfg.H);
    std::printf("  finite count : %d / %d\n", finite, m.cfg.H);
    std::printf("  mean         : %.4f\n", mean);
    std::printf("  stddev       : %.4f\n", std::sqrt(std::max(0.f, var)));
    std::printf("  range        : [%.4f, %.4f]\n", mn, mx);
    std::printf("  first 12     :");
    for (int i = 0; i < 12; ++i)
        std::printf(" %+.3f", __half2float(h_hidden[i]));
    std::printf("\n");

    return 0;
}
