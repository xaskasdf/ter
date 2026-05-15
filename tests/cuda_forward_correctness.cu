// test_cuda_forward_correctness.cu -- validate that mm_packed_v4 (the v11
// warp-coop kernel inside ter_cuda_forward_packed.cu) computes matmul
// correctly. Random ternary W (col-major packed) × random int8 X compared
// against a scalar reference dot product.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>

#define CK(call) do { cudaError_t e=(call); if(e){std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);} } while(0)

// Verbatim copy of mm_packed_v4 from ter_cuda_forward_packed.cu
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
        float scale = scale_smem;
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub)
            if (j_base + sub < N)
                out[j_base + sub] = __float2half((float)acc[sub] * scale);
    }
}

struct Shape { const char* name; int K; int N; };

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n", prop.name);

    Shape shapes[] = {
        {"tiny",     128,    64},
        {"medium",  2048,   512},
        {"big",     2048,  8192},
        {"huge",    8192,  2048},
    };

    std::mt19937 rng(42);
    int total_pass = 0, total_fail = 0;

    for (auto& s : shapes) {
        int K = s.K, N = s.N;
        float scale_x = 0.0312f, w_scale = 0.05f;

        std::uniform_int_distribution<int> dx(-127, 127);
        std::uniform_int_distribution<int> dt(0, 2);

        // Random X (int8), random ternary codes
        std::vector<int8_t> hX(K);
        for (auto& v : hX) v = (int8_t)dx(rng);
        std::vector<int> codes((size_t)K * N);
        for (auto& c : codes) c = dt(rng);

        // Pack W col-major: W_col[j_byte * K + k]
        size_t Wp = (size_t)K * (N / 4);
        std::vector<uint8_t> hW_col(Wp, 0);
        for (int jb = 0; jb < N/4; ++jb)
            for (int k = 0; k < K; ++k) {
                uint8_t b = 0;
                for (int t = 0; t < 4; ++t) {
                    int c = codes[(size_t)k * N + jb * 4 + t];
                    b |= (c & 0x3) << (t * 2);
                }
                hW_col[(size_t)jb * K + k] = b;
            }

        // Reference: scalar dot product
        std::vector<float> ref(N);
        for (int j = 0; j < N; ++j) {
            int acc = 0;
            for (int k = 0; k < K; ++k) {
                int wv = (codes[(size_t)k * N + j] == 1) ? 1
                       : (codes[(size_t)k * N + j] == 2) ? -1 : 0;
                acc += (int)hX[k] * wv;
            }
            ref[j] = (float)acc * scale_x * w_scale;
        }

        // Device buffers
        int8_t *dX; uint8_t *dW; __half *dOut; float *dScale;
        CK(cudaMalloc(&dX, K)); CK(cudaMalloc(&dW, Wp));
        CK(cudaMalloc(&dOut, N * sizeof(__half)));
        CK(cudaMalloc(&dScale, sizeof(float)));
        CK(cudaMemcpy(dX, hX.data(), K, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW, hW_col.data(), Wp, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dScale, &scale_x, sizeof(float), cudaMemcpyHostToDevice));

        // Launch v11 kernel
        int t = 256;
        int blocks = (N / 4 + 8 - 1) / 8;
        mm_packed_v4<<<blocks, t>>>(dX, dW, K, N, dScale, w_scale, dOut);
        CK(cudaDeviceSynchronize());

        // Compare
        std::vector<__half> hOut(N);
        CK(cudaMemcpy(hOut.data(), dOut, N * sizeof(__half), cudaMemcpyDeviceToHost));

        float max_abs = 0, max_rel = 0;
        int n_bad = 0;
        for (int j = 0; j < N; ++j) {
            float kernel_v = __half2float(hOut[j]);
            float diff = fabsf(kernel_v - ref[j]);
            float rel = diff / (fabsf(ref[j]) + 1e-6f);
            if (diff > max_abs) max_abs = diff;
            if (rel > max_rel) max_rel = rel;
            // tolerance: fp16 precision is ~1e-3 rel, plus some accumulation
            if (rel > 0.01f && diff > 0.01f) n_bad++;
        }

        bool pass = (n_bad == 0);
        std::printf("%-8s K=%d N=%d  max_abs=%.5f max_rel=%.5f  %s (%d bad of %d)\n",
            s.name, K, N, max_abs, max_rel,
            pass ? "PASS" : "FAIL", n_bad, N);

        if (pass) ++total_pass; else ++total_fail;

        cudaFree(dX); cudaFree(dW); cudaFree(dOut); cudaFree(dScale);
    }

    std::printf("\nResult: %d PASS, %d FAIL\n", total_pass, total_fail);
    return total_fail > 0 ? 1 : 0;
}
