#include "bonus.h"
#include "bench.h" // 包含项目通用的 debug_log 等
#include <random>
#include <cmath>
#include <iostream>

// ---------------------------------------------------------
// 极限优化区 (Pedal to the metal!)
// ---------------------------------------------------------

// 如果 CMake 找到了 BLAS，引入 C 接口头文件
#ifdef BONUS_USE_BLAS
extern "C" {
    #include <cblas.h>
}
#else
    #include <immintrin.h> // AVX2 & FMA 支持
    #ifdef _OPENMP
        #include <omp.h>   // OpenMP 多线程支持
    #endif
#endif

// 学生的极致优化版
void stu_bonus(std::span<float> C, std::span<const float> A, std::span<const float> B, int M, int N, int K) {
#ifdef BONUS_USE_BLAS
    // 方案 A：降维打击 (直接调用 OpenBLAS)
    // 性能可以达到 CPU 理论峰值的 90% 以上
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f,
                A.data(), K,
                B.data(), N,
                0.0f, C.data(), N);
#else
    // 方案 B：手写极限压榨 (转置 + OMP + AVX2 + FMA)
    
    // 1. 内存重排 (Transpose B)
    // 为什么？因为矩阵乘法中 B 是按列访问的，会导致极其严重的 Cache Miss。
    // 转置后，变成按行访问，对 CPU 缓存极度友好！
    std::vector<float> B_T(K * N);
    
    #pragma omp parallel for collapse(2) if(K*N > 10000)
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            B_T[j * K + k] = B[k * N + j];
        }
    }

    // 2. 多线程 + FMA 并行计算
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // 初始化 AVX 寄存器为 0
            __m256 sum_vec = _mm256_setzero_ps();
            int k = 0;
            
            // 每次循环处理 8 个 float (256 bits)
            for (; k + 7 < K; k += 8) {
                // 加载 A 的一行 和 B_T 的一行 (原 B 的一列)
                __m256 a_vec = _mm256_loadu_ps(&A[i * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&B_T[j * K + k]);
                
                // 乘加指令 (FMA): sum_vec = a_vec * b_vec + sum_vec
                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            }
            
            // 将 256 位寄存器中的 8 个结果水平相加
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, sum_vec);
            float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + 
                        tmp[4] + tmp[5] + tmp[6] + tmp[7];

            // 处理尾部不足 8 个的数据
            for (; k < K; ++k) {
                sum += A[i * K + k] * B_T[j * K + k];
            }
            
            C[i * N + j] = sum;
        }
    }
#endif
}

// ---------------------------------------------------------
// 辅助代码区 (基准实现与测试框架)
// ---------------------------------------------------------

// Naive 参考实现：最基础的三重循环 $O(N^3)$
void naive_bonus(std::span<float> C, std::span<const float> A, std::span<const float> B, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 初始化测试数据
void initialize_bonus(bonus_args *args, int M, int N, int K) {
    if (!args) return;
    args->M = M;
    args->N = N;
    args->K = K;

    args->A.resize(M * K);
    args->B.resize(K * N);
    args->C.resize(M * N, 0.0f);
    args->ref_C.resize(M * N, 0.0f);

    std::mt19937 gen(42); // 固定种子以保证每次运行一致
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto &val : args->A) val = dist(gen);
    for (auto &val : args->B) val = dist(gen);
}

void naive_bonus_wrapper(void *ctx) {
    auto &args = *static_cast<bonus_args *>(ctx);
    naive_bonus(args.ref_C, args.A, args.B, args.M, args.N, args.K);
}

void stu_bonus_wrapper(void *ctx) {
    auto &args = *static_cast<bonus_args *>(ctx);
    stu_bonus(args.C, args.A, args.B, args.M, args.N, args.K);
}

// 精度校验逻辑
bool bonus_check(void *stu_ctx, void *ref_ctx, void (*naive_func)(void *)) {
    // 运行参考函数
    if (naive_func) {
        naive_func(ref_ctx);
    }

    auto &stu_args = *static_cast<bonus_args *>(stu_ctx);
    auto &ref_args = *static_cast<bonus_args *>(ref_ctx);

    if (stu_args.C.size() != ref_args.ref_C.size()) return false;

    float max_diff = 0.0f;
    for (size_t i = 0; i < ref_args.ref_C.size(); ++i) {
        float r = ref_args.ref_C[i];
        float s = stu_args.C[i];
        float diff = std::abs(r - s);
        
        // 浮点数计算允许微小的精度误差 (由于 FMA 和顺序改变)
        if (diff > 1e-4f) {
            debug_log("\tDEBUG: fail at index {}: ref={} stu={} diff={}\n", i, r, s, diff);
            return false;
        }
        max_diff = std::max(max_diff, diff);
    }
    
    debug_log("\tDEBUG: bonus_check passed. max_diff=%f\n", max_diff);
    return true;
}