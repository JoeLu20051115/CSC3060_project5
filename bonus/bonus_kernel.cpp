#include "bonus_kernel.h"
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <iostream>

void stu_bonus(std::span<float> C, std::span<const float> A, std::span<const float> B, int M, int N, int K) {
#ifdef BONUS_USE_BLAS
    // 如果有 OpenBLAS，直接调用
    #include <cblas.h>
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N);
#else
    float* a_ptr = const_cast<float*>(A.data());
    float* b_ptr = const_cast<float*>(B.data());
    float* c_ptr = C.data();

    // 关键优化 1：初始化 C 为 0。
    // 在广播累加模式中，我们直接在 C 上进行 FMA 操作。
    std::fill(C.begin(), C.end(), 0.0f);

    // 关键优化 2：OpenMP 任务划分。
    // 使用 schedule(dynamic, 1) 应对可能的负载不均，确保 40 核跑满。
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            // 关键优化 3：广播 A[i][k] 到 512 位寄存器（包含 16 个相同的 float）
            __m512 va = _mm512_set1_ps(a_ptr[i * K + k]);
            
            int j = 0;
            // 关键优化 4：AVX-512 内层循环（处理 N 维度）
            // 连续内存访问 B[k][j] 和 C[i][j]，对 Cache 极其友好，无需转置
            for (; j + 31 < N; j += 32) {
                // 展开 2 次以提升指令并行度 (ILP)
                __m512 vb1 = _mm512_loadu_ps(&b_ptr[k * N + j]);
                __m512 vc1 = _mm512_loadu_ps(&c_ptr[i * N + j]);
                _mm512_storeu_ps(&c_ptr[i * N + j], _mm512_fmadd_ps(va, vb1, vc1));

                __m512 vb2 = _mm512_loadu_ps(&b_ptr[k * N + j + 16]);
                __m512 vc2 = _mm512_loadu_ps(&c_ptr[i * N + j + 16]);
                _mm512_storeu_ps(&c_ptr[i * N + j + 16], _mm512_fmadd_ps(va, vb2, vc2));
            }

            for (; j + 15 < N; j += 16) {
                __m512 vb = _mm512_loadu_ps(&b_ptr[k * N + j]);
                __m512 vc = _mm512_loadu_ps(&c_ptr[i * N + j]);
                _mm512_storeu_ps(&c_ptr[i * N + j], _mm512_fmadd_ps(va, vb, vc));
            }

            // 标量收尾
            for (; j < N; ++j) {
                c_ptr[i * N + j] += a_ptr[i * K + k] * b_ptr[k * N + j];
            }
        }
    }
#endif
}

// ... 其余 initialize_bonus 和 bonus_check 保持逻辑不变 ...