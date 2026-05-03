#include "matmul.h"
#include <immintrin.h>
#include <omp.h>
#include <algorithm>

void stu_matmul(matmul_args& args) {
    float* A = args.A.data();
    float* B = args.B.data();
    float* C = args.C.data();
    int n = args.n;

    // 针对服务器 L3 Cache 的分块
    const int BLOCK_SIZE = 128; 

    // 加速密码：手动控制并行域，减少 OpenMP 启动开销
    #pragma omp parallel
    {
        // 动态获取线程信息，用于负载均衡
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        // 这种分块方式确保了 A 的行在不同线程间是隔离的，避免 C 矩阵的 Cache Line 伪共享
        #pragma omp for collapse(2) schedule(static)
        for (int bi = 0; bi < n; bi += BLOCK_SIZE) {
            for (int bj = 0; bj < n; bj += BLOCK_SIZE) {
                for (int bk = 0; bk < n; bk += BLOCK_SIZE) {
                    
                    // 内部 Micro-kernel
                    for (int i = bi; i < std::min(bi + BLOCK_SIZE, n); i += 8) {
                        for (int j = bj; j < std::min(bj + BLOCK_SIZE, n); j += 16) {
                            
                            // 预取（Prefetch）：告诉 CPU 下一波要用的数据
                            _mm_prefetch((const char*)&B[bk * n + j], _MM_HINT_T0);

                            __m512 c0 = _mm512_loadu_ps(&C[(i + 0) * n + j]);
                            __m512 c1 = _mm512_loadu_ps(&C[(i + 1) * n + j]);
                            __m512 c2 = _mm512_loadu_ps(&C[(i + 2) * n + j]);
                            __m512 c3 = _mm512_loadu_ps(&C[(i + 3) * n + j]);
                            __m512 c4 = _mm512_loadu_ps(&C[(i + 4) * n + j]);
                            __m512 c5 = _mm512_loadu_ps(&C[(i + 5) * n + j]);
                            __m512 c6 = _mm512_loadu_ps(&C[(i + 6) * n + j]);
                            __m512 c7 = _mm512_loadu_ps(&C[(i + 7) * n + j]);

                            for (int k = bk; k < std::min(bk + BLOCK_SIZE, n); ++k) {
                                __m512 b_vec = _mm512_loadu_ps(&B[k * n + j]);
                                
                                // AVX-512 FMA 配合广播优化
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i + 0) * n + k]), b_vec, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i + 1) * n + k]), b_vec, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i + 2) * n + k]), b_vec, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i + 3) * n + k]), b_vec, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i + 4) * n + k]), b_vec, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i + 5) * n + k]), b_vec, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i + 6) * n + k]), b_vec, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i + 7) * n + k]), b_vec, c7);
                            }

                            _mm512_storeu_ps(&C[(i + 0) * n + j], c0);
                            _mm512_storeu_ps(&C[(i + 1) * n + j], c1);
                            _mm512_storeu_ps(&C[(i + 2) * n + j], c2);
                            _mm512_storeu_ps(&C[(i + 3) * n + j], c3);
                            _mm512_storeu_ps(&C[(i + 4) * n + j], c4);
                            _mm512_storeu_ps(&C[(i + 5) * n + j], c5);
                            _mm512_storeu_ps(&C[(i + 6) * n + j], c6);
                            _mm512_storeu_ps(&C[(i + 7) * n + j], c7);
                        }
                    }
                }
            }
        }
    }
}