#include "grff.h"
#include <algorithm>
#include <cmath>
#include <random>

void initialize_grff(grff_args *args, const size_t size, const std::uint_fast64_t seed) {
    if (!args) return;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    args->a_features.resize(size);
    args->b_features.resize(size);
    args->c_features.resize(size);
    args->f_output.resize(size);

    // 为我们新增的缓存分配空间
    args->buf_a_prime.resize(size);
    args->buf_b_prime_base.resize(size);

    for (size_t i = 0; i < size; ++i) {
        args->a_features[i] = dist(gen);
        args->b_features[i] = dist(gen);
        args->c_features[i] = dist(gen);
    }
}

// -------------------------------------------------------------------------
// Naive Implementation
// -------------------------------------------------------------------------
void naive_grff(grff_args& args) {
    size_t n = args.a_features.size();
    std::vector<float> G(n), A_prime(n), Smooth_A(n), B_prime(n), C_prime(n), H(n), E(n);

    for (size_t i = 0; i < n; ++i) G[i] = 0.5f * ((args.a_features[i] * args.b_features[i]) / (1.0f + std::abs(args.a_features[i] * args.b_features[i])) + 1.0f);
    for (size_t i = 0; i < n; ++i) A_prime[i] = args.a_features[i] + G[i];
    
    float sum_a = 0.0f;
    for (size_t i = 0; i < n; ++i) sum_a += A_prime[i];
    float avg_a = sum_a / static_cast<float>(n);

    Smooth_A[0] = A_prime[0];
    for (size_t i = 1; i < n; ++i) Smooth_A[i] = (A_prime[i] + A_prime[i-1]) * 0.5f; 
    for (size_t i = 0; i < n; ++i) B_prime[i] = args.b_features[i] * (1.0f - G[i]) * avg_a;
    for (size_t i = 0; i < n; ++i) C_prime[i] = args.c_features[i] + (Smooth_A[i] / (1.0f + std::abs(Smooth_A[i])));
    for (size_t i = 0; i < n; ++i) H[i] = Smooth_A[i] * C_prime[i];
    for (size_t i = 0; i < n; ++i) E[i] = (H[i] + B_prime[i]) / (1.0f + std::abs(Smooth_A[i]));
    
    for (size_t i = 0; i < n; ++i) {
        float result = C_prime[i] - E[i];
        args.f_output[i] = std::max(result, 0.0f);
    }
}

// -------------------------------------------------------------------------
// Optimized Student Implementation (Loop Fusion)
// -------------------------------------------------------------------------
void stu_grff(grff_args& args) {
    const size_t n = args.a_features.size();
    const float* a = args.a_features.data();
    const float* b = args.b_features.data();
    const float* c = args.c_features.data();
    
    float* a_prime = args.buf_a_prime.data();
    float* b_prime_base = args.buf_b_prime_base.data();
    float* out = args.f_output.data();

    float sum_a = 0.0f;

    // 【第一趟融合循环】：融合了原版的 Stage 1, 2，并计算 Stage 3 需要的 sum
    // 同时提前算好 Stage 5 需要的 b * (1 - G) 部分，防止第二趟再次读取 B 数组
    #pragma GCC unroll 4
    for (size_t i = 0; i < n; ++i) {
        float val_a = a[i];
        float val_b = b[i];
        float ab = val_a * val_b;
        
        float g = 0.5f * ((ab / (1.0f + std::abs(ab))) + 1.0f);
        float a_p = val_a + g;
        
        a_prime[i] = a_p;
        b_prime_base[i] = val_b * (1.0f - g);
        sum_a += a_p;
    }

    // 全局屏障：计算 average
    const float avg_a = sum_a / static_cast<float>(n);

    // 维护一个前驱变量，用于计算 Smooth_A[i]
    float prev_a_p = a_prime[0];

    // 【第二趟融合循环】：融合了原版的 Stage 4, 5, 6, 7, 8, 9
    // 直接复用寄存器数据，不产生多余的中间向量写入
    #pragma GCC unroll 4
    for (size_t i = 0; i < n; ++i) {
        float curr_a_p = a_prime[i];
        float smooth_a = (i == 0) ? curr_a_p : (curr_a_p + prev_a_p) * 0.5f;
        prev_a_p = curr_a_p; // 更新滑动窗口

        float abs_smooth_a = std::abs(smooth_a);
        float denom = 1.0f + abs_smooth_a;
        
        float b_p = b_prime_base[i] * avg_a;
        float c_p = c[i] + (smooth_a / denom);
        
        float h = smooth_a * c_p;
        float e = (h + b_p) / denom;
        
        out[i] = std::max(c_p - e, 0.0f);
    }
}

// -------------------------------------------------------------------------
// Wrappers and Checker
// -------------------------------------------------------------------------
void naive_grff_wrapper(void *ctx) {
    auto &args = *static_cast<grff_args *>(ctx);
    naive_grff(args);
}

void stu_grff_wrapper(void *ctx) {
    auto &args = *static_cast<grff_args *>(ctx);
    stu_grff(args);
}

bool grff_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);
    auto &stu_args = *static_cast<grff_args *>(stu_ctx);
    auto &ref_args = *static_cast<grff_args *>(ref_ctx);
    const auto eps = ref_args.epsilon;
    const double atol = 1e-6;

    if (stu_args.f_output.size() != ref_args.f_output.size()) return false;

    for (size_t i = 0; i < ref_args.f_output.size(); ++i) {
        double r = static_cast<double>(ref_args.f_output[i]);
        double s = static_cast<double>(stu_args.f_output[i]);
        double err = std::abs(s - r);

        if (err > (atol + eps * std::abs(r))) {
            debug_log("DEBUG: GRFF fail at %zu: ref=%f stu=%f\n", i, r, s);
            return false;
        }
    }
    return true;
}