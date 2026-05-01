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
    // 手动循环展开 4 路
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float val_a0 = a[i], val_b0 = b[i];
        float val_a1 = a[i+1], val_b1 = b[i+1];
        float val_a2 = a[i+2], val_b2 = b[i+2];
        float val_a3 = a[i+3], val_b3 = b[i+3];

        float ab0 = val_a0 * val_b0;
        float ab1 = val_a1 * val_b1;
        float ab2 = val_a2 * val_b2;
        float ab3 = val_a3 * val_b3;

        float g0 = 0.5f * ((ab0 / (1.0f + std::abs(ab0))) + 1.0f);
        float g1 = 0.5f * ((ab1 / (1.0f + std::abs(ab1))) + 1.0f);
        float g2 = 0.5f * ((ab2 / (1.0f + std::abs(ab2))) + 1.0f);
        float g3 = 0.5f * ((ab3 / (1.0f + std::abs(ab3))) + 1.0f);

        float a_p0 = val_a0 + g0;
        float a_p1 = val_a1 + g1;
        float a_p2 = val_a2 + g2;
        float a_p3 = val_a3 + g3;

        a_prime[i] = a_p0;
        a_prime[i+1] = a_p1;
        a_prime[i+2] = a_p2;
        a_prime[i+3] = a_p3;

        b_prime_base[i] = val_b0 * (1.0f - g0);
        b_prime_base[i+1] = val_b1 * (1.0f - g1);
        b_prime_base[i+2] = val_b2 * (1.0f - g2);
        b_prime_base[i+3] = val_b3 * (1.0f - g3);

        sum_a += a_p0 + a_p1 + a_p2 + a_p3;
    }
    // 处理剩余元素
    for (; i < n; ++i) {
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
    // 手动循环展开 4 路
    i = 0;
    for (; i + 3 < n; i += 4) {
        float curr_a_p0 = a_prime[i];
        float curr_a_p1 = a_prime[i+1];
        float curr_a_p2 = a_prime[i+2];
        float curr_a_p3 = a_prime[i+3];

        float smooth_a0 = (i == 0) ? curr_a_p0 : (curr_a_p0 + prev_a_p) * 0.5f;
        float smooth_a1 = (curr_a_p1 + curr_a_p0) * 0.5f;
        float smooth_a2 = (curr_a_p2 + curr_a_p1) * 0.5f;
        float smooth_a3 = (curr_a_p3 + curr_a_p2) * 0.5f;
        prev_a_p = curr_a_p3;

        float denom0 = 1.0f + std::abs(smooth_a0);
        float denom1 = 1.0f + std::abs(smooth_a1);
        float denom2 = 1.0f + std::abs(smooth_a2);
        float denom3 = 1.0f + std::abs(smooth_a3);

        float b_p0 = b_prime_base[i] * avg_a;
        float b_p1 = b_prime_base[i+1] * avg_a;
        float b_p2 = b_prime_base[i+2] * avg_a;
        float b_p3 = b_prime_base[i+3] * avg_a;

        float c_p0 = c[i] + (smooth_a0 / denom0);
        float c_p1 = c[i+1] + (smooth_a1 / denom1);
        float c_p2 = c[i+2] + (smooth_a2 / denom2);
        float c_p3 = c[i+3] + (smooth_a3 / denom3);

        float h0 = smooth_a0 * c_p0;
        float h1 = smooth_a1 * c_p1;
        float h2 = smooth_a2 * c_p2;
        float h3 = smooth_a3 * c_p3;

        float e0 = (h0 + b_p0) / denom0;
        float e1 = (h1 + b_p1) / denom1;
        float e2 = (h2 + b_p2) / denom2;
        float e3 = (h3 + b_p3) / denom3;

        out[i] = std::max(c_p0 - e0, 0.0f);
        out[i+1] = std::max(c_p1 - e1, 0.0f);
        out[i+2] = std::max(c_p2 - e2, 0.0f);
        out[i+3] = std::max(c_p3 - e3, 0.0f);
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        float curr_a_p = a_prime[i];
        float smooth_a = (i == 0) ? curr_a_p : (curr_a_p + prev_a_p) * 0.5f;
        prev_a_p = curr_a_p;

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