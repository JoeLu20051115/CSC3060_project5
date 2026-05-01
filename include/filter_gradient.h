#ifndef FILTER_GRADIENT_H
#define FILTER_GRADIENT_H

#include "bench.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

inline constexpr std::chrono::nanoseconds BASELINE_FILTER_GRADIENT{25000000};

struct data_struct {
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    std::vector<float> d;
    std::vector<float> e;
    std::vector<float> f;
    std::vector<float> g;
    std::vector<float> h;
    std::vector<float> i;
};

// 1. 定义缓存友好的新结构体 (AoS)
struct PixelAoS {
    float a, b, c, d, e, f, g, h, i;
};

struct filter_gradient_args {
    data_struct data; 
    
    std::size_t width;
    std::size_t height;
    float out;
    double epsilon;

    // 2. 将预转换的新结构加入到 benchmark 上下文（不计入时间）
    std::vector<PixelAoS> aos_data;

    explicit filter_gradient_args(double epsilon_in = 1e-6)
        : width(0), height(0), out(0.0f), epsilon(epsilon_in) {}
};

void naive_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height);

// 3. 修改 stu 签名：使用新结构
void stu_filter_gradient(float& out, const std::vector<PixelAoS>& data,
                   std::size_t width, std::size_t height);

void naive_filter_gradient_wrapper(void* ctx);
void stu_filter_gradient_wrapper(void* ctx);

void initialize_filter_gradient(filter_gradient_args* args,
                        std::size_t width,
                        std::size_t height,
                        std::uint_fast64_t seed);

bool filter_gradient_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func);

#endif // FILTER_GRADIENT_H