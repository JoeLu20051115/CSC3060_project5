#ifndef BONUS_H
#define BONUS_H

#include <span>
#include <vector>
#include <cstdint>

// Bonus 参数结构体
struct bonus_args {
    int M;
    int N;
    int K;
    // 矩阵数据：使用连续的 1D 数组模拟 2D 矩阵 (Row-Major)
    std::vector<float> A;      // M x K
    std::vector<float> B;      // K x N
    std::vector<float> C;      // M x N (Student 结果)
    std::vector<float> ref_C;  // M x N (Naive 结果，用于校验)
};

// 接口声明
void initialize_bonus(bonus_args *args, int M, int N, int K);
void naive_bonus_wrapper(void *ctx);
void stu_bonus_wrapper(void *ctx);
bool bonus_check(void *stu_ctx, void *ref_ctx, void (*naive_func)(void *));

// 核心计算函数声明
void stu_bonus(std::span<float> C, std::span<const float> A, std::span<const float> B, int M, int N, int K);

#endif // BONUS_H