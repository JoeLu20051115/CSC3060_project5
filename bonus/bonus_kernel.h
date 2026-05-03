#ifndef BONUS_KERNEL_H
#define BONUS_KERNEL_H

#include <span>
#include <vector>

// Bonus 参数结构体
struct bonus_args {
    int M, N, K;
    std::vector<float> A;      // M x K
    std::vector<float> B;      // K x N
    std::vector<float> C;      // M x N
    std::vector<float> ref_C;  // M x N
};

// 核心计算函数
void stu_bonus(std::span<float> C, std::span<const float> A, std::span<const float> B, int M, int N, int K);

// 包装与初始化接口
void initialize_bonus(bonus_args *args, int M, int N, int K);
void stu_bonus_wrapper(void *ctx);
void naive_bonus_wrapper(void *ctx);
bool bonus_check(void *stu_ctx, void *ref_ctx, void (*naive_func)(void *));

#endif