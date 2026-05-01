#ifndef GRAPH_H
#define GRAPH_H

#include "bench.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

const std::chrono::nanoseconds BASELINE_GRAPH{5000000};

struct Edge {
    int to;
    Edge* next;
};

struct Node {
    Edge* edges;
};

struct Graph {
    int n;
    Node* nodes;
};

// 1. 新增：缓存友好的 CSR 格式图结构
struct CSRGraph {
    int n;
    const int* row_ptr;
    const int* col_idx;
};

struct graph_args {
    Graph graph;
    std::vector<Node> nodes;
    std::vector<Edge> edge_storage;
    std::uint64_t out;
    double epsilon;
    
    // 2. 在末尾追加 CSR 数据存储容器
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_idx;
    CSRGraph csr_graph;

    explicit graph_args(double epsilon_in = 1e-6)
        : graph{0, nullptr}, out{0}, epsilon{epsilon_in}, csr_graph{0, nullptr, nullptr} {}
};

void naive_graph(std::uint64_t& out, const Graph& graph);

// 3. 修改 stu_graph 签名：接收你的 CSRGraph
void stu_graph(std::uint64_t& out, const CSRGraph& graph);

void naive_graph_wrapper(void* ctx);
void stu_graph_wrapper(void* ctx);

void initialize_graph(graph_args* args,
                       std::size_t node_count,
                       int avg_degree,
                       std::uint_fast64_t seed);

bool graph_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func);

#endif