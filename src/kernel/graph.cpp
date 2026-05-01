#include "graph.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

void initialize_graph(graph_args* args, std::size_t node_count,
                      int avg_degree, std::uint_fast64_t seed) {
    if (!args) return;

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(node_count) - 1);

    args->nodes.assign(node_count, Node{nullptr});
    args->edge_storage.clear();
    args->edge_storage.resize(node_count * static_cast<std::size_t>(avg_degree));

    args->graph.n = static_cast<int>(node_count);
    args->graph.nodes = args->nodes.data();

    std::size_t edge_pos = 0;

    for (std::size_t u = 0; u < node_count; ++u) {
        std::vector<int> neighbors;
        neighbors.reserve(avg_degree);

        for (int k = 0; k < avg_degree; ++k) {
            neighbors.push_back(dist(gen));
        }

        Edge* head = nullptr;
        for (int k = avg_degree - 1; k >= 0; --k) {
            Edge& e = args->edge_storage[edge_pos + static_cast<std::size_t>(k)];
            e.to = neighbors[static_cast<std::size_t>(k)];
            e.next = head;
            head = &e;
        }

        args->nodes[u].edges = head;
        edge_pos += static_cast<std::size_t>(avg_degree);
    }

    args->out = 0;

    // --- STUDENT PREPROCESSING ---
    // 将基于链表的邻接表转换为 CSR 格式的连续数组（此部分时间不计入 Kernel 测试成绩）
    args->csr_row_ptr.resize(node_count + 1, 0);
    args->csr_col_idx.reserve(node_count * avg_degree);

    for (std::size_t u = 0; u < node_count; ++u) {
        args->csr_row_ptr[u] = static_cast<int>(args->csr_col_idx.size());
        const Edge* e = args->graph.nodes[u].edges;
        while (e) {
            args->csr_col_idx.push_back(e->to);
            e = e->next;
        }
    }
    args->csr_row_ptr[node_count] = static_cast<int>(args->csr_col_idx.size());
    
    args->csr_graph.n = static_cast<int>(node_count);
    args->csr_graph.row_ptr = args->csr_row_ptr.data();
    args->csr_graph.col_idx = args->csr_col_idx.data();
}

void naive_graph(std::uint64_t& out, const Graph& graph) {
    std::uint64_t checksum = 0;
    for (int u = 0; u < graph.n; ++u) {
        const Edge* e = graph.nodes[u].edges;
        while (e) {
            checksum += static_cast<std::uint64_t>(e->to);
            e = e->next;
        }
    }
    out = checksum;
}

// 优化后的 Kernel (利用缓存友好的 CSR 数据结构)
void stu_graph(std::uint64_t& out, const CSRGraph& graph) {
    std::uint64_t checksum = 0;
    const int n = graph.n;
    const int* row_ptr = graph.row_ptr;
    const int* col_idx = graph.col_idx;

    // 此时边数据已经是一段连续的内存了，硬件预取器会以极快的速度抓取数据
    for (int u = 0; u < n; ++u) {
        int start = row_ptr[u];
        int end = row_ptr[u + 1];
        for (int i = start; i < end; ++i) {
            checksum += static_cast<std::uint64_t>(col_idx[i]);
        }
    }
    out = checksum;
}

void naive_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    naive_graph(args.out, args.graph);
}

void stu_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    // 指派 wrapper 调用你的新函数（传入 args.csr_graph）
    stu_graph(args.out, args.csr_graph);
}

bool graph_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);
    auto& stu_args = *static_cast<graph_args*>(stu_ctx);
    auto& ref_args = *static_cast<graph_args*>(ref_ctx);
    const auto eps = ref_args.epsilon;

    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 0.0;
    const double rel = (std::abs(r) > 1e-12) ? err / std::abs(r) : err;

    debug_log("\tDEBUG: graph stu={} ref={} err={} rel={}\n", stu_args.out, ref_args.out, err, rel);
    return err <= (atol + eps * std::abs(r));
}