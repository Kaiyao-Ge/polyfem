// cl_kernels.hpp: kernels shared by the CL replay programs (copied from joint/stage0/L1_bench/l1_bench.cpp, which is frozen):
// HessianBlock, V0 (the current per-block algorithm: add_blocks of NonsmoothNewton.cpp), CSR / K_R row-accumulation assembly with the position map,
// Q_R gather, SpMM, SpMV, V1_val (scatter of new block values into the fixed CSR), cache flush, quantile, interleaving of arms.
#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using SpMatR = Eigen::SparseMatrix<double, Eigen::RowMajor>;

// ------------------------------------------------------------------------------------------------ the block type (copy of polysolve Problem::HessianBlock)
struct HessianBlock
{
    static constexpr int max_size = 12;
    std::array<int, max_size> indices;
    int size = 0;
    Eigen::Matrix<double, max_size, max_size, Eigen::DontAlign> H;
};

static double now_ns()
{
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::nano>(clk::now().time_since_epoch()).count();
}

// ------------------------------------------------------------------------------------------------ V0: the current algorithm (add_blocks in NonsmoothNewton.cpp)
template <bool RowMajorQ>
static void v0_add_blocks(Mat &A, const std::vector<HessianBlock> &blocks, const RowMat &Qr, const Mat &Q, const int k)
{
    Eigen::Matrix<double, HessianBlock::max_size, Eigen::Dynamic> LQ(HessianBlock::max_size, k);
    for (Eigen::Index ib = 0; ib < Eigen::Index(blocks.size()); ++ib)
    {
        const auto &b = blocks[ib];
        LQ.setZero();
        bool any = false;
        for (int i = 0; i < b.size; ++i)
            if (b.indices[i] >= 0)
            {
                if (RowMajorQ)
                    LQ.row(i) = Qr.block(b.indices[i], 0, 1, k);
                else
                    LQ.row(i) = Q.block(b.indices[i], 0, 1, k);
                any = true;
            }
        if (any)
            A.noalias() += LQ.topRows(b.size).transpose() * (b.H.topLeftCorner(b.size, b.size) * LQ.topRows(b.size));
    }
}

// ------------------------------------------------------------------------------------------------ V1: K_R and the merged triple product
struct CSR
{
    int nR = 0;
    std::vector<int> Rdof;             ///< local row -> global dof (ascending)
    std::vector<int> ptr, col;
    std::vector<double> val;
};

struct AsmWork
{
    std::vector<int> loc, occptr, occ, cursor, pos;
};

/// K_R by row accumulation: the occurrences (block, local row) are grouped by local row (counting sort), each row is summed with a marker array; columns unsorted within a row
static void assemble_rowacc(const std::vector<HessianBlock> &blocks, const int n, AsmWork &w, CSR &K, std::vector<int> *posmap = nullptr)
{
    if (posmap)
        posmap->assign(blocks.size() * 144, -1);
    w.loc.assign(n, -1);
    for (const auto &b : blocks)
        for (int i = 0; i < b.size; ++i)
            if (b.indices[i] >= 0)
                w.loc[b.indices[i]] = 0;
    K.Rdof.clear();
    int nR = 0;
    for (int d = 0; d < n; ++d)
        if (w.loc[d] >= 0)
        {
            w.loc[d] = nR++;
            K.Rdof.push_back(d);
        }
    K.nR = nR;
    w.occptr.assign(nR + 1, 0);
    for (const auto &b : blocks)
        for (int i = 0; i < b.size; ++i)
            if (b.indices[i] >= 0)
                ++w.occptr[w.loc[b.indices[i]] + 1];
    for (int r = 0; r < nR; ++r)
        w.occptr[r + 1] += w.occptr[r];
    w.occ.resize(w.occptr[nR]);
    w.cursor.assign(w.occptr.begin(), w.occptr.end() - 1);
    for (int ib = 0; ib < int(blocks.size()); ++ib)
    {
        const auto &b = blocks[ib];
        for (int i = 0; i < b.size; ++i)
            if (b.indices[i] >= 0)
                w.occ[w.cursor[w.loc[b.indices[i]]]++] = ib * 12 + i;
    }
    K.ptr.assign(nR + 1, 0);
    K.col.clear();
    K.val.clear();
    K.col.reserve(std::size_t(w.occptr[nR]) / 2);
    K.val.reserve(std::size_t(w.occptr[nR]) / 2);
    w.pos.assign(nR, -1);
    for (int r = 0; r < nR; ++r)
    {
        const std::size_t start = K.col.size();
        for (int o = w.occptr[r]; o < w.occptr[r + 1]; ++o)
        {
            const auto &b = blocks[w.occ[o] / 12];
            const int i = w.occ[o] % 12;
            for (int j = 0; j < b.size; ++j)
            {
                const int dj = b.indices[j];
                if (dj < 0)
                    continue;
                const int c = w.loc[dj];
                const int p = w.pos[c];
                int idx;
                if (p < 0)
                {
                    idx = int(K.col.size());
                    w.pos[c] = idx;
                    K.col.push_back(c);
                    K.val.push_back(b.H(i, j));
                }
                else
                {
                    idx = p;
                    K.val[p] += b.H(i, j);
                }
                if (posmap)
                    (*posmap)[std::size_t(w.occ[o] / 12) * 144 + i * 12 + j] = idx;
            }
        }
        for (std::size_t e = start; e < K.col.size(); ++e)
            w.pos[K.col[e]] = -1;
        K.ptr[r + 1] = int(K.col.size());
    }
}

/// K_R by Eigen triplets (setFromTriplets, sorts and sums); the same local numbering. Returns the matrix; Rdof / nR in K (ptr / col / val not filled)
static SpMatR assemble_triplets(const std::vector<HessianBlock> &blocks, const int n, AsmWork &w, CSR &K)
{
    w.loc.assign(n, -1);
    for (const auto &b : blocks)
        for (int i = 0; i < b.size; ++i)
            if (b.indices[i] >= 0)
                w.loc[b.indices[i]] = 0;
    K.Rdof.clear();
    int nR = 0;
    for (int d = 0; d < n; ++d)
        if (w.loc[d] >= 0)
        {
            w.loc[d] = nR++;
            K.Rdof.push_back(d);
        }
    K.nR = nR;
    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(blocks.size() * 144);
    for (const auto &b : blocks)
        for (int i = 0; i < b.size; ++i)
            if (b.indices[i] >= 0)
                for (int j = 0; j < b.size; ++j)
                    if (b.indices[j] >= 0)
                        trip.emplace_back(w.loc[b.indices[i]], w.loc[b.indices[j]], b.H(i, j));
    SpMatR M(nR, nR);
    M.setFromTriplets(trip.begin(), trip.end());
    return M;
}

static void gather_QR(const CSR &K, const RowMat &Qr, const int k, RowMat &QR)
{
    QR.resize(K.nR, k);
    for (int r = 0; r < K.nR; ++r)
        std::memcpy(&QR(r, 0), &Qr(K.Rdof[r], 0), sizeof(double) * k);
}

static void spmm(const int *ptr, const int *col, const double *val, const int nR, const double *__restrict Q, double *__restrict W, const int k)
{
    for (int r = 0; r < nR; ++r)
    {
        double *__restrict w = W + std::size_t(r) * k;
        for (int j = 0; j < k; ++j)
            w[j] = 0.0;
        for (int e = ptr[r]; e < ptr[r + 1]; ++e)
        {
            const double v = val[e];
            const double *__restrict q = Q + std::size_t(col[e]) * k;
            for (int j = 0; j < k; ++j)
                w[j] += v * q[j];
        }
    }
}

static void spmv(const int *ptr, const int *col, const double *val, const int nR, const double *__restrict q, double *__restrict y)
{
    for (int r = 0; r < nR; ++r)
    {
        double s = 0;
        for (int e = ptr[r]; e < ptr[r + 1]; ++e)
            s += val[e] * q[col[e]];
        y[r] = s;
    }
}

static std::vector<double> g_flush;
static volatile double g_sink = 0;
static void flush_cache()
{
    if (g_flush.empty())
        g_flush.assign(std::size_t(16) << 20, 1.0); // 128 MB: 4 x the L3 of this machine
    for (auto &x : g_flush)
        x += 1.0;
    g_sink = g_flush[12345];
}

static void v1_val(const std::vector<HessianBlock> &blocks, const std::vector<int> &posmap, std::vector<double> &val)
{
    std::fill(val.begin(), val.end(), 0.0);
    for (std::size_t ib = 0; ib < blocks.size(); ++ib)
    {
        const auto &b = blocks[ib];
        const int *pm = &posmap[ib * 144];
        for (int i = 0; i < b.size; ++i)
            if (b.indices[i] >= 0)
                for (int j = 0; j < b.size; ++j)
                    if (b.indices[j] >= 0)
                        val[pm[i * 12 + j]] += b.H(i, j);
    }
}

/// refreshed blocks: A += L^T (H - H_ref) L for every block of the batch
static double quantile(std::vector<double> t, double q)
{
    std::sort(t.begin(), t.end());
    const double x = q * (t.size() - 1);
    const std::size_t i = std::size_t(x);
    return i + 1 < t.size() ? t[i] + (x - i) * (t[i + 1] - t[i]) : t[i];
}

static void interleave(std::vector<std::function<void(int)>> &arms, const int reps)
{
    const int na = int(arms.size());
    for (int r = -2; r < reps; ++r)
        for (int a = 0; a < na; ++a)
            arms[(a + (r < 0 ? 0 : r)) % na](r);
}


static double thread_cpu_ns()
{
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return double(ts.tv_sec) * 1e9 + double(ts.tv_nsec);
}

struct Rng
{
    std::mt19937_64 g;
    explicit Rng(std::uint64_t s) : g(s) {}
    double normal() { return std::normal_distribution<double>(0.0, 1.0)(g); }
    int below(int n) { return int(std::uniform_int_distribution<int>(0, n - 1)(g)); }
};

static Mat random_sym(const int k, Rng &rng)
{
    Mat G(k, k);
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
            G(i, j) = rng.normal();
    return Mat(0.5 * (G + G.transpose()));
}

/// median / quartiles of a sample
struct Quart
{
    double med = 0, q1 = 0, q3 = 0, mn = 0, mx = 0;
    int n = 0;
};
static Quart quart_of(const std::vector<double> &t)
{
    Quart q;
    q.n = int(t.size());
    if (t.empty())
        return q;
    q.med = quantile(t, 0.5);
    q.q1 = quantile(t, 0.25);
    q.q3 = quantile(t, 0.75);
    q.mn = *std::min_element(t.begin(), t.end());
    q.mx = *std::max_element(t.begin(), t.end());
    return q;
}

static std::string loadavg_line()
{
    std::ifstream f("/proc/loadavg");
    std::string l;
    std::getline(f, l);
    return l;
}
