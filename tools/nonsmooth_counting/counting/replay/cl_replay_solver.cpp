// cl_replay_solver: replay pricing programs that use the compile conditions of NonsmoothNewton.cpp (EIGEN_USE_MKL_ALL, SSE2, -O3) and CHOLMOD as the solver links it.
//   l4a  numeric factorization / analysis / back-solve of the exported matrices (successful and failed), CHOLMOD supernodal LLT like Eigen::CholmodSupernodalLLT
//   l4c  merged form on the exported real block tables and Q (cold and hot cache), arms interleaved
//   l4e  small operations (3 x 3 and 12 x 12 eigen decompositions, sorting, spmv, dot / axpy / gemv, dense k x k solves, row gathers)
// Output: TSV files in --out. Nothing here runs on its own: joint/cl/B/run_B_replay.sh calls it after the exports exist.
#include "cl_io.hpp"
#include "cl_kernels.hpp"

#include <Eigen/Eigenvalues>
#include <memory>
#include <cholmod.h>

// ================================================================================================ L4a
struct FactRec
{
    IndexRow ix;
    FactFile f;
    cholmod_sparse A;
    std::vector<double> an_cpu, f_cpu, s_cpu, s_cold_cpu, an_wall, f_wall, s_wall;
    double fl = -1, lnz = -1, fl_done = -1;
    long minor = -1;
    int success = 0, perm_ok = -1;
};

static void make_sparse(FactRec &r)
{
    std::memset(&r.A, 0, sizeof(cholmod_sparse));
    r.A.nrow = std::size_t(r.f.n);
    r.A.ncol = std::size_t(r.f.n);
    r.A.nzmax = std::size_t(r.f.nnz);
    r.A.p = r.f.outer.data();
    r.A.i = r.f.inner.data();
    r.A.x = r.f.values.data();
    r.A.stype = -1; // lower triangle, like Eigen's selfadjointView<Lower>
    r.A.itype = CHOLMOD_INT;
    r.A.xtype = CHOLMOD_REAL;
    r.A.dtype = CHOLMOD_DOUBLE;
    r.A.sorted = 1;
    r.A.packed = 1;
}

static int l4a(const std::vector<std::string> &dirs, const std::string &out, const int reps)
{
    std::vector<FactRec> recs;
    for (const auto &d : dirs)
        for (const auto &ix : read_index(d))
            if (ix.type == "fact")
            {
                FactRec r;
                r.ix = ix;
                r.f = read_fact(d + "/" + ix.file);
                recs.push_back(std::move(r));
            }
    for (auto &r : recs)
        make_sparse(r);
    std::printf("l4a: %zu factorization records\n", recs.size());
    cholmod_common c;
    cholmod_start(&c);
    c.final_asis = 1; // as CholmodSupernodalLLT (EigenSolver: CholmodSupernodalLLT -> final_asis = 1, supernodal = CHOLMOD_SUPERNODAL)
    c.supernodal = CHOLMOD_SUPERNODAL;
    c.print = 0;
    const double beta[2] = {0.0, 0.0};
    Rng rng(7);
    const int nrec = int(recs.size());
    for (int round = -1; round < reps; ++round) // round -1: warm-up
        for (int j = 0; j < nrec; ++j)
        {
            FactRec &r = recs[std::size_t((j + std::max(round, 0)) % nrec)];
            double w0 = now_ns(), c0 = thread_cpu_ns();
            cholmod_factor *L = cholmod_analyze(&r.A, &c);
            double w1 = now_ns(), c1 = thread_cpu_ns();
            if (L == nullptr)
            {
                std::fprintf(stderr, "analyze failed for %s\n", r.ix.id.c_str());
                continue;
            }
            const double fl = c.fl, lnz = c.lnz;
            double w2 = now_ns(), c2 = thread_cpu_ns();
            cholmod_factorize_p(&r.A, const_cast<double *>(beta), nullptr, 0, L, &c);
            double w3 = now_ns(), c3 = thread_cpu_ns();
            const bool ok = (L->minor == L->n);
            double ws = 0, cs = 0;
            if (ok)
            {
                cholmod_dense *b = cholmod_allocate_dense(L->n, 1, L->n, CHOLMOD_REAL, &c);
                double *bx = static_cast<double *>(b->x);
                for (std::size_t i = 0; i < L->n; ++i)
                    bx[i] = rng.normal();
                const double w4 = now_ns(), c4 = thread_cpu_ns();
                cholmod_dense *x = cholmod_solve(CHOLMOD_A, L, b, &c);
                ws = now_ns() - w4;
                cs = thread_cpu_ns() - c4;
                cholmod_free_dense(&x, &c);
                // the same solve with cold caches (a 96 MB sweep first, outside the timed region): in the solver the factor is not in the cache when the next solve comes
                {
                    static std::vector<char> sweep(96u << 20, 1);
                    char acc = 0;
                    for (std::size_t i = 0; i < sweep.size(); i += 64)
                        acc = char(acc + sweep[i]++);
                    g_sink = acc;
                }
                const double w5 = now_ns(), c5 = thread_cpu_ns();
                cholmod_dense *xc = cholmod_solve(CHOLMOD_A, L, b, &c);
                const double cold_cpu = thread_cpu_ns() - c5;
                (void)w5;
                cholmod_free_dense(&xc, &c);
                cholmod_free_dense(&b, &c);
                if (round >= 0)
                    r.s_cold_cpu.push_back(cold_cpu);
            }
            if (round >= 0)
            {
                r.an_wall.push_back(w1 - w0);
                r.an_cpu.push_back(c1 - c0);
                r.f_wall.push_back(w3 - w2);
                r.f_cpu.push_back(c3 - c2);
                if (ok)
                {
                    r.s_wall.push_back(ws);
                    r.s_cpu.push_back(cs);
                }
            }
            if (round == 0)
            {
                r.fl = fl;
                r.lnz = lnz;
                r.minor = long(L->minor);
                r.success = ok ? 1 : 0;
                if (ok)
                    r.fl_done = fl;
                else
                {
                    const int *cc = static_cast<const int *>(L->ColCount);
                    if (cc != nullptr)
                    {
                        double s = 0;
                        for (std::size_t jj = 0; jj < L->minor; ++jj)
                            s += double(cc[jj]) * double(cc[jj]);
                        r.fl_done = s;
                    }
                    const int *perm = static_cast<const int *>(L->Perm);
                    if (perm != nullptr && r.f.perm_minor >= 0 && L->minor < L->n)
                        r.perm_ok = (perm[L->minor] == r.f.perm_minor) ? 1 : 0;
                }
            }
            cholmod_free_factor(&L, &c);
        }
    cholmod_finish(&c);
    std::ofstream o(out);
    o << "id\tfile\tkind\tstep\touter\tpoint\tn\tnnz\texport_success\texport_minor\treplay_success\treplay_minor\tperm_ok\tfl\tlnz\tfl_done\t"
         "an_cpu_med\tan_cpu_q1\tan_cpu_q3\tf_cpu_med\tf_cpu_q1\tf_cpu_q3\ts_cpu_med\ts_cpu_q1\ts_cpu_q3\tf_wall_med\tsolver_cpu_ns\treps\tloadavg_end\ts_cold_cpu_med\n";
    const std::string la = loadavg_line();
    for (auto &r : recs)
    {
        const Quart a = quart_of(r.an_cpu), f = quart_of(r.f_cpu), s = quart_of(r.s_cpu), fw = quart_of(r.f_wall);
        o << r.ix.id << '\t' << r.ix.file << '\t' << r.ix.kind << '\t' << r.ix.step << '\t' << r.ix.outer << '\t' << r.ix.point << '\t' << r.f.n << '\t' << r.f.nnz << '\t' << r.f.success << '\t' << r.f.minor << '\t' << r.success << '\t'
          << r.minor << '\t' << r.perm_ok << '\t' << r.fl << '\t' << r.lnz << '\t' << r.fl_done << '\t' << a.med << '\t' << a.q1 << '\t' << a.q3 << '\t' << f.med << '\t' << f.q1 << '\t' << f.q3 << '\t' << s.med << '\t' << s.q1 << '\t' << s.q3
          << '\t' << fw.med << '\t' << r.ix.solver_cpu_ns << '\t' << f.n << '\t' << la << '\t' << quart_of(r.s_cold_cpu).med << '\n';
    }
    return 0;
}

// ================================================================================================ L4c
struct ModeOut
{
    std::ofstream o;
    explicit ModeOut(const std::string &p) : o(p)
    {
        o << "id\tmode\tarm\tmed_ns\tq1_ns\tq3_ns\treps\tn\tk\tnb\tnb_old\tn_new\tn6\tn9\tn12\tnR\tnnzK\twritten_old\tsolver_cpu_ns\n";
    }
};

static int l4c(const std::vector<std::string> &dirs, const std::string &out, const int reps)
{
    ModeOut mo(out);
    for (const auto &d : dirs)
        for (const auto &ix : read_index(d))
        {
            if (ix.type != "blocks")
                continue;
            const BlocksFile bf = read_blocks(d + "/" + ix.file);
            const int n = int(bf.n), k = int(bf.k), cap = int(bf.cap);
            std::printf("l4c %s (%s): n %d k %d cap %d blocks %ld (new %ld)\n", ix.id.c_str(), ix.file.c_str(), n, k, cap, bf.nb, bf.n_new);
            std::fflush(stdout);
            RowMat Qr(n, cap);
            std::memcpy(Qr.data(), bf.Qr.data(), sizeof(double) * bf.Qr.size());
            const Mat Qdummy(0, 0);
            std::vector<HessianBlock> all, olds, news;
            long n6 = 0, n9 = 0, n12 = 0;
            for (const auto &b : bf.blk)
            {
                HessianBlock h;
                h.size = b.size;
                for (int i = 0; i < 12; ++i)
                    h.indices[i] = b.indices[i];
                for (int j = 0; j < 12; ++j)
                    for (int i = 0; i < 12; ++i)
                        h.H(i, j) = b.H[j * 12 + i];
                all.push_back(h);
                (b.is_new ? news : olds).push_back(h);
                (b.size == 6 ? n6 : b.size == 9 ? n9 : n12) += 1;
            }
            Rng rng(11);
            const Mat G = random_sym(k, rng);
            Mat A(k, k);
            // fixed structure from the old blocks
            AsmWork w;
            CSR K;
            std::vector<int> posmap;
            assemble_rowacc(olds, n, w, K, &posmap);
            RowMat QR, W;
            gather_QR(K, Qr, k, QR);
            W.resize(K.nR, k);
            std::vector<double> val(K.col.size());
            spmm(K.ptr.data(), K.col.data(), K.val.data(), K.nR, QR.data(), W.data(), k);
            long written = 0;
            for (const auto &b : olds)
            {
                int nv = 0;
                for (int i = 0; i < b.size; ++i)
                    nv += b.indices[i] >= 0;
                written += long(nv) * nv;
            }
            for (int cold = 0; cold <= 1; ++cold)
            {
                std::map<std::string, std::vector<double>> T;
                std::vector<std::function<void(int)>> arms;
                const auto timed = [&](const char *name, const std::function<void()> &body) {
                    arms.push_back([&, name, body](int r) {
                        if (cold)
                            flush_cache();
                        else
                            body(); // hot: the same call just before, untimed
                        const double t0 = now_ns();
                        body();
                        const double t = now_ns() - t0;
                        if (r >= 0)
                            T[name].push_back(t);
                    });
                };
                timed("cur", [&] {
                    A = G;
                    v0_add_blocks<true>(A, all, Qr, Qdummy, k);
                    g_sink = A(0, 0);
                });
                timed("struct", [&] {
                    AsmWork w2;
                    CSR K2;
                    assemble_rowacc(olds, n, w2, K2);
                    g_sink = double(K2.col.size());
                });
                timed("write", [&] {
                    v1_val(olds, posmap, val);
                    g_sink = val.empty() ? 0.0 : val[0];
                });
                timed("gather", [&] {
                    gather_QR(K, Qr, k, QR);
                    g_sink = QR.size() ? QR(0, 0) : 0.0;
                });
                timed("spmm", [&] {
                    spmm(K.ptr.data(), K.col.data(), K.val.data(), K.nR, QR.data(), W.data(), k);
                    g_sink = W.size() ? W(0, 0) : 0.0;
                });
                timed("gemm", [&] {
                    A = G;
                    A.noalias() += QR.transpose() * W;
                    g_sink = A(0, 0);
                });
                timed("new", [&] {
                    A = G;
                    v0_add_blocks<true>(A, news, Qr, Qdummy, k);
                    g_sink = A(0, 0);
                });
                interleave(arms, reps);
                for (auto &kv : T)
                {
                    const Quart q = quart_of(kv.second);
                    mo.o << ix.id << '\t' << (cold ? "cold" : "hot") << '\t' << kv.first << '\t' << q.med << '\t' << q.q1 << '\t' << q.q3 << '\t' << q.n << '\t' << n << '\t' << k << '\t' << all.size() << '\t' << olds.size() << '\t' << news.size() << '\t' << n6
                         << '\t' << n9 << '\t' << n12 << '\t' << K.nR << '\t' << K.col.size() << '\t' << written << '\t' << ix.solver_cpu_ns << '\n';
                }
                mo.o.flush();
            }
        }
    return 0;
}

// ================================================================================================ L4e
struct Item
{
    std::string name;
    std::string param;
    double units;              ///< operations per call (ns per unit = time / units)
    std::function<double()> fn; ///< returns the elapsed ns of one call
};

static int l4e(const std::string &out, const int reps)
{
    Rng rng(5);
    std::vector<Item> items;
    // --- 3 x 3 and 12 x 12 symmetric eigen decompositions (N1, P1 fallback, R1)
    {
        const int N3 = 20000;
        auto m3 = std::make_shared<std::vector<Eigen::Matrix3d>>(N3);
        for (auto &m : *m3)
        {
            Eigen::Matrix3d r;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    r(i, j) = rng.normal();
            m = 0.5 * (r + r.transpose());
        }
        items.push_back({"eig3", "", double(N3), [m3, N3] {
                             const double t0 = now_ns();
                             double s = 0;
                             for (int i = 0; i < N3; ++i)
                             {
                                 Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es((*m3)[std::size_t(i)], Eigen::EigenvaluesOnly);
                                 s += es.eigenvalues()[0];
                             }
                             g_sink = s;
                             return now_ns() - t0;
                         }});
        const int N12 = 3000;
        auto m12 = std::make_shared<std::vector<Eigen::Matrix<double, 12, 12>>>(N12);
        for (auto &m : *m12)
        {
            Eigen::Matrix<double, 12, 12> r;
            for (int i = 0; i < 12; ++i)
                for (int j = 0; j < 12; ++j)
                    r(i, j) = rng.normal();
            m = 0.5 * (r + r.transpose());
        }
        items.push_back({"eig12_vectors", "", double(N12), [m12, N12] {
                             const double t0 = now_ns();
                             double s = 0;
                             for (int i = 0; i < N12; ++i)
                             {
                                 Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> es((*m12)[std::size_t(i)]);
                                 s += es.eigenvalues()[0] + es.eigenvectors()(0, 0);
                             }
                             g_sink = s;
                             return now_ns() - t0;
                         }});
        items.push_back({"eig12_values", "", double(N12), [m12, N12] {
                             const double t0 = now_ns();
                             double s = 0;
                             for (int i = 0; i < N12; ++i)
                             {
                                 Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> es((*m12)[std::size_t(i)], Eigen::EigenvaluesOnly);
                                 s += es.eigenvalues()[0];
                             }
                             g_sink = s;
                             return now_ns() - t0;
                         }});
    }
    // --- sorting (S1): ns per m log2 m
    for (int m : {100, 1000, 10000, 100000})
    {
        auto src = std::make_shared<std::vector<double>>(std::size_t(m));
        for (auto &x : *src)
            x = rng.normal();
        const int copies = std::max(1, 200000 / m);
        items.push_back({"sort", "m=" + std::to_string(m), double(copies) * m * std::log2(double(m)), [src, copies, m] {
                             std::vector<double> v(src->size());
                             double tt = 0;
                             for (int c = 0; c < copies; ++c)
                             {
                                 v = *src;
                                 const double t0 = now_ns();
                                 std::sort(v.begin(), v.end());
                                 tt += now_ns() - t0;
                                 g_sink = v[std::size_t(m) / 2];
                             }
                             return tt;
                         }});
    }
    // --- spmv with a FEM-like matrix (3 x 3 blocks, about 14 neighbour nodes), A2: add a column (nnz per column), per nnz
    for (int nn : {3200, 18400})
    {
        auto K = std::make_shared<CSR>();
        const int n = 3 * nn;
        K->nR = n;
        K->ptr.assign(std::size_t(n) + 1, 0);
        std::mt19937 g(3);
        std::vector<std::vector<int>> nbr;
        nbr.resize(std::size_t(nn));
        for (int a = 0; a < nn; ++a)
            for (int d = -7; d <= 7; ++d)
            {
                const int b = a + d + int(g() % 5) - 2; // near the diagonal, a little scattered like a mesh numbering
                if (b >= 0 && b < nn)
                    nbr[std::size_t(a)].push_back(b);
            }
        for (int a = 0; a < nn; ++a)
        {
            auto &v = nbr[std::size_t(a)];
            v.push_back(a);
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            for (int c = 0; c < 3; ++c)
            {
                for (int b : v)
                    for (int c2 = 0; c2 < 3; ++c2)
                    {
                        K->col.push_back(3 * b + c2);
                        K->val.push_back(1.0 / (1.0 + c + c2 + b % 7));
                    }
                K->ptr[std::size_t(3 * a + c) + 1] = int(K->col.size());
            }
        }
        auto x = std::make_shared<Vec>(Vec::Random(n));
        auto y = std::make_shared<Vec>(Vec::Zero(n));
        const double nnz = double(K->col.size());
        const int calls = std::max(1, int(2e7 / nnz));
        items.push_back({"spmv", "n=" + std::to_string(n), nnz * calls, [K, x, y, calls] {
                             const double t0 = now_ns();
                             for (int c = 0; c < calls; ++c)
                                 spmv(K->ptr.data(), K->col.data(), K->val.data(), K->nR, x->data(), y->data());
                             g_sink = (*y)[0];
                             return now_ns() - t0;
                         }});
        // dot / axpy over n
        items.push_back({"dot", "n=" + std::to_string(n), double(n) * calls, [x, y, calls] {
                             const double t0 = now_ns();
                             double s = 0;
                             for (int c = 0; c < calls; ++c)
                                 s += x->dot(*y) + c;
                             g_sink = s;
                             return now_ns() - t0;
                         }});
        items.push_back({"axpy", "n=" + std::to_string(n), double(n) * calls, [x, y, calls] {
                             const double t0 = now_ns();
                             for (int c = 0; c < calls; ++c)
                                 y->noalias() += (1e-9 * (c + 1)) * (*x);
                             g_sink = (*y)[1];
                             return now_ns() - t0;
                         }});
        // Q^T v and Q c (orthogonalization): row-major n x k
        for (int k : {5, 16, 40})
        {
            auto Q = std::make_shared<RowMat>(RowMat::Random(n, k));
            auto c = std::make_shared<Vec>(Vec::Zero(k));
            const int reps2 = std::max(1, int(2e7 / (double(n) * k)));
            items.push_back({"gemvT", "n=" + std::to_string(n) + ",k=" + std::to_string(k), double(n) * k * reps2, [Q, c, x, reps2] {
                                 const double t0 = now_ns();
                                 for (int r = 0; r < reps2; ++r)
                                     c->noalias() = Q->transpose() * (*x);
                                 g_sink = (*c)[0];
                                 return now_ns() - t0;
                             }});
            items.push_back({"gemv", "n=" + std::to_string(n) + ",k=" + std::to_string(k), double(n) * k * reps2, [Q, c, y, reps2] {
                                 const double t0 = now_ns();
                                 for (int r = 0; r < reps2; ++r)
                                     y->noalias() -= (*Q) * (*c);
                                 g_sink = (*y)[0];
                                 return now_ns() - t0;
                             }});
            // M1 / G2: Q^T w with w nonzero on r random rows (row gather of the row-major Q)
            const int rr = std::max(10, n / 20);
            auto rows = std::make_shared<std::vector<int>>(std::size_t(rr));
            for (auto &q : *rows)
                q = rng.below(n);
            const int reps3 = std::max(1, int(2e7 / (double(rr) * k)));
            items.push_back({"rowgather", "n=" + std::to_string(n) + ",k=" + std::to_string(k) + ",rows=" + std::to_string(rr), double(rr) * k * reps3, [Q, c, x, rows, k, reps3] {
                                 const double t0 = now_ns();
                                 for (int r = 0; r < reps3; ++r)
                                 {
                                     c->setZero();
                                     for (const int q : *rows)
                                     {
                                         const double w = (*x)[q];
                                         const double *qr = &(*Q)(q, 0);
                                         for (int j = 0; j < k; ++j)
                                             (*c)[j] += qr[j] * w;
                                     }
                                 }
                                 g_sink = (*c)[0];
                                 return now_ns() - t0;
                             }});
        }
    }
    // --- dense k x k Cholesky solve (the LM step of the subspace model)
    for (int k : {5, 10, 16, 20, 30, 40})
    {
        auto S = std::make_shared<Mat>(random_sym(k, rng));
        *S = (*S) * S->transpose() + Mat::Identity(k, k);
        auto b = std::make_shared<Vec>(Vec::Random(k));
        const int calls = std::max(1, int(2e6 / (double(k) * k * k + 1)));
        items.push_back({"dense_llt_solve", "k=" + std::to_string(k), double(calls), [S, b, calls] {
                             const double t0 = now_ns();
                             double s = 0;
                             for (int c = 0; c < calls; ++c)
                             {
                                 Eigen::LLT<Mat> llt(*S);
                                 s += llt.solve(*b)[0];
                             }
                             g_sink = s;
                             return now_ns() - t0;
                         }});
    }
    // --- the dense algebra of one inner step of the path and of the subspace model, as sequences of the same Eigen operations as PathRunner::run and SubspaceModel (ns per call; hot: repeated, cold: after a sweep of 96 MB)
    //     path_iter  8 products with the n x k basis and about 22 vector operations (counted in PathRunner::run: p_cur, pe, xe, the trial point, the two passes of the orthogonalization, the copies of the points,
    //                the model-residual vector, its norms)
    //     add_column the column copy, one spmv with the smooth Hessian, k + 1 dot products (SubspaceModel::add_column)
    //     eval_deriv H Q z, Q^T g (two products) and three vector operations (SubspaceModel::eval with the derivative)
    //     reread_set one spmv, Q^T r (one product) and five vector operations (the correction of the model gradient at a re-read point)
    for (int nn : {3200, 18400})
    {
        const int n = 3 * nn;
        auto K = std::make_shared<CSR>();
        K->nR = n;
        K->ptr.assign(std::size_t(n) + 1, 0);
        for (int r = 0; r < n; ++r)
        {
            for (int d = -21; d <= 21; ++d)
            {
                const int c = r + d * 3 / 2;
                if (c >= 0 && c < n)
                {
                    K->col.push_back(c);
                    K->val.push_back(1.0 / (1.0 + std::abs(d)));
                }
            }
            K->ptr[std::size_t(r) + 1] = int(K->col.size());
        }
        for (int k : {5, 16, 40})
        {
            const int cap = k + 2;
            auto Q = std::make_shared<Mat>(Mat::Random(n, cap));
            auto HQ = std::make_shared<Mat>(Mat::Random(n, cap));
            auto x = std::make_shared<Vec>(Vec::Random(n));
            auto g = std::make_shared<Vec>(Vec::Random(n));
            auto g0 = std::make_shared<Vec>(Vec::Random(n));
            auto Md = std::make_shared<Vec>(Vec::Constant(n, 1.0));
            auto zk = std::make_shared<Vec>(Vec::Random(k));
            auto dz = std::make_shared<Vec>(Vec::Random(k) * 1e-3);
            const std::string par = "n=" + std::to_string(n) + ",k=" + std::to_string(k);
            const auto sweep = [] {
                static std::vector<char> big(96u << 20, 1);
                char acc = 0;
                for (std::size_t i = 0; i < big.size(); i += 64)
                    acc = char(acc + big[i]++);
                g_sink = acc;
            };
            const auto path_iter = [Q, x, g, g0, Md, zk, dz, k, n]() {
                const auto Qk = Q->leftCols(k);
                Vec p_cur = Qk * (*zk);
                Vec pe = Qk * ((*zk) + (*dz));
                Vec xe = (*x) + Qk * ((*zk) + (*dz));
                Vec t1 = (*x) + p_cur, t2 = (*x) + pe;
                Vec y = (*x) + Qk * (*zk);
                Vec p_new = pe, c1 = p_new, c2 = p_cur;
                // the polyline bookkeeping of the path: copies of the point and the coefficients, the nonsmooth state's copy of y, further x + p sums of the guards
                Vec pts_copy = p_new, y_copy = y, s1 = (*x) + p_new, s2 = (*x) + p_cur, s3 = (*g) + (*x);
                Vec cg = (*g0), cy = xe;
                g_sink = pts_copy[0] + y_copy[0] + s1[0] + s2[0] + s3[0] + cg[0] + cy[0];
                Vec r = (*g) - (*g0);
                r -= 1e-9 * p_new;
                Vec u = r;
                double nu = std::sqrt((Md->array() * u.array() * u.array()).sum());
                Vec v = u;
                for (int pass = 0; pass < 2; ++pass)
                {
                    const Vec w = Qk.transpose() * (Md->asDiagonal() * v);
                    v -= Qk * w;
                }
                const double nv = std::sqrt((Md->array() * v.array() * v.array()).sum());
                Vec qn = v / (nv + 1e-300);
                g_sink = nu + nv + qn[0] + xe[0] + t1[0] + t2[0] + y[0] + c1[0] + c2[0];
            };
            const auto add_column = [Q, HQ, K, x, k, n]() {
                const int j = k;
                Q->col(j) = *x;
                spmv(K->ptr.data(), K->col.data(), K->val.data(), K->nR, x->data(), HQ->col(j).data());
                double s = 0;
                for (int i = 0; i <= j; ++i)
                    s += Q->col(i).dot(HQ->col(j));
                g_sink = s + x->dot(*x);
            };
            const auto eval_deriv = [Q, HQ, g, g0, zk, k, n]() {
                const Vec Hp = HQ->leftCols(k) * (*zk);
                const Vec full = (*g) + Hp + (*g0);
                const Vec eg = Q->leftCols(k).transpose() * full;
                g_sink = eg[0] + Hp[0];
            };
            const auto reread_set = [Q, K, x, g, g0, k, n]() {
                Vec hp(n);
                spmv(K->ptr.data(), K->col.data(), K->val.data(), K->nR, x->data(), hp.data());
                Vec mg = (*g0) + hp;
                mg += *g;
                const Vec rR = (*g) - mg;
                const Vec cq = Q->leftCols(k).transpose() * rR;
                g_sink = cq[0];
            };
            const std::vector<std::pair<std::string, std::function<void()>>> ops = {{"path_iter", path_iter}, {"add_column", add_column}, {"eval_deriv", eval_deriv}, {"reread_set", reread_set}};
            for (const auto &op : ops)
            {
                const auto f = op.second;
                const int reps_hot = std::max(1, int(2e7 / (double(n) * (k + 4))));
                items.push_back({op.first, par, double(reps_hot), [f, reps_hot] {
                                     f();
                                     const double t0 = now_ns();
                                     for (int r = 0; r < reps_hot; ++r)
                                         f();
                                     return now_ns() - t0;
                                 }});
                items.push_back({op.first + "_cold", par, 1.0, [f, sweep] {
                                     sweep();
                                     const double t0 = now_ns();
                                     f();
                                     return now_ns() - t0;
                                 }});
            }
        }
    }
    // --- run, arms (items) interleaved by rotation
    std::map<std::string, std::vector<double>> T;
    const int ni = int(items.size());
    std::printf("l4e: %d items\n", ni);
    for (int r = -2; r < reps; ++r)
        for (int a = 0; a < ni; ++a)
        {
            Item &it = items[std::size_t((a + std::max(r, 0)) % ni)];
            const double t = it.fn();
            if (r >= 0)
                T[it.name + "|" + it.param].push_back(t / it.units);
        }
    std::ofstream o(out);
    o << "item\tparam\tns_per_unit_med\tq1\tq3\treps\tloadavg_end\n";
    const std::string la = loadavg_line();
    for (auto &kv : T)
    {
        const Quart q = quart_of(kv.second);
        const auto bar = kv.first.find('|');
        o << kv.first.substr(0, bar) << '\t' << kv.first.substr(bar + 1) << '\t' << q.med << '\t' << q.q1 << '\t' << q.q3 << '\t' << q.n << '\t' << la << '\n';
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: cl_replay_solver (l4a|l4c) --export DIR [--export DIR ...] --out FILE [--reps N]\n       cl_replay_solver l4e --out FILE [--reps N]\n");
        return 2;
    }
    const std::string cmd = argv[1];
    std::vector<std::string> dirs;
    std::string out;
    int reps = cmd == "l4a" ? 7 : 21;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--export" && i + 1 < argc)
            dirs.push_back(argv[++i]);
        else if (a == "--out" && i + 1 < argc)
            out = argv[++i];
        else if (a == "--reps" && i + 1 < argc)
            reps = std::atoi(argv[++i]);
        else
        {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (out.empty())
    {
        std::fprintf(stderr, "--out FILE is required\n");
        return 2;
    }
    if (cmd == "l4a")
        return l4a(dirs, out, reps);
    if (cmd == "l4c")
        return l4c(dirs, out, reps);
    if (cmd == "l4e")
        return l4e(out, reps);
    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}
