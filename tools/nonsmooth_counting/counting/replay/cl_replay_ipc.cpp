// cl_replay_ipc: replay pricing programs that need ipc-toolkit and the polyfem compile flags (EIGEN_DONT_VECTORIZE, as the solver's contact code).
//   l4b  CCD and collision-set replay of the exported surface records: broad phase (Candidates::build), narrow phase (stock: Tight Inclusion on all cached candidates; certified: the certificate pass and
//        the exported Tight Inclusion queries), NormalCollisions::build (own broad phase, and from the cached candidates)
//   l4d  element kernel (fastp1: value, gradient, Hessian, projection, strain-space form) on the exported exact elements; contact kernel (barrier potential per collision type)
// Output: TSV files. Nothing runs on its own: joint/cl/B/run_B_replay.sh calls it after the exports exist.
#include "cl_io.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <ipc/broad_phase/create_broad_phase.hpp>
#include <cstdlib>
#include <ipc/candidates/candidates.hpp>
#include <ipc/candidates/edge_edge.hpp>
#include <ipc/candidates/edge_vertex.hpp>
#include <ipc/candidates/face_vertex.hpp>
#include <ipc/candidates/vertex_vertex.hpp>
#include <ipc/ccd/tight_inclusion_ccd.hpp>
#include <ipc/collision_mesh.hpp>
#include <ipc/collisions/normal/normal_collisions.hpp>
#include <ipc/ipc.hpp>
#include <ipc/potentials/barrier_potential.hpp>

#include <polyfem/solver/forms/FastP1.hpp> // from the include path of the export build (the export tree pf/ or the production tree)

#include <algorithm>
#include <chrono>
#include <functional>
#include <map>
#include <memory>

static double now_ns()
{
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::nano>(clk::now().time_since_epoch()).count();
}
static volatile double g_sink = 0;

static double quantile(std::vector<double> t, double q)
{
    std::sort(t.begin(), t.end());
    const double x = q * double(t.size() - 1);
    const std::size_t i = std::size_t(x);
    return i + 1 < t.size() ? t[i] + (x - double(i)) * (t[i + 1] - t[i]) : t[i];
}
struct Quart
{
    double med = 0, q1 = 0, q3 = 0;
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
    return q;
}
static std::string loadavg_line()
{
    std::ifstream f("/proc/loadavg");
    std::string l;
    std::getline(f, l);
    return l;
}
/// arms in rounds, the start of the rotation moves by one each round; two warm-up rounds
static void interleave(std::vector<std::function<void(int)>> &arms, const int reps)
{
    const int na = int(arms.size());
    for (int r = -2; r < reps; ++r)
        for (int a = 0; a < na; ++a)
            arms[std::size_t((a + (r < 0 ? 0 : r)) % na)](r);
}

static Eigen::MatrixXd to_matrix(const std::vector<double> &v, const long nv)
{
    Eigen::MatrixXd M(nv, 3);
    for (long i = 0; i < nv; ++i)
        for (int c = 0; c < 3; ++c)
            M(i, c) = v[std::size_t(3 * i + c)];
    return M;
}

// ================================================================================================ L4b
// The certificate of the solver's narrow phase (BilateralSegmentCertifier::max_step_size of the PolyFEM code), copied: the translation-removed displacement bound and the AABB gap
// of a stencil, and the rule that decides which candidates are handed to Tight Inclusion.
static void cert_bounds(const ipc::VectorMax12d &xa, const ipc::VectorMax12d &xb, int na, double &utr, double &gap)
{
    const int nv = int(xa.size() / 3);
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (int i = 0; i < nv; ++i)
        mean += xb.segment<3>(3 * i) - xa.segment<3>(3 * i);
    mean /= nv;
    double ua = 0, ub = 0;
    for (int i = 0; i < nv; ++i)
    {
        const double e = (xb.segment<3>(3 * i) - xa.segment<3>(3 * i) - mean).norm();
        (i < na ? ua : ub) = std::max(i < na ? ua : ub, e);
    }
    utr = ua + ub;
    Eigen::Vector3d amin = Eigen::Vector3d::Constant(INFINITY), amax = -amin, bmin = amin, bmax = -amin;
    for (int i = 0; i < nv; ++i)
    {
        const Eigen::Vector3d p = xa.segment<3>(3 * i);
        if (i < na)
        {
            amin = amin.cwiseMin(p);
            amax = amax.cwiseMax(p);
        }
        else
        {
            bmin = bmin.cwiseMin(p);
            bmax = bmax.cwiseMax(p);
        }
    }
    double g2 = 0;
    for (int k = 0; k < 3; ++k)
    {
        const double g = std::max({bmin[k] - amax[k], amin[k] - bmax[k], 0.0});
        g2 += g * g;
    }
    gap = std::sqrt(g2);
}

/// the certificate pass of the solver over the candidates C: flags the candidates left for Tight Inclusion and drops the provably far ones (dedupe); returns the number flagged
static long cert_pass(ipc::Candidates &C, const Eigen::MatrixXd &V0, const Eigen::MatrixXd &V1, const ipc::CollisionMesh &mesh, const double dhat, std::vector<char> &to_ti)
{
    constexpr double cert_c = 0.8;
    const auto &E = mesh.edges();
    const auto &Fc = mesh.faces();
    const double margin = 1e-3 * dhat, far_thr = 1.001 * dhat;
    const std::size_t begin_ee = C.vv_candidates.size() + C.ev_candidates.size(), begin_fv = begin_ee + C.ee_candidates.size(), end_fv = begin_fv + C.fv_candidates.size();
    const std::size_t nC = C.size();
    to_ti.assign(nC, 0);
    std::vector<char> keep(nC, 1);
    for (std::size_t i = 0; i < nC; ++i)
    {
        if (i >= begin_ee && i < end_fv)
        {
            const ipc::CollisionStencil &s = C[i];
            const ipc::VectorMax12d xa = s.dof(V0, E, Fc), xb = s.dof(V1, E, Fc);
            const int na = i >= begin_fv ? 1 : 2;
            double utr, gap;
            cert_bounds(xa, xb, na, utr, gap);
            if (gap >= (utr + margin) / cert_c)
            {
                if (gap - utr >= far_thr)
                    keep[i] = 0;
                continue;
            }
            const double Da = std::sqrt(s.compute_distance(xa));
            if (Da - utr >= far_thr)
                keep[i] = 0;
            if (utr <= cert_c * Da - margin)
                continue;
        }
        to_ti[i] = 1;
    }
    long n_ti = 0;
    for (std::size_t i = 0; i < nC; ++i)
        n_ti += to_ti[i];
    const auto filt = [&](auto &vec, std::size_t off) {
        std::size_t w = 0;
        for (std::size_t i = 0; i < vec.size(); ++i)
            if (keep[off + i])
                vec[w++] = vec[i];
        vec.erase(vec.begin() + w, vec.end());
    };
    filt(C.ee_candidates, begin_ee);
    filt(C.fv_candidates, begin_fv);
    return n_ti;
}

static std::unique_ptr<ipc::CollisionMesh> load_mesh(const std::string &d, const std::vector<IndexRow> &idx)
{
    for (const auto &ix : idx)
        if (ix.type == "ccdmesh")
        {
            const CMeshFile cm = read_cmesh(d + "/" + ix.file);
            Eigen::MatrixXi E(cm.ne, 2), F(cm.nf, 3);
            for (long i = 0; i < cm.ne; ++i)
                for (int c = 0; c < 2; ++c)
                    E(i, c) = cm.edges[std::size_t(2 * i + c)];
            for (long i = 0; i < cm.nf; ++i)
                for (int c = 0; c < 3; ++c)
                    F(i, c) = cm.faces[std::size_t(3 * i + c)];
            std::printf("collision mesh %ld vertices, %ld edges, %ld faces\n", cm.nv, cm.ne, cm.nf);
            return std::make_unique<ipc::CollisionMesh>(to_matrix(cm.rest, cm.nv), E, F);
        }
    return nullptr;
}

// Arms (kind 0, the CCD call of a segment; the timed region of every arm starts after its inputs are made):
//   broad_build        Candidates::build with a fresh broad-phase object (made outside the timed region)
//   narrow_cached      the stock narrow phase: Tight Inclusion on all cached candidates (surf records with narrow=stock or none)
//   narrow_cert_pass   the certificate pass of the solver's narrow phase over all candidates (no Tight Inclusion; surf records with narrow=cert)
//   narrow_ti          Tight Inclusion on the queries that the solver really handed to it, with the tmax of each, as exported (tiq record of a narrow=cert record)
//   collisions_cached  NormalCollisions::build from the cached candidates at the end of the segment (what the solver does at the trial points of a segment)
// kind 1, a collision-set build with its own broad phase: collisions_build.
// 11-23: the earlier version of this function kept V1 and the Tight Inclusion object inside the `if (kind == 0)` block, and the arms (lambdas capturing them by reference) ran after the block
// had ended: a use after scope. The broad phase and the narrow phase read freed memory (garbage coordinates): the candidate count of the second round was not that of the first, the broad phase
// ran out of memory, and Tight Inclusion reported overflows on the garbage. Both are made before the arms now and live as long as they do.
static int l4b(const std::vector<std::string> &dirs, const std::string &out, const int reps)
{
    std::ofstream o(out);
    o << "id\tkind\tnv\tbroad_phase\tuse_cached\tn_cand_logged\tvv\tev\tee\tfv\tpv\tarm\tmed_ns\tq1_ns\tq3_ns\treps\tsolver_cpu_ns\tloadavg_end\tnarrow\tq_vv\tq_ev\tq_ee\tq_fv\tflagged\n";
    for (const auto &d : dirs)
    {
        const auto idx = read_index(d);
        std::unique_ptr<ipc::CollisionMesh> mesh = load_mesh(d, idx);
        if (!mesh)
        {
            std::fprintf(stderr, "l4b: %s has no ccdmesh record, skipped\n", d.c_str());
            continue;
        }
        for (const auto &ix : idx)
        {
            if (ix.type != "surf")
                continue;
            const SurfFile sf = read_surf(d + "/" + ix.file);
            const std::string narrow = ix.extra.find("narrow=cert") != std::string::npos ? "cert" : ix.extra.find("narrow=stock") != std::string::npos ? "stock" : "none";
            const Eigen::MatrixXd V0 = to_matrix(sf.V0, sf.nv);
            const Eigen::MatrixXd V1 = sf.kind == 0 ? to_matrix(sf.V1, sf.nv) : Eigen::MatrixXd();
            const ipc::TightInclusionCCD tight(sf.ccd_tol, sf.ccd_max_iter);
            const auto bpm = static_cast<ipc::BroadPhaseMethod>(sf.broad_phase);
            std::shared_ptr<ipc::BroadPhase> bp;
            try
            {
                bp = ipc::create_broad_phase(bpm);
            }
            catch (const std::exception &e)
            {
                std::fprintf(stderr, "l4b: %s: broad phase %ld not available (%s), skipped\n", ix.id.c_str(), sf.broad_phase, e.what());
                continue;
            }
            std::map<std::string, std::vector<double>> T;
            std::vector<std::function<void(int)>> arms;
            ipc::Candidates cand;
            ipc::NormalCollisions coll;
            long cvv = 0, cev = 0, cee = 0, cfv = 0, cpv = 0;
            long qc[4] = {0, 0, 0, 0}, flagged = -1;
            TiqFile tq;
            bool have_tiq = false;
            // the queries as stencils and positions: they live as long as the arms that read them (the arms run after the blocks below end)
            std::vector<std::unique_ptr<ipc::CollisionStencil>> st;
            std::vector<ipc::VectorMax12d> xa, xb;
            std::unique_ptr<ipc::TightInclusionCCD> tight_q;
            const auto put = [&](const char *name, int r, double t) {
                if (r >= 0)
                    T[name].push_back(t);
            };
            if (sf.kind == 0)
            {
                cand.build(*mesh, V0, V1, sf.inflation, bp.get());
                cvv = long(cand.vv_candidates.size());
                cev = long(cand.ev_candidates.size());
                cee = long(cand.ee_candidates.size());
                cfv = long(cand.fv_candidates.size());
                cpv = long(cand.pv_candidates.size());
                if (std::getenv("CL_L4B_SKIP_BROAD") == nullptr)
                    arms.push_back([&](int r) {
                        // a fresh broad-phase object per build, made outside the timed region
                        std::shared_ptr<ipc::BroadPhase> bpr = ipc::create_broad_phase(bpm);
                        ipc::Candidates c2;
                        const double t0 = now_ns();
                        c2.build(*mesh, V0, V1, sf.inflation, bpr.get());
                        const double t = now_ns() - t0;
                        g_sink = double(c2.size());
                        put("broad_build", r, t);
                    });
                if (narrow == "cert")
                {
                    std::vector<char> to_ti;
                    {
                        ipc::Candidates c3 = cand; // the pass drops candidates: on a copy
                        flagged = cert_pass(c3, V0, V1, *mesh, sf.dhat, to_ti);
                    }
                    for (const auto &jx : idx)
                        if (jx.type == "tiq" && jx.id == "tiq_" + ix.id.substr(5)) // the two files of one export item share the number
                        {
                            tq = read_tiq(d + "/" + jx.file);
                            have_tiq = true;
                            break;
                        }
                    if (have_tiq)
                        for (const auto &q : tq.q)
                            ++qc[q.family];
                    arms.push_back([&](int r) {
                        ipc::Candidates c3 = cand; // outside the timed region
                        std::vector<char> flags;
                        const double t0 = now_ns();
                        const long nf = cert_pass(c3, V0, V1, *mesh, sf.dhat, flags);
                        const double t = now_ns() - t0;
                        g_sink = double(nf);
                        put("narrow_cert_pass", r, t);
                    });
                    if (have_tiq)
                    {
                        // the queries as stencils (the indices do not matter to ccd(): it reads the positions it is given)
                        for (const auto &q : tq.q)
                        {
                            if (q.family == 0)
                                st.push_back(std::make_unique<ipc::VertexVertexCandidate>(0, 1));
                            else if (q.family == 1)
                                st.push_back(std::make_unique<ipc::EdgeVertexCandidate>(0, 0));
                            else if (q.family == 2)
                                st.push_back(std::make_unique<ipc::EdgeEdgeCandidate>(0, 1));
                            else
                                st.push_back(std::make_unique<ipc::FaceVertexCandidate>(0, 0));
                            ipc::VectorMax12d a(3 * q.nv), b(3 * q.nv);
                            for (int i = 0; i < 3 * q.nv; ++i)
                            {
                                a[i] = q.xa[i];
                                b[i] = q.xb[i];
                            }
                            xa.push_back(a);
                            xb.push_back(b);
                        }
                        tight_q = std::make_unique<ipc::TightInclusionCCD>(tq.tol, tq.max_iter);
                        arms.push_back([&](int r) {
                            const double t0 = now_ns();
                            double acc = 0;
                            for (std::size_t i = 0; i < st.size(); ++i)
                            {
                                double toi = 1;
                                if (st[i]->ccd(xa[i], xb[i], toi, tq.min_distance, tq.q[i].tmax, *tight_q))
                                    acc += toi;
                            }
                            const double t = now_ns() - t0;
                            g_sink = acc;
                            put("narrow_ti", r, t);
                        });
                    }
                }
                else if (std::getenv("CL_L4B_SKIP_NARROW") == nullptr)
                    arms.push_back([&](int r) {
                        const double t0 = now_ns();
                        const double a = cand.compute_collision_free_stepsize(*mesh, V0, V1, sf.dmin, tight);
                        const double t = now_ns() - t0;
                        g_sink = a;
                        put("narrow_cached", r, t);
                    });
                // the collision set built from the cached candidates at the end of the segment (the trial points of a segment)
                arms.push_back([&](int r) {
                    ipc::NormalCollisions c2;
                    const double t0 = now_ns();
                    c2.build(cand, *mesh, V1, sf.dhat);
                    const double t = now_ns() - t0;
                    g_sink = double(c2.size());
                    put("collisions_cached", r, t);
                });
                if (std::getenv("CL_L4B_UNCACHED") != nullptr)
                    arms.push_back([&](int r) {
                        const double t0 = now_ns();
                        const double a = ipc::compute_collision_free_stepsize(*mesh, V0, V1, sf.dmin, bp.get(), tight);
                        const double t = now_ns() - t0;
                        g_sink = a;
                        put("full_uncached", r, t);
                    });
            }
            else
            {
                coll.build(*mesh, V0, sf.dhat, sf.dmin, bp.get());
                cvv = long(coll.vv_collisions.size());
                cev = long(coll.ev_collisions.size());
                cee = long(coll.ee_collisions.size());
                cfv = long(coll.fv_collisions.size());
                cpv = long(coll.pv_collisions.size());
                arms.push_back([&](int r) {
                    ipc::NormalCollisions c2;
                    const double t0 = now_ns();
                    c2.build(*mesh, V0, sf.dhat, sf.dmin, bp.get());
                    const double t = now_ns() - t0;
                    g_sink = double(c2.size());
                    put("collisions_build", r, t);
                });
            }
            std::printf("l4b %s kind %ld nv %ld narrow=%s cand/coll vv %ld ev %ld ee %ld fv %ld (ccd tol %g, max_iter %ld, inflation %g)", ix.id.c_str(), sf.kind, sf.nv, narrow.c_str(), cvv, cev, cee, cfv, sf.ccd_tol, sf.ccd_max_iter, sf.inflation);
            if (narrow == "cert")
                std::printf("; flagged for TI %ld, queries exported %zu%s", flagged, tq.q.size(), have_tiq ? (flagged == long(tq.q.size()) ? " (equal)" : " (DIFFERENT)") : " (no tiq record)");
            std::printf("\n");
            std::fflush(stdout);
            interleave(arms, reps);
            const std::string la = loadavg_line();
            for (auto &kv : T)
            {
                const Quart q = quart_of(kv.second);
                o << ix.id << '\t' << sf.kind << '\t' << sf.nv << '\t' << sf.broad_phase << '\t' << sf.use_cached << '\t' << sf.n_cand_logged << '\t' << cvv << '\t' << cev << '\t' << cee << '\t' << cfv << '\t' << cpv << '\t' << kv.first << '\t'
                  << q.med << '\t' << q.q1 << '\t' << q.q3 << '\t' << q.n << '\t' << ix.solver_cpu_ns << '\t' << la << '\t' << narrow << '\t' << qc[0] << '\t' << qc[1] << '\t' << qc[2] << '\t' << qc[3] << '\t' << flagged << '\n';
            }
            o.flush();
        }
    }
    return 0;
}

// ================================================================================================ L4t
// The Tight Inclusion queries of every call of the run (tiq_all.bin of the export directory, CLTIQA01: per recorded call seq, step, outer, number of queries, weight; then the queries) replayed
// call by call, in the order of the call with the tmax of each query: the cost of the narrow phase's Tight Inclusion part of the whole run (the weight of a call is the number of calls it stands for
// when the budget of the export made it store only every 2nd, 4th, ... call). Heavy-tailed: a few calls (sliding flat contact) cost orders of magnitude more than the rest.
static int l4t(const std::vector<std::string> &dirs, const std::string &out)
{
    std::ofstream o(out);
    o << "export\tseq\tstep\touter\tnq\tweight\treps\tmed_ns\tq_vv\tq_ev\tq_ee\tq_fv\n";
    for (const auto &d : dirs)
    {
        std::FILE *f = std::fopen((d + "/tiq_all.bin").c_str(), "rb");
        if (f == nullptr)
        {
            std::fprintf(stderr, "l4t: %s has no tiq_all.bin, skipped\n", d.c_str());
            continue;
        }
        char magic[9] = {0};
        if (std::fread(magic, 1, 8, f) != 8 || std::strcmp(magic, "CLTIQA01") != 0)
        {
            std::fprintf(stderr, "l4t: %s/tiq_all.bin: bad magic\n", d.c_str());
            std::fclose(f);
            continue;
        }
        // the tolerance and iteration limit of the Tight Inclusion object are those of the surf records of the same export (the same solver object)
        double tol = 1e-7;
        long max_iter = 1000000;
        for (const auto &ix : read_index(d))
            if (ix.type == "surf")
            {
                const SurfFile sf = read_surf(d + "/" + ix.file);
                tol = sf.ccd_tol;
                max_iter = sf.ccd_max_iter;
                break;
            }
        const ipc::TightInclusionCCD tight(tol, max_iter);
        long long hdr[5];
        long ncalls = 0;
        double tot = 0, wq = 0;
        while (std::fread(hdr, sizeof(long long), 5, f) == 5)
        {
            const long nq = long(hdr[3]);
            std::vector<std::unique_ptr<ipc::CollisionStencil>> st;
            std::vector<ipc::VectorMax12d> xa, xb;
            std::vector<double> tmax;
            long qc[4] = {0, 0, 0, 0};
            for (long i = 0; i < nq; ++i)
            {
                int fam, nv;
                double tm, a[12], b[12];
                if (std::fread(&fam, sizeof(int), 1, f) != 1 || std::fread(&nv, sizeof(int), 1, f) != 1 || std::fread(&tm, sizeof(double), 1, f) != 1 || std::fread(a, sizeof(double), 12, f) != 12
                    || std::fread(b, sizeof(double), 12, f) != 12)
                {
                    std::fprintf(stderr, "l4t: %s/tiq_all.bin: short read\n", d.c_str());
                    break;
                }
                if (fam == 0)
                    st.push_back(std::make_unique<ipc::VertexVertexCandidate>(0, 1));
                else if (fam == 1)
                    st.push_back(std::make_unique<ipc::EdgeVertexCandidate>(0, 0));
                else if (fam == 2)
                    st.push_back(std::make_unique<ipc::EdgeEdgeCandidate>(0, 1));
                else
                    st.push_back(std::make_unique<ipc::FaceVertexCandidate>(0, 0));
                ++qc[fam];
                ipc::VectorMax12d va(3 * nv), vb(3 * nv);
                for (int j = 0; j < 3 * nv; ++j)
                {
                    va[j] = a[j];
                    vb[j] = b[j];
                }
                xa.push_back(va);
                xb.push_back(vb);
                tmax.push_back(tm);
            }
            const auto once = [&]() {
                const double t0 = now_ns();
                double acc = 0;
                for (std::size_t i = 0; i < st.size(); ++i)
                {
                    double toi = 1;
                    if (st[i]->ccd(xa[i], xb[i], toi, 0.0, tmax[i], tight))
                        acc += toi;
                }
                g_sink = acc;
                return now_ns() - t0;
            };
            std::vector<double> t;
            t.push_back(once());
            const int reps = t[0] < 2e6 ? 7 : t[0] < 5e7 ? 3 : 1;
            for (int r = 1; r < reps; ++r)
                t.push_back(once());
            const double med = quantile(t, 0.5);
            o << d << '\t' << hdr[0] << '\t' << hdr[1] << '\t' << hdr[2] << '\t' << nq << '\t' << hdr[4] << '\t' << reps << '\t' << med << '\t' << qc[0] << '\t' << qc[1] << '\t' << qc[2] << '\t' << qc[3] << '\n';
            tot += med * double(hdr[4]);
            wq += double(nq) * double(hdr[4]);
            ++ncalls;
        }
        std::fclose(f);
        std::printf("l4t %s: %ld calls stored, %.0f queries (weighted), %.3f s\n", d.c_str(), ncalls, wq, tot * 1e-9);
        std::fflush(stdout);
        o.flush();
    }
    return 0;
}

// ================================================================================================ L4d
/// numerical PSD projection of a symmetric 12 x 12 matrix (the fallback of fastp1::project_f)
static void project_numeric(const Eigen::Matrix<double, 12, 12> &H, Eigen::Matrix<double, 12, 12> &out)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> es(H);
    Eigen::Matrix<double, 12, 1> ev = es.eigenvalues().cwiseMax(0.0);
    out = es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
}

static int l4d(const std::vector<std::string> &dirs, const std::string &out, const int reps)
{
    std::ofstream o(out);
    o << "id\ttype\tn_units\tarm\tmed_ns_per_unit\tq1\tq3\treps\tvv\tev\tee\tfv\tpv\tloadavg_end\n";
    for (const auto &d : dirs)
    {
        const auto idx = read_index(d);
        std::unique_ptr<ipc::CollisionMesh> mesh = load_mesh(d, idx);
        for (const auto &ix : idx)
        {
            // ---------------------------------------------------------------- exact elements
            if (ix.type == "elem")
            {
                const ElemFile ef = read_elem(d + "/" + ix.file);
                const int ne = int(ef.ne);
                std::vector<polyfem::solver::fastp1::Elem> el;
                el.resize(std::size_t(ne));
                for (int e = 0; e < ne; ++e)
                {
                    auto &x = el[std::size_t(e)];
                    for (int i = 0; i < 12; ++i)
                    {
                        x.B[i] = ef.B[std::size_t(12 * e + i)];
                        x.dofs[i] = i;
                    }
                    x.da = ef.da[std::size_t(e)];
                    x.lambda = ef.lambda[std::size_t(e)];
                    x.mu = ef.mu[std::size_t(e)];
                    polyfem::solver::fastp1::fill_derived(x);
                }
                std::printf("l4d %s: %d elements\n", ix.id.c_str(), ne);
                std::fflush(stdout);
                std::map<std::string, std::vector<double>> T;
                std::vector<std::function<void(int)>> arms;
                const auto add = [&](const char *name, const std::function<void(int)> &body) {
                    arms.push_back([&, name, body](int r) {
                        const double t0 = now_ns();
                        body(r);
                        const double t = now_ns() - t0;
                        if (r >= 0)
                            T[name].push_back(t / ne);
                    });
                };
                add("eval_value", [&](int) {
                    double s = 0;
                    for (int e = 0; e < ne; ++e)
                        s += polyfem::solver::fastp1::eval<false, false>(el[std::size_t(e)], &ef.u[std::size_t(12 * e)], nullptr, nullptr);
                    g_sink = s;
                });
                add("eval_grad", [&](int) {
                    double s = 0, g[12];
                    for (int e = 0; e < ne; ++e)
                        s += polyfem::solver::fastp1::eval<true, false>(el[std::size_t(e)], &ef.u[std::size_t(12 * e)], g, nullptr);
                    g_sink = s + g[0];
                });
                add("eval_grad_hess", [&](int) {
                    double s = 0, g[12], H[144];
                    for (int e = 0; e < ne; ++e)
                        s += polyfem::solver::fastp1::eval<true, true>(el[std::size_t(e)], &ef.u[std::size_t(12 * e)], g, H);
                    g_sink = s + g[0] + H[0];
                });
#ifdef CL_FASTP1_FULL // the experiment kernel only (project_f, eval_f); the production FastP1.hpp has eval<G, H> only
                add("project_f_analytic", [&](int) {
                    double H[144];
                    int fails = 0;
                    for (int e = 0; e < ne; ++e)
                        fails += polyfem::solver::fastp1::project_f(el[std::size_t(e)], &ef.u[std::size_t(12 * e)], H, true) ? 0 : 1;
                    g_sink = H[0] + fails;
                });
#endif
                add("project_numeric", [&](int) {
                    double g[12], H[144];
                    Eigen::Matrix<double, 12, 12> P;
                    double s = 0;
                    for (int e = 0; e < ne; ++e)
                    {
                        polyfem::solver::fastp1::eval<true, true>(el[std::size_t(e)], &ef.u[std::size_t(12 * e)], g, H);
                        project_numeric(Eigen::Map<const Eigen::Matrix<double, 12, 12>>(H), P);
                        s += P(0, 0);
                    }
                    g_sink = s;
                });
#ifdef CL_FASTP1_FULL
                add("eval_f_strain_space", [&](int) {
                    double C[81], P[9], s = 0;
                    for (int e = 0; e < ne; ++e)
                    {
                        polyfem::solver::fastp1::eval_f(el[std::size_t(e)], &ef.u[std::size_t(12 * e)], el[std::size_t(e)].da, C, P);
                        s += C[0] + P[0];
                    }
                    g_sink = s;
                });
#endif
                interleave(arms, reps);
                const std::string la = loadavg_line();
                for (auto &kv : T)
                {
                    const Quart q = quart_of(kv.second);
                    o << ix.id << "\telem\t" << ne << '\t' << kv.first << '\t' << q.med << '\t' << q.q1 << '\t' << q.q3 << '\t' << q.n << "\t0\t0\t0\t0\t0\t" << la << '\n';
                }
                o.flush();
            }
            // ---------------------------------------------------------------- contact pairs by type
            if (ix.type == "surf" && mesh)
            {
                const SurfFile sf = read_surf(d + "/" + ix.file);
                if (sf.kind != 1)
                    continue;
                const Eigen::MatrixXd V = to_matrix(sf.V0, sf.nv);
                std::shared_ptr<ipc::BroadPhase> bp;
                try
                {
                    bp = ipc::create_broad_phase(static_cast<ipc::BroadPhaseMethod>(sf.broad_phase));
                }
                catch (const std::exception &)
                {
                    continue;
                }
                ipc::NormalCollisions all;
                all.build(*mesh, V, sf.dhat, sf.dmin, bp.get());
                const long cvv = long(all.vv_collisions.size()), cev = long(all.ev_collisions.size()), cee = long(all.ee_collisions.size()), cfv = long(all.fv_collisions.size()), cpv = long(all.pv_collisions.size());
                std::printf("l4d %s: collisions vv %ld ev %ld ee %ld fv %ld\n", ix.id.c_str(), cvv, cev, cee, cfv);
                std::fflush(stdout);
                const ipc::BarrierPotential pot(sf.dhat, 1.0); // stiffness 1 (the price does not depend on it)
                // one collision set per type (the others emptied)
                std::vector<std::pair<std::string, std::shared_ptr<ipc::NormalCollisions>>> sets;
                const auto only = [&](const char *name, const std::function<void(ipc::NormalCollisions &)> &keep) {
                    auto c = std::make_shared<ipc::NormalCollisions>(all);
                    c->vv_collisions.clear();
                    c->ev_collisions.clear();
                    c->ee_collisions.clear();
                    c->fv_collisions.clear();
                    c->pv_collisions.clear();
                    keep(*c);
                    sets.push_back({name, c});
                };
                if (cev)
                    only("ev", [&](ipc::NormalCollisions &c) { c.ev_collisions = all.ev_collisions; });
                if (cee)
                    only("ee", [&](ipc::NormalCollisions &c) { c.ee_collisions = all.ee_collisions; });
                if (cfv)
                    only("fv", [&](ipc::NormalCollisions &c) { c.fv_collisions = all.fv_collisions; });
                std::map<std::string, std::vector<double>> T;
                std::map<std::string, long> units;
                std::vector<std::function<void(int)>> arms;
                for (auto &s : sets)
                {
                    const std::string name = s.first;
                    const auto cs = s.second;
                    const long nu = long(cs->size());
                    units[name] = nu;
                    const auto add = [&](const std::string &arm, const std::function<void()> &body) {
                        arms.push_back([&T, name, arm, body, nu](int r) {
                            const double t0 = now_ns();
                            body();
                            const double t = now_ns() - t0;
                            if (r >= 0)
                                T[name + ":" + arm].push_back(t / double(nu));
                        });
                    };
                    add("value", [&, cs] { g_sink = pot(*cs, *mesh, V); });
                    add("gradient", [&, cs] {
                        const Eigen::VectorXd g = pot.gradient(*cs, *mesh, V);
                        g_sink = g.size() ? g[0] : 0.0;
                    });
                    add("hessian", [&, cs] {
                        const auto H = pot.hessian(*cs, *mesh, V, ipc::PSDProjectionMethod::NONE);
                        g_sink = double(H.nonZeros());
                    });
                    add("hessian_psd", [&, cs] {
                        const auto H = pot.hessian(*cs, *mesh, V, ipc::PSDProjectionMethod::CLAMP);
                        g_sink = double(H.nonZeros());
                    });
                }
                interleave(arms, reps);
                const std::string la = loadavg_line();
                for (auto &kv : T)
                {
                    const Quart q = quart_of(kv.second);
                    const std::string type = kv.first.substr(0, kv.first.find(':'));
                    o << ix.id << "\tcontact_" << type << '\t' << units[type] << '\t' << kv.first.substr(kv.first.find(':') + 1) << '\t' << q.med << '\t' << q.q1 << '\t' << q.q3 << '\t' << q.n << '\t' << cvv << '\t' << cev << '\t' << cee << '\t' << cfv << '\t' << cpv
                      << '\t' << la << '\n';
                }
                o.flush();
            }
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: cl_replay_ipc (l4b|l4d) --export DIR [--export DIR ...] --out FILE [--reps N]\n");
        return 2;
    }
    const std::string cmd = argv[1];
    std::vector<std::string> dirs;
    std::string out;
    int reps = 15;
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
    if (cmd == "l4b")
        return l4b(dirs, out, reps);
    if (cmd == "l4d")
        return l4d(dirs, out, reps);
    if (cmd == "l4t")
        return l4t(dirs, out);
    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}
