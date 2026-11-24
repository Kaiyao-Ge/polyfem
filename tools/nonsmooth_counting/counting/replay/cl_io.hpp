// cl_io.hpp: readers of the export files of joint/cl/interfaces.md (section 2) and the export index (plus tiq: CLTIQ001, the narrow-phase queries handed to Tight Inclusion).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct IndexRow
{
    std::string id, file, type, kind;
    long step = -1, outer = -1, point = -1;
    int success = -1;
    long minor = -1, n = -1, nnz = -1;
    std::string extra;
    double solver_cpu_ns = 0;
};

/// export/index.tsv (header row; columns id file type step outer point kind success minor n nnz extra solver_cpu_ns)
static std::vector<IndexRow> read_index(const std::string &dir)
{
    std::vector<IndexRow> rows;
    std::ifstream f(dir + "/index.tsv");
    if (!f)
    {
        std::fprintf(stderr, "cannot open %s/index.tsv\n", dir.c_str());
        std::exit(2);
    }
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line))
    {
        if (line.empty())
            continue;
        std::vector<std::string> c;
        std::stringstream ss(line);
        std::string x;
        while (std::getline(ss, x, '\t'))
            c.push_back(x);
        if (c.size() < 13)
        {
            std::fprintf(stderr, "index.tsv: short row: %s\n", line.c_str());
            std::exit(2);
        }
        IndexRow r;
        r.id = c[0];
        r.file = c[1];
        r.type = c[2];
        r.step = std::atol(c[3].c_str());
        r.outer = std::atol(c[4].c_str());
        r.point = std::atol(c[5].c_str());
        r.kind = c[6];
        r.success = std::atoi(c[7].c_str());
        r.minor = std::atol(c[8].c_str());
        r.n = std::atol(c[9].c_str());
        r.nnz = std::atol(c[10].c_str());
        r.extra = c[11];
        r.solver_cpu_ns = std::atof(c[12].c_str());
        rows.push_back(r);
    }
    return rows;
}

struct BinFile
{
    FILE *f = nullptr;
    std::string path;
    explicit BinFile(const std::string &p, const char *magic) : path(p)
    {
        f = std::fopen(p.c_str(), "rb");
        if (!f)
        {
            std::fprintf(stderr, "cannot open %s\n", p.c_str());
            std::exit(2);
        }
        char m[9] = {0};
        if (std::fread(m, 1, 8, f) != 8 || std::strcmp(m, magic) != 0)
        {
            std::fprintf(stderr, "%s: bad magic (expected %s, got %s)\n", p.c_str(), magic, m);
            std::exit(2);
        }
    }
    ~BinFile()
    {
        if (f)
            std::fclose(f);
    }
    template <typename T>
    T get()
    {
        T v;
        if (std::fread(&v, sizeof(T), 1, f) != 1)
        {
            std::fprintf(stderr, "%s: short read\n", path.c_str());
            std::exit(2);
        }
        return v;
    }
    template <typename T>
    void read(std::vector<T> &v, std::size_t n)
    {
        v.resize(n);
        if (n && std::fread(v.data(), sizeof(T), n, f) != n)
        {
            std::fprintf(stderr, "%s: short read of %zu items\n", path.c_str(), n);
            std::exit(2);
        }
    }
};

struct FactFile
{
    long n = 0, nnz = 0, minor = 0, success = 0, perm_minor = -1;
    std::vector<int32_t> outer, inner;
    std::vector<double> values;
};
static FactFile read_fact(const std::string &path)
{
    BinFile b(path, "CLFACT01");
    FactFile r;
    r.n = b.get<int64_t>();
    r.nnz = b.get<int64_t>();
    r.minor = b.get<int64_t>();
    r.success = b.get<int64_t>();
    r.perm_minor = b.get<int64_t>();
    b.read(r.outer, std::size_t(r.n + 1));
    b.read(r.inner, std::size_t(r.nnz));
    b.read(r.values, std::size_t(r.nnz));
    return r;
}

struct BlocksFile
{
    long n = 0, k = 0, cap = 0, nb = 0, n_new = 0, step = 0, outer = 0, point = 0;
    std::vector<double> Qr; ///< n x cap row-major
    struct Blk
    {
        int32_t size;
        int32_t indices[12];
        int32_t kind, is_new;
        double H[144];
    };
    std::vector<Blk> blk;
};
static BlocksFile read_blocks(const std::string &path)
{
    BinFile b(path, "CLBLK001");
    BlocksFile r;
    r.n = b.get<int64_t>();
    r.k = b.get<int64_t>();
    r.cap = b.get<int64_t>();
    r.nb = b.get<int64_t>();
    r.n_new = b.get<int64_t>();
    r.step = b.get<int64_t>();
    r.outer = b.get<int64_t>();
    r.point = b.get<int64_t>();
    b.read(r.Qr, std::size_t(r.n) * std::size_t(r.cap));
    r.blk.resize(std::size_t(r.nb));
    for (auto &x : r.blk)
    {
        x.size = b.get<int32_t>();
        for (int i = 0; i < 12; ++i)
            x.indices[i] = b.get<int32_t>();
        x.kind = b.get<int32_t>();
        x.is_new = b.get<int32_t>();
        for (int i = 0; i < 144; ++i)
            x.H[i] = b.get<double>();
    }
    return r;
}

struct ElemFile
{
    long ne = 0, step = 0, outer = 0, point = 0;
    std::vector<int32_t> ids;
    std::vector<double> B, lambda, mu, da, u;
};
static ElemFile read_elem(const std::string &path)
{
    BinFile b(path, "CLELM001");
    ElemFile r;
    r.ne = b.get<int64_t>();
    r.step = b.get<int64_t>();
    r.outer = b.get<int64_t>();
    r.point = b.get<int64_t>();
    b.read(r.ids, std::size_t(r.ne));
    b.read(r.B, std::size_t(r.ne) * 12);
    b.read(r.lambda, std::size_t(r.ne));
    b.read(r.mu, std::size_t(r.ne));
    b.read(r.da, std::size_t(r.ne));
    b.read(r.u, std::size_t(r.ne) * 12);
    return r;
}

struct CMeshFile
{
    long nv = 0, ne = 0, nf = 0, dim = 3;
    std::vector<double> rest;
    std::vector<int32_t> edges, faces;
};
static CMeshFile read_cmesh(const std::string &path)
{
    BinFile b(path, "CLCMSH01");
    CMeshFile r;
    r.nv = b.get<int64_t>();
    r.ne = b.get<int64_t>();
    r.nf = b.get<int64_t>();
    r.dim = b.get<int64_t>();
    b.read(r.rest, std::size_t(r.nv) * 3);
    b.read(r.edges, std::size_t(r.ne) * 2);
    b.read(r.faces, std::size_t(r.nf) * 3);
    return r;
}

struct SurfFile
{
    long kind = 0, nv = 0, broad_phase = 0, use_cached = 0, ccd_max_iter = 0, n_cand_logged = 0, step = 0, outer = 0, point = 0;
    double dhat = 0, dmin = 0, inflation = 0, ccd_tol = 0, alpha_logged = 0;
    std::vector<double> V0, V1;
};
static SurfFile read_surf(const std::string &path)
{
    BinFile b(path, "CLSURF01");
    SurfFile r;
    r.kind = b.get<int64_t>();
    r.nv = b.get<int64_t>();
    r.broad_phase = b.get<int64_t>();
    r.use_cached = b.get<int64_t>();
    r.ccd_max_iter = b.get<int64_t>();
    r.n_cand_logged = b.get<int64_t>();
    r.step = b.get<int64_t>();
    r.outer = b.get<int64_t>();
    r.point = b.get<int64_t>();
    r.dhat = b.get<double>();
    r.dmin = b.get<double>();
    r.inflation = b.get<double>();
    r.ccd_tol = b.get<double>();
    r.alpha_logged = b.get<double>();
    b.read(r.V0, std::size_t(r.nv) * 3);
    if (r.kind == 0)
        b.read(r.V1, std::size_t(r.nv) * 3);
    return r;
}

/// tiq_*.bin (CLTIQ001): the narrow-phase queries that the solver handed to Tight Inclusion in one CCD call, in the order of the call
struct TiqFile
{
    long nq = 0, step = 0, outer = 0, point = 0, max_iter = 0;
    double tol = 0, min_distance = 0;
    struct Q
    {
        int32_t family, nv; ///< 0 vertex-vertex, 1 edge-vertex, 2 edge-edge, 3 face-vertex; the number of vertices
        double tmax;
        double xa[12], xb[12];
    };
    std::vector<Q> q;
};
static TiqFile read_tiq(const std::string &path)
{
    BinFile b(path, "CLTIQ001");
    TiqFile r;
    r.nq = b.get<int64_t>();
    r.step = b.get<int64_t>();
    r.outer = b.get<int64_t>();
    r.point = b.get<int64_t>();
    r.max_iter = b.get<int64_t>();
    r.tol = b.get<double>();
    r.min_distance = b.get<double>();
    r.q.resize(std::size_t(r.nq));
    for (auto &x : r.q)
    {
        x.family = b.get<int32_t>();
        x.nv = b.get<int32_t>();
        x.tmax = b.get<double>();
        for (int i = 0; i < 12; ++i)
            x.xa[i] = b.get<double>();
        for (int i = 0; i < 12; ++i)
            x.xb[i] = b.get<double>();
    }
    return r;
}
