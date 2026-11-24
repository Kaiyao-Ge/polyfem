// Canary benchmark (cl_spec.md V-a): a compute-bound and a memory-bound workload, a few seconds in total, one number each.
// Run before and after every PolyFEM run, pinned to the same cpu (taskset -c 3), inside the shared lock.
//   compute: 12 x 12 dense matrix product, everything in L1 (SSE2 scalar code, the compile flags of the solver)
//   gather:  random reads from a 64 MB table through a 32 MB index array (latency / cache / SMT sensitive)
//   stream:  triad a = b + s c over three 64 MB arrays (bandwidth)
// also records /proc/loadavg before and after, cpu time over wall time, and the busy fraction of the SMT sibling cpu (from /proc/stat) during the run.
// usage: canary [--out FILE] [--chunks N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <numeric>
#include <random>
#ifdef __linux__
#include <sched.h>
#endif
#include <sstream>
#include <string>
#include <vector>

static double wall_ns()
{
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::nano>(clk::now().time_since_epoch()).count();
}
static double thread_cpu_ns()
{
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return double(ts.tv_sec) * 1e9 + double(ts.tv_nsec);
}
static std::string read_line(const std::string &path)
{
    std::ifstream f(path);
    std::string l;
    std::getline(f, l);
    return l;
}
static volatile double g_sink = 0;

static double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

/// cpu -> its SMT sibling (sysfs thread_siblings_list "a,b" or "a-b"); -1 if unknown
static int sibling_of(const int cpu, std::string &source)
{
    const std::string l = read_line("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
    if (!l.empty())
    {
        std::vector<int> ids;
        std::string tok;
        std::stringstream ss(l);
        while (std::getline(ss, tok, ','))
        {
            const auto dash = tok.find('-');
            if (dash == std::string::npos)
                ids.push_back(std::atoi(tok.c_str()));
            else
                for (int c = std::atoi(tok.substr(0, dash).c_str()); c <= std::atoi(tok.substr(dash + 1).c_str()); ++c)
                    ids.push_back(c);
        }
        for (int c : ids)
            if (c != cpu)
            {
                source = "sysfs";
                return c;
            }
        source = "sysfs_no_sibling";
        return -1;
    }
    source = "guess_xor1"; // no topology files (WSL): adjacent logical cpus are siblings on this machine, a guess
    return cpu ^ 1;
}

struct CpuTimes
{
    double busy = 0, total = 0;
    bool ok = false;
};
static CpuTimes read_cpu_times(const int cpu)
{
    CpuTimes r;
    std::ifstream f("/proc/stat");
    std::string line;
    const std::string key = "cpu" + std::to_string(cpu) + " ";
    while (std::getline(f, line))
        if (line.compare(0, key.size(), key) == 0)
        {
            std::istringstream ss(line.substr(key.size()));
            double v[10] = {0};
            for (int i = 0; i < 10 && (ss >> v[i]); ++i)
                ;
            // user nice system idle iowait irq softirq steal
            r.total = v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7];
            r.busy = v[0] + v[1] + v[2] + v[5] + v[6] + v[7];
            r.ok = r.total > 0;
            break;
        }
    return r;
}

// --- workloads ------------------------------------------------------------------------------------------------------------------------
static double compute_chunk(const int reps)
{
    static double A[144], B[144], C[144];
    for (int i = 0; i < 144; ++i)
    {
        A[i] = 1.0 + 1e-3 * i;
        B[i] = 1.0 - 1e-3 * i;
    }
    const double t0 = wall_ns();
    for (int r = 0; r < reps; ++r)
    {
        for (int i = 0; i < 12; ++i)
            for (int j = 0; j < 12; ++j)
            {
                double s = 0;
                for (int k = 0; k < 12; ++k)
                    s += A[i * 12 + k] * B[k * 12 + j];
                C[i * 12 + j] = s;
            }
        // feed back so that the compiler cannot hoist the product, and keep the values bounded
        for (int i = 0; i < 144; ++i)
            A[i] = 0.5 * (A[i] + 1e-3 * C[i]);
    }
    const double t = wall_ns() - t0;
    g_sink = C[0] + A[5];
    return t / reps;
}

int main(int argc, char **argv)
{
    std::string out;
    int chunks = 5;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--out" && i + 1 < argc)
            out = argv[++i];
        else if (a == "--chunks" && i + 1 < argc)
            chunks = std::max(3, std::atoi(argv[++i]));
        else
        {
            std::fprintf(stderr, "usage: canary [--out FILE] [--chunks N]\n");
            return 2;
        }
    }
    const std::string la_before = read_line("/proc/loadavg");
#ifdef __linux__
    const int cpu = sched_getcpu();
#else
    const int cpu = -1; // no sched_getcpu on macOS; the /proc and /sys readings below are empty there (sibling_busy -1)
#endif
    std::string sib_source;
    const int sib = cpu >= 0 ? sibling_of(cpu, sib_source) : -1;
    const CpuTimes s0 = sib >= 0 ? read_cpu_times(sib) : CpuTimes();
    const double w0 = wall_ns(), c0 = thread_cpu_ns();

    // compute-bound: calibrate the repetitions for about 0.25 s per chunk, then chunks x
    int creps = 20000;
    while (compute_chunk(creps) * creps < 0.05e9 && creps < (1 << 28))
        creps *= 2;
    creps = int(std::max<double>(1000.0, creps * 0.25e9 / std::max(1.0, compute_chunk(creps) * creps)));
    std::vector<double> tc;
    for (int c = 0; c < chunks; ++c)
        tc.push_back(compute_chunk(creps));

    // memory-bound: gather
    const std::size_t NT = std::size_t(8) << 20; // 8M doubles = 64 MB
    const std::size_t NI = std::size_t(4) << 20; // 4M indices (uint64) = 32 MB
    std::vector<double> table(NT);
    std::vector<std::uint64_t> idx(NI);
    {
        std::mt19937_64 g(12345);
        for (std::size_t i = 0; i < NT; ++i)
            table[i] = 1.0 + double(i % 1000) * 1e-6;
        for (std::size_t i = 0; i < NI; ++i)
            idx[i] = g() % NT;
    }
    std::vector<double> tg;
    for (int c = 0; c < chunks; ++c)
    {
        const double t0 = wall_ns();
        double s = 0;
        for (std::size_t i = 0; i < NI; ++i)
            s += table[idx[i]];
        tg.push_back((wall_ns() - t0) / double(NI));
        g_sink = s;
    }
    // stream triad
    const std::size_t NS = std::size_t(8) << 20;
    std::vector<double> a(NS, 0.0), b(NS, 1.0), cc(NS, 2.0);
    std::vector<double> ts;
    for (int c = 0; c < chunks; ++c)
    {
        const double t0 = wall_ns();
        const double sc = 1.0 + 1e-3 * c;
        for (std::size_t i = 0; i < NS; ++i)
            a[i] = b[i] + sc * cc[i];
        ts.push_back((wall_ns() - t0) / double(NS));
        g_sink = a[NS / 2];
    }

    const double w1 = wall_ns(), c1 = thread_cpu_ns();
    const CpuTimes s1 = sib >= 0 ? read_cpu_times(sib) : CpuTimes();
    double busy = -1;
    if (s0.ok && s1.ok && s1.total > s0.total)
        busy = (s1.busy - s0.busy) / (s1.total - s0.total);
    const std::string la_after = read_line("/proc/loadavg");

    char buf[1200];
    std::snprintf(buf, sizeof buf,
                  "{\"compute_ns_per_op\": %.4f, \"stream_ns_per_elem\": %.5f, \"gather_ns_per_elem\": %.5f, "
                  "\"cpu_over_wall\": %.4f, \"loadavg_before\": \"%s\", \"loadavg_after\": \"%s\", "
                  "\"cpu\": %d, \"sibling_cpu\": %d, \"sibling_busy\": %.4f, \"sibling_source\": \"%s\", \"wall_s\": %.3f}\n",
                  median(tc), median(ts), median(tg), (c1 - c0) / (w1 - w0), la_before.c_str(), la_after.c_str(), cpu, sib, busy, sib_source.c_str(), (w1 - w0) * 1e-9);
    std::fputs(buf, stdout);
    if (!out.empty())
    {
        std::ofstream f(out);
        f << buf;
    }
    return 0;
}
