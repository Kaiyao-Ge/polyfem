# fit_replay.py: turn the replay TSVs into unit_costs_cl.json (the unit prices judge.py multiplies the counts with).
# usage: python3 -I fit_replay.py --replay DIR --out unit_costs_cl.json   (needs numpy and scipy; the output format is that of joint/cl/B/fit_replay.py, with the narrow_cert_pass, narrow_ti and collisions_cached entries of L4b added)     (DIR holds replay_l4a.tsv, replay_l4b.tsv, replay_l4c.tsv, replay_l4d.tsv, replay_l4e.tsv, canary_*.json; any may be missing)
# Fits are relative-weighted non-negative least squares; the residual (median |relative error|) is stored with every fit.
import argparse, csv, json, math, os, sys
import numpy as np
from scipy.optimize import nnls

ap = argparse.ArgumentParser()
ap.add_argument("--replay", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = a.replay

def rows(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))

def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def wnnls(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    ok = np.isfinite(y) & (y > 0) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    if len(y) == 0:
        return None, None
    c, _ = nnls(X / y[:, None], np.ones(len(y)))
    rel = X @ c / y - 1
    return c, rel

def fit(names, X, y):
    c, rel = wnnls(X, y)
    if c is None:
        return None
    return dict(coef=dict(zip(names, map(float, c))), n=int(len(rel)), median_abs_rel=float(np.median(abs(rel))), max_abs_rel=float(abs(rel).max()))

out = {"meta": {"replay_dir": R, "unit": "ns (thread CPU time for L4a, wall for the others, single thread, taskset -c 3)"}}
for tag in ("canary_before.json", "canary_after.json"):
    p = os.path.join(R, tag)
    if os.path.exists(p):
        out["meta"][tag[:-5]] = json.load(open(p))

# ---------------------------------------------------------------- L4a
r = rows("replay_l4a.tsv")
if r:
    ok = [x for x in r if x["replay_success"] == "1"]
    bad = [x for x in r if x["replay_success"] == "0"]
    L = {"n_records": len(r), "n_success": len(ok), "n_failed": len(bad)}
    if ok:
        L["factor"] = fit(["ns_per_flop", "ns_const"], [[fnum(x["fl"]), 1] for x in ok], [fnum(x["f_cpu_med"]) for x in ok])
        L["analyze"] = fit(["ns_per_nnz", "ns_per_n"], [[fnum(x["nnz"]), fnum(x["n"])] for x in ok], [fnum(x["an_cpu_med"]) for x in ok])
        L["solve"] = fit(["ns_per_lnz", "ns_const"], [[fnum(x["lnz"]), 1] for x in ok], [fnum(x["s_cpu_med"]) for x in ok])
        if "s_cold_cpu_med" in ok[0]:
            L["solve_cold"] = fit(["ns_per_lnz", "ns_const"], [[fnum(x["lnz"]), 1] for x in ok], [fnum(x["s_cold_cpu_med"]) for x in ok])
    if ok:
        # the mean over the replayed records of the run, for estimates without the sizes of the individual factorizations (estimate.py without an export)
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        L["summary"] = dict(n_success=len(ok), n=mean([fnum(x["n"]) for x in ok]), nnz=mean([fnum(x["nnz"]) for x in ok]), fl=mean([fnum(x["fl"]) for x in ok]), lnz=mean([fnum(x["lnz"]) for x in ok]),
                            factor_ns=mean([fnum(x["f_cpu_med"]) for x in ok]), analyze_ns=mean([fnum(x["an_cpu_med"]) for x in ok]), solve_ns=mean([fnum(x["s_cpu_med"]) for x in ok]), solve_cold_ns=mean([fnum(x.get("s_cold_cpu_med")) for x in ok]),
                            fail_factor_ns=mean([fnum(x["f_cpu_med"]) for x in bad]) if bad else float("nan"), n_failed=len(bad))
    if bad and ok and L.get("factor"):
        u = L["factor"]["coef"]["ns_per_flop"]; c0 = L["factor"]["coef"]["ns_const"]
        tab = []
        for x in bad:
            n, m, fl, fd, tf = fnum(x["n"]), fnum(x["replay_minor"]), fnum(x["fl"]), fnum(x["fl_done"]), fnum(x["f_cpu_med"])
            if n > 0 and fl > 0 and fd >= 0:
                tab.append([m / n, fd / fl, tf / (u * fl + c0)])
        tab.sort()
        L["failed"] = {
            "fit_time_vs_flops_done": fit(["ns_per_flop_done", "ns_const"], [[fnum(x["fl_done"]), 1] for x in bad], [fnum(x["f_cpu_med"]) for x in bad]),
            "table_minor_frac__flops_fraction__time_fraction_of_full": tab,
            "note": "time_fraction = replay time of the failed factorization / (ns_per_flop * fl + ns_const) of the same matrix; flops_fraction = fl_done / fl",
            "perm_check_failed": int(sum(1 for x in bad if x.get("perm_ok") == "0"))}
    elif bad:
        L["failed"] = {"table_minor_frac__flops_fraction__time_fraction_of_full": [], "note": "no successful record to price the full factorization"}
    out["L4a"] = L

# ---------------------------------------------------------------- L4b
r = rows("replay_l4b.tsv")
if r:
    rec = {}
    for x in r:
        rec.setdefault(x["id"], {})[x["arm"]] = x
    L = {}
    ncand = lambda x: fnum(x["vv"]) + fnum(x["ev"]) + fnum(x["ee"]) + fnum(x["fv"])
    pooled = lambda rs, y, cnt: sum(y) / max(1.0, sum(cnt(x) for x in rs))
    for arm in ("broad_build", "narrow_cached", "narrow_cert_pass", "narrow_ti", "collisions_cached", "full_uncached"):
        rs = [v[arm] for v in rec.values() if arm in v]
        if not rs:
            continue
        y = [fnum(x["med_ns"]) for x in rs]
        if arm == "narrow_cached":
            # the stock narrow phase: Tight Inclusion on every candidate, by type
            L[arm] = fit(["ns_per_vv", "ns_per_ev", "ns_per_ee", "ns_per_fv"], [[fnum(x["vv"]), fnum(x["ev"]), fnum(x["ee"]), fnum(x["fv"])] for x in rs], y)
            L[arm]["pooled_ns_per_candidate"] = pooled(rs, y, ncand)
        elif arm == "narrow_cert_pass":
            # the certificate pass of the certified narrow phase: every edge-edge and face-vertex candidate is measured, the others only flagged
            L[arm] = fit(["ns_per_eefv_candidate", "ns_per_other_candidate"], [[fnum(x["ee"]) + fnum(x["fv"]), fnum(x["vv"]) + fnum(x["ev"])] for x in rs], y)
            L[arm]["pooled_ns_per_candidate"] = pooled(rs, y, ncand)
        elif arm == "narrow_ti":
            # Tight Inclusion on the queries that the solver handed to it, by family
            qn = lambda x: fnum(x["q_vv"]) + fnum(x["q_ev"]) + fnum(x["q_ee"]) + fnum(x["q_fv"])
            L[arm] = fit(["ns_per_q_vv", "ns_per_q_ev", "ns_per_q_ee", "ns_per_q_fv"], [[fnum(x["q_vv"]), fnum(x["q_ev"]), fnum(x["q_ee"]), fnum(x["q_fv"])] for x in rs], y)
            L[arm]["pooled_ns_per_query"] = pooled(rs, y, qn)
            L[arm]["n_queries"] = float(sum(qn(x) for x in rs))
        elif arm == "collisions_cached":
            L[arm] = fit(["ns_per_candidate", "ns_const"], [[ncand(x), 1] for x in rs], y)
            L[arm]["pooled_ns_per_candidate"] = pooled(rs, y, ncand)
        else:
            L[arm] = fit(["ns_per_vertex", "ns_per_cand"], [[fnum(x["nv"]), ncand(x)] for x in rs], y)
            L[arm]["pooled_ns_per_candidate"] = pooled(rs, y, ncand)
    rs = [v["collisions_build"] for v in rec.values() if "collisions_build" in v]
    if rs:
        L["collisions_build"] = fit(["ns_per_vertex", "ns_per_collision"], [[fnum(x["nv"]), fnum(x["vv"]) + fnum(x["ev"]) + fnum(x["ee"]) + fnum(x["fv"]) + fnum(x["pv"])] for x in rs], [fnum(x["med_ns"]) for x in rs])
    L["note"] = ("C1 broad_build per call (vertices, candidates); C2 stock: narrow_cached per candidate by type; C2 certified: narrow_cert_pass per candidate + narrow_ti per Tight Inclusion query by family "
                 "(the queries are those the solver really handed to Tight Inclusion, with their tmax); C3 collisions_build (own broad phase) and collisions_cached (from the cached candidates of the segment)")
    out["L4b"] = L

# ---------------------------------------------------------------- L4t: the Tight Inclusion queries of every recorded call of the run, replayed
r = rows("replay_l4t.tsv")
if r:
    L = {}
    for x in r:
        d = L.setdefault(os.path.abspath(x["export"]), {"calls_stored": 0, "calls_weighted": 0.0, "queries_weighted": 0.0, "ti_ns_total": 0.0})
        w = fnum(x["weight"])
        w = w if w == w and w > 0 else 1.0
        d["calls_stored"] += 1
        d["calls_weighted"] += w
        d["queries_weighted"] += w * fnum(x["nq"])
        d["ti_ns_total"] += w * fnum(x["med_ns"])
    L["note"] = "ti_ns_total: the replay time of the Tight Inclusion queries of all (weighted) calls of the run of that export directory"
    out["L4t"] = L

# ---------------------------------------------------------------- L4c
r = rows("replay_l4c.tsv")
if r:
    L = {}
    for mode in ("hot", "cold"):
        by = {}
        for x in r:
            if x["mode"] == mode:
                by.setdefault(x["arm"], []).append(x)
        M = {}
        g = lambda x, k: fnum(x[k])
        if "write" in by:
            M["write"] = fit(["ns_per_written", "ns_per_nnzK"], [[g(x, "written_old"), g(x, "nnzK")] for x in by["write"]], [g(x, "med_ns") for x in by["write"]])
        if "spmm" in by:
            M["spmm"] = fit(["ns_per_nnzK_k"], [[g(x, "nnzK") * g(x, "k")] for x in by["spmm"]], [g(x, "med_ns") for x in by["spmm"]])
        if "gemm" in by:
            M["gemm"] = fit(["ns_per_R_k2", "ns_per_R_k"], [[g(x, "nR") * g(x, "k") ** 2, g(x, "nR") * g(x, "k")] for x in by["gemm"]], [g(x, "med_ns") for x in by["gemm"]])
        if "gather" in by:
            M["gather"] = fit(["ns_per_R_k"], [[g(x, "nR") * g(x, "k")] for x in by["gather"]], [g(x, "med_ns") for x in by["gather"]])
        if "struct" in by:
            M["struct"] = fit(["ns_per_raw_entry", "ns_per_R"], [[g(x, "written_old"), g(x, "nR")] for x in by["struct"]], [g(x, "med_ns") for x in by["struct"]])
        if "new" in by:
            rs = [x for x in by["new"] if g(x, "n_new") > 0]
            if rs:
                M["new"] = fit(["ns_per_block_k", "ns_per_block"], [[g(x, "n_new") * g(x, "k"), g(x, "n_new")] for x in rs], [g(x, "med_ns") for x in rs])
        if "cur" in by:
            # per block per column of the current form on the real layout (all sizes together)
            M["cur"] = fit(["ns_per_block_k"], [[g(x, "nb") * g(x, "k")] for x in by["cur"]], [g(x, "med_ns") for x in by["cur"]])
            M["cur"]["records"] = [dict(id=x["id"], nb=g(x, "nb"), k=g(x, "k"), n6=g(x, "n6"), n9=g(x, "n9"), n12=g(x, "n12"), med_ns=g(x, "med_ns"), solver_cpu_ns=g(x, "solver_cpu_ns")) for x in by["cur"]]
        # per record table: the pieces and their sum, for the paired comparison at the same points
        per = {}
        for arm, xs in by.items():
            for x in xs:
                per.setdefault(x["id"], {})[arm] = g(x, "med_ns")
        M["records"] = per
        L[mode] = M
    L["note"] = "merged form per step = write + gather + spmm + gemm + new; struct once per outer; the current form = cur"
    out["L4c"] = L

# ---------------------------------------------------------------- L4d
r = rows("replay_l4d.tsv")
if r:
    acc = {}
    for x in r:
        acc.setdefault((x["type"], x["arm"]), []).append(fnum(x["med_ns_per_unit"]))
    L = {}
    for (t, arm), v in sorted(acc.items()):
        L.setdefault(t, {})[arm] = dict(ns_per_unit=float(np.median(v)), n_records=len(v), min=float(min(v)), max=float(max(v)))
    out["L4d"] = L

# ---------------------------------------------------------------- L4e
r = rows("replay_l4e.tsv")
if r:
    L = {}
    for x in r:
        L.setdefault(x["item"], {})[x["param"]] = dict(ns_per_unit=fnum(x["ns_per_unit_med"]), q1=fnum(x["q1"]), q3=fnum(x["q3"]))
    out["L4e"] = L

json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out)
