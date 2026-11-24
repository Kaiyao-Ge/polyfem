#!/usr/bin/env python3
"""estimate.py: the cost estimate of a NonsmoothNewton run from its counters and measured unit prices (the external counting tool of
tools/nonsmooth_counting/counting/README.md, section 2).

    estimate = sum over cost items of (count x unit price)

Inputs
  --sim FILE       sim.json of the run (the counters of solver_info[*].info.NonsmoothNewton and .contact_term, the wall timers of PolyFEM)
  --prices FILE    unit prices, the output of replay/fit_replay.py (unit_costs_cl.json): measured on THIS machine by replaying the exported matrices, blocks, elements and CCD queries
  --counts FILE    optional, cl_counts.jsonl of an export run (the sizes: factorizations with flops and lnz, outer iterations, accepted points with their blocks, CCD calls with their
                   candidates and the Tight Inclusion queries). With it the size-dependent items use the sizes; without it they use the mean prices of the replayed records
                   (and C1-C3 and the items that need sizes fall back to the PolyFEM wall timers or are not estimated). Every row of the output says which basis it used.
  --n-elements N   elastic elements of the scene (default: num_elements of sim.json)
  --l4c-mode hot|cold|mid   the unit prices of the reduced-matrix kernels (default mid: geometric mean of hot and cold cache)
  --json FILE      also write the result as JSON

Only the Python standard library is used (runs on macOS and Linux). The comparison with the wall timers of sim.json is for reference only: the prices are measured on isolated
replays, the timers come from the run itself; they are not expected to agree to better than tens of percent, and times from different machines are not compared.

Items (counting_tool.md section 2):
  F1 numeric factorizations   F2 symbolic analyses   F3 back-substitutions (method and Newton chain)
  C1 broad phase   C2 narrow phase   C3 collision-set builds
  G1 smooth assembly (Hessian and gradient per outer iteration)   G2 re-read   E1 exact elastic elements   E2 exact contact pairs
  A1 reduced matrix   A2 subspace algebra   S1 selection (counted, not priced)   V1 true energy and validity   H1 Newton chain assembly
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict


# ----------------------------------------------------------------------------------------------------------------------------- small helpers
def get(d, *path, **kw):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return kw.get("default")
        d = d[p]
    return d


def coef(fit, name, default=0.0):
    if not fit:
        return default
    v = (fit.get("coef") or {}).get(name, default)
    return default if v is None or v != v else v


def fnum(x, default=0.0):
    try:
        v = float(x)
        return default if v != v else v
    except (TypeError, ValueError):
        return default


def parse_params(p):
    out = {}
    for kv in p.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


# ----------------------------------------------------------------------------------------------------------------------------- unit prices
class Prices:
    def __init__(self, uc, l4c_mode="mid"):
        self.uc = uc or {}
        self.mode = l4c_mode
        self.missing = defaultdict(set)

    def have(self, *path):
        return get(self.uc, *path) is not None

    # ---- L4a: CHOLMOD
    def factor_flops(self, fl):
        f = get(self.uc, "L4a", "factor")
        if not f:
            return None
        return coef(f, "ns_per_flop") * fl + coef(f, "ns_const")

    def factor(self, row):
        """numeric factorization of one fact row (failed ones: by the flops done before the failing column)"""
        L = get(self.uc, "L4a") or {}
        fl, fl_done = fnum(row.get("fl"), -1), fnum(row.get("fl_done"), -1)
        if row.get("success", True):
            return self.factor_flops(fl) if fl > 0 else self.factor_mean(True)
        if fl_done >= 0:
            ff = get(L, "failed", "fit_time_vs_flops_done")
            if ff:
                return coef(ff, "ns_per_flop_done") * fl_done + coef(ff, "ns_const")
            return self.factor_flops(fl_done)
        tab = get(L, "failed", "table_minor_frac__flops_fraction__time_fraction_of_full") or []
        n, minor = fnum(row.get("n")), fnum(row.get("minor"))
        if tab and n > 0 and fl > 0:
            x = minor / n
            xs, ys = [t[0] for t in tab], [t[2] for t in tab]
            if x <= xs[0]:
                fr = ys[0]
            elif x >= xs[-1]:
                fr = ys[-1]
            else:
                fr = ys[-1]
                for i in range(len(xs) - 1):
                    if xs[i] <= x <= xs[i + 1]:
                        w = (x - xs[i]) / max(xs[i + 1] - xs[i], 1e-300)
                        fr = ys[i] + w * (ys[i + 1] - ys[i])
                        break
            return fr * self.factor_flops(fl)
        return self.factor_mean(False)

    def summary(self, key):
        v = get(self.uc, "L4a", "summary", key)
        return None if v is None or v != v else v

    def factor_mean(self, success):
        return self.summary("factor_ns") if success else (self.summary("fail_factor_ns") or self.summary("factor_ns"))

    def analyze(self, nnz, n):
        a = get(self.uc, "L4a", "analyze")
        return None if not a else coef(a, "ns_per_nnz") * nnz + coef(a, "ns_per_n") * n

    def solve(self, lnz, hot=0.5):
        """back-substitution: hot = the share of the solves that come right after the factorization (the factor is in the cache), the others run with cold caches; the replay measures both
        (without the cold measurement: the hot price)"""
        s, sc = get(self.uc, "L4a", "solve"), get(self.uc, "L4a", "solve_cold")
        if not s or lnz is None or lnz <= 0:
            return self.summary("solve_ns")
        h = coef(s, "ns_per_lnz") * lnz + coef(s, "ns_const")
        if sc:
            c = coef(sc, "ns_per_lnz") * lnz + coef(sc, "ns_const")
            if c > 0:
                return hot * h + (1.0 - hot) * c
        return h

    # ---- L4b: CCD
    def _pooled(self, arm, key):
        return get(self.uc, "L4b", arm, key)

    def ccd(self, r, with_ti=True):
        """ns of one ccd row; None if the arm was not replayed. with_ti=False: the certificate pass only (the Tight Inclusion part of the run is then taken from the replay of all its queries, L4t)"""
        L = get(self.uc, "L4b") or {}
        kind, cand = r.get("kind"), r.get("cand") or {}
        nv, calls = fnum(r.get("nv")), max(fnum(r.get("calls"), 1), 1)
        c = dict((t, fnum(cand.get(t))) for t in ("vv", "ev", "ee", "fv"))
        nc = sum(c.values())
        if kind == "broad":
            f = L.get("broad_build")
            return None if not f else calls * coef(f, "ns_per_vertex") * nv + coef(f, "ns_per_cand") * nc
        if kind == "narrow":
            if r.get("path") == "cert":
                cp, ti = L.get("narrow_cert_pass"), L.get("narrow_ti")
                if not cp or (not ti and with_ti):
                    return None
                t = coef(cp, "ns_per_eefv_candidate") * (c["ee"] + c["fv"]) + coef(cp, "ns_per_other_candidate") * (c["vv"] + c["ev"])
                if not with_ti:
                    return t
                q = r.get("ti") or {}
                pooled = ti.get("pooled_ns_per_query") or 0.0
                for fam in ("vv", "ev", "ee", "fv"):
                    p = coef(ti, "ns_per_q_" + fam)
                    t += (p if p > 0 else pooled) * fnum(q.get(fam))
                t += pooled * fnum(q.get("other"))
                return t
            f = L.get("narrow_cached")
            if not f:
                return None
            pooled = f.get("pooled_ns_per_candidate") or 0.0
            return sum((coef(f, "ns_per_" + t) or pooled) * c[t] for t in ("vv", "ev", "ee", "fv"))
        if kind == "collisions":
            if r.get("cached"):
                f = L.get("collisions_cached")
                return None if not f else coef(f, "ns_per_candidate") * fnum(r.get("ncand_in")) + coef(f, "ns_const")
            f = L.get("collisions_build")
            return None if not f else coef(f, "ns_per_vertex") * nv * calls + coef(f, "ns_per_collision") * nc
        return None

    # ---- L4d: element and contact kernels
    def elem(self, arm):
        return get(self.uc, "L4d", "elem", arm, "ns_per_unit")

    def contact(self, typ, arm):
        return get(self.uc, "L4d", "contact_" + typ, arm, "ns_per_unit")

    def contact_mean(self, arm):
        v = [self.contact(t, arm) for t in ("ee", "fv", "ev")]
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None

    # ---- L4e: small operations (the entry whose n and k are nearest)
    def small(self, item, n=None, k=None):
        d = get(self.uc, "L4e", item) or {}
        best, bd = None, None
        for p, v in d.items():
            kv = parse_params(p)
            dist = 0.0
            if n is not None and "n" in kv:
                dist += abs(math.log(max(kv["n"], 1.0) / max(n, 1.0)))
            if k is not None and "k" in kv:
                dist += abs(kv["k"] - k) / 40.0
            if bd is None or dist < bd:
                best, bd = v.get("ns_per_unit"), dist
        return best

    # ---- L4c: reduced matrix
    def _l4c(self, mode):
        return get(self.uc, "L4c", mode) or {}

    def a1_current(self, pt):
        modes = ["hot", "cold"] if self.mode == "mid" else [self.mode]
        vals = []
        for m in modes:
            c = self._l4c(m).get("cur")
            if not c:
                return None
            vals.append(coef(c, "ns_per_block_k") * fnum(pt.get("nb")) * fnum(pt.get("k")))
        return vals[0] if len(vals) == 1 else math.sqrt(max(vals[0], 0.0) * max(vals[1], 0.0))

    def a1_merged_parts(self, pt, m):
        M = self._l4c(m)
        if not M.get("write"):
            return None
        k, R, nnz = fnum(pt.get("k")), fnum(pt.get("a1_nR")), fnum(pt.get("a1_nnzK"))
        new, written_in = fnum(pt.get("a1_new_blocks")), fnum(pt.get("a1_written_in"))
        return {
            "write": coef(M.get("write"), "ns_per_written") * written_in + coef(M.get("write"), "ns_per_nnzK") * nnz,
            "spmm": coef(M.get("spmm"), "ns_per_nnzK_k") * nnz * k,
            "gemm": coef(M.get("gemm"), "ns_per_R_k2") * R * k * k + coef(M.get("gemm"), "ns_per_R_k") * R * k,
            "gather": coef(M.get("gather"), "ns_per_R_k") * R * k,
            "new": (coef(M.get("new"), "ns_per_block_k") * k + coef(M.get("new"), "ns_per_block")) * new,
            "struct": (coef(M.get("struct"), "ns_per_raw_entry") * fnum(pt.get("a1_written")) + coef(M.get("struct"), "ns_per_R") * R) if pt.get("a1_struct_built") else 0.0,
        }

    def a1_merged(self, pt):
        modes = ["hot", "cold"] if self.mode == "mid" else [self.mode]
        parts = [self.a1_merged_parts(pt, m) for m in modes]
        if any(p is None for p in parts):
            return None
        if len(parts) == 1:
            return sum(parts[0].values())
        return sum(math.sqrt(max(parts[0][k], 0.0) * max(parts[1][k], 0.0)) for k in parts[0])


# ----------------------------------------------------------------------------------------------------------------------------- inputs
def load_sim(path):
    d = json.load(open(path))
    solves = []
    for e in d.get("solver_info", []):
        info = e.get("info", {})
        solves.append({"t": e.get("t"), "type": e.get("type"), "info": info})
    return d, solves


def load_counts(path):
    rows = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows[json.loads(line)["t"]].append(json.loads(line))
    return rows


def sim_totals(solves):
    nn = defaultdict(float)
    ct = defaultdict(float)
    timers = defaultdict(float)
    for s in solves:
        info = s["info"]
        it = max(fnum(info.get("iterations"), 0), 1)
        for k, v in (info.get("NonsmoothNewton") or {}).items():
            if isinstance(v, (int, float)):
                nn[k] += v
        for k, v in (info.get("contact_term") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                ct[k] += v
        # the timers of the solver are per iteration (solver_info = total / iterations), total_time is the total
        for k, v in info.items():
            if k.startswith("time_") and isinstance(v, (int, float)):
                timers[k] += v * it
        timers["total_time"] += fnum(info.get("total_time"))
        nn["_iterations"] += it
        nn["_solves"] += 1
    return nn, ct, timers


# ----------------------------------------------------------------------------------------------------------------------------- the estimate
class Result:
    def __init__(self):
        self.items = {}  # item -> dict(ns, basis, count, note)

    def add(self, item, ns, basis, count=None, note=""):
        r = self.items.setdefault(item, {"ns": 0.0, "basis": set(), "count": 0.0, "note": set()})
        if ns is not None:
            r["ns"] += ns
        r["basis"].add(basis)
        if count:
            r["count"] += count
        if note:
            r["note"].add(note)


def estimate(sim, solves, counts, P, n_elements, l4c_mode):
    R = Result()
    nn, ct, timers = sim_totals(solves)
    n_dofs = fnum(sim.get("num_dofs"), 0) or fnum(sim.get("mat_size"), 0)
    blind = "ev_select_calls" in nn
    own = nn["path_iterations"] + nn["line_search_iterations"] + nn["path_failures"]  # iterations in which the method formed B
    have = counts is not None
    facts = counts["fact"] if have else []
    outers = counts["outer"] if have else []
    points = counts["point"] if have else []
    ccds = counts["ccd"] if have else []
    kmean = (sum(fnum(p.get("k")) for p in points) / len(points)) if points else None
    if kmean is None:
        kmean = max(1.0, 1.0 + nn["inner_steps"] / max(nn["path_iterations"], 1.0)) if nn["path_iterations"] else 1.0
    # the size of the reduced system: the matrices handed to the factorization, else the accepted points, else the dofs of the scene
    nrow = (fnum(facts[0].get("n")) if facts else 0) or (fnum(points[0].get("n")) if points else 0) or n_dofs

    # ---- F1, F2, F3: factorizations and back-substitutions
    if have and facts:
        for r in facts:
            c = P.factor(r)
            chain = r.get("kind") == "chain"
            R.add("F1", c, "export: each factorization by its flops (fl) and, failed, by the flops done", 1, "of which Newton chain" if chain else "")
            if chain:
                R.add("F1_chain", c, "export", 1)
            if r.get("pattern_changed"):
                R.add("F2", P.analyze(fnum(r.get("nnz")), fnum(r.get("n"))), "export: the symbolic analyses that happened (pattern_changed)", 1)
            if chain:
                R.add("F3", P.solve(fnum(r.get("lnz"))), "export: one back-substitution per chain factorization, by lnz", 1)
                R.add("F3_chain", P.solve(fnum(r.get("lnz"))), "export", 1)
        for o in outers:
            nb = fnum((o.get("d") or {}).get("back_substitutions"))
            R.add("F3", nb * (P.solve(fnum(o.get("lnz"))) or 0.0), "export: the back-substitutions of each outer iteration by its lnz", nb)
        if not any(r.get("pattern_changed") for r in facts):
            R.add("F2", 0.0, "export: no analysis recorded")
    else:
        nf, nfail = nn["factorizations"], nn["factorization_failures"]
        fm, ff, sm = P.factor_mean(True), P.factor_mean(False), P.summary("solve_ns")
        if fm is not None:
            R.add("F1", (nf - nfail) * fm + nfail * (ff or fm), "mean price of the replayed records x the counters factorizations, factorization_failures (the Newton chain's factorizations are not in sim.json counters)", nf)
        if sm is not None:
            R.add("F3", nn["back_substitutions"] * sm, "mean price x the counter back_substitutions (method only; the chain's solves are not counted)", nn["back_substitutions"])
        R.add("F2", None, "not estimated: the number of analyses is not in sim.json (needs the export)")

    # ---- C1..C3: CCD and collision sets
    if have and ccds:
        for r in ccds:
            code = {"broad": "C1", "narrow": "C2", "collisions": "C3"}.get(r.get("kind"))
            if code is None:
                continue
            c = P.ccd(r)
            R.add(code, c, "export: the candidates of each call by type; narrow phase: the certificate pass plus the Tight Inclusion queries the solver really made" if code == "C2" else "export: the candidates of each call", 1,
                  "" if c is not None else "arm not replayed (no unit price)")
    else:
        for code, key in (("C1", "time_broad_phase_ccd"), ("C2", "time_ccd"), ("C3", "time_constraint_set_update")):
            R.add(code, None, "not estimated without the export (the candidates by type are not in sim.json); the PolyFEM timer is the reference column")

    # ---- G1: smooth assembly (Hessian and gradient) in the iterations that formed B; projected iterations assemble the projected Hessian again
    gh, proj, gv = P.elem("eval_grad_hess"), P.elem("project_numeric"), P.elem("eval_value")
    ne = n_elements
    if gh is not None and ne:
        if have and outers:
            exact_by_outer = {}
            for p in points:
                key = (p.get("step"), p.get("outer"))
                if key not in exact_by_outer:
                    exact_by_outer[key] = fnum(p.get("exact_el"))
            for o in outers:
                units = ne - exact_by_outer.get((o.get("step"), o.get("outer")), 0.0)
                pj = fnum((o.get("d") or {}).get("projected"))
                R.add("G1", units * gh + pj * units * (proj or 0.0), "export: elements minus the exact ones, per outer iteration (+ the projected assembly where B was projected)", 1)
        else:
            R.add("G1", own * ne * gh + nn["projected_iterations"] * ne * (proj or 0.0), "elements of the scene x the counters (iterations with B, projected_iterations); the exact elements of the blind selection are not subtracted" if blind else "elements of the scene x the counters (iterations with B, projected_iterations)", own)
    else:
        R.add("G1", None, "not estimated: no element kernel price or no element count")

    # ---- G2 / M1: re-read (the true gradient at accepted inner points: one sparse product with the smooth Hessian, Q^T r)
    rr = nn["reread_calls"] + nn["reread_after_promotion"]
    if rr:
        sp, gt = P.small("spmv", nrow), P.small("gemvT", nrow, kmean)
        nnz = P.summary("nnz") or (fnum(facts[0].get("nnz")) if facts else None)
        if sp is not None and gt is not None and nnz:
            R.add("G2", rr * (nnz * sp + nrow * kmean * gt), "counters reread_calls + reread_after_promotion x (spmv with the factorized matrix's nnz + Q^T r at k columns, k from the export or from inner_steps)", rr)
        else:
            R.add("G2", None, "not estimated: no price for spmv / gemvT or no nnz")

    # ---- E1 / E2: the exact elements and contact pairs at accepted points (derivatives) and at trial points (values)
    if have and points:
        ghp = gh
        e1 = e2 = 0.0
        ok1 = ok2 = True
        for p in points:
            ee = fnum(p.get("exact_el"))
            if ee and ghp is not None:
                e1 += ee * ghp
            pairs = p.get("exact_pairs") or {}
            for t in ("ee", "fv", "ev", "vv"):
                n_t = fnum(pairs.get(t))
                if n_t:
                    g, h = P.contact("ev" if t == "vv" else t, "gradient"), P.contact("ev" if t == "vv" else t, "hessian")
                    if g is None or h is None:
                        ok2 = False
                    else:
                        e2 += n_t * (g + h)
        R.add("E1", e1 if ok1 else None, "export: exact elements at every accepted point x the energy+gradient+Hessian kernel", len(points))
        R.add("E2", e2 if ok2 else None, "export: exact pairs by type at every accepted point x (gradient + Hessian) of the type (vertex-vertex priced as edge-vertex)", len(points))
        # trial points: the exact term's value at every rebuilt state
        trials = nn["nonsmooth_updates"]
        if trials and points:
            mean_el = sum(fnum(p.get("exact_el")) for p in points) / len(points)
            cv = P.contact_mean("value")
            mean_pairs = sum(sum(fnum(v) for v in (p.get("exact_pairs") or {}).values()) for p in points) / len(points)
            if gv is not None and cv is not None:
                R.add("E1", trials * mean_el * gv, "counter nonsmooth_updates x the mean exact elements of the accepted points x the value kernel", trials)
                R.add("E2", trials * mean_pairs * cv, "counter nonsmooth_updates x the mean exact pairs of the accepted points x the value kernel", trials)
    else:
        R.add("E1", None, "not estimated without the export (the number of exact elements is not in sim.json; the energy split has none unless the first layer lifted some)")
        R.add("E2", None, "not estimated without the export (the number of exact contact pairs is not in sim.json)")

    # ---- A1: reduced matrix, one build per accepted point
    if have and points:
        cur_all = mer_all = 0.0
        cur_ok = mer_ok = True
        form = defaultdict(int)
        for p in points:
            form[p.get("a1_form")] += 1
            if p.get("a1_form") == "global" or not p.get("nb"):
                continue
            c = P.a1_current(p)
            m = P.a1_merged(p)
            if c is None:
                cur_ok = False
            else:
                cur_all += c
            if m is None:
                mer_ok = False
            else:
                mer_all += m
        used = max(form.items(), key=lambda kv: kv[1])[0] if form else "current"
        R.add("A1", (mer_all if used == "merged_tau0" else cur_all) if (mer_ok if used == "merged_tau0" else cur_ok) else None,
              "export: the blocks of each accepted point (form of the run: %s)" % used, len(points))
        R.items["A1"]["alt"] = {"current_form_ns": cur_all if cur_ok else None, "merged_form_ns": mer_all if mer_ok else None}
    else:
        R.add("A1", None, "not estimated without the export (the blocks of the accepted points are not in sim.json)")

    # ---- A2: subspace algebra: per accepted inner step the dense products with the n x k basis (orthogonalization, model gradient, the step) and the k x k solve
    gt, gvv, dl = P.small("gemvT", nrow, kmean), P.small("gemv", nrow, kmean), P.small("dense_llt_solve", None, kmean)
    inner = nn["inner_steps"]
    if inner and gt is not None and gvv is not None:
        R.add("A2", inner * (3 * nrow * kmean * (gt + gvv) + (dl or 0.0)), "counter inner_steps x (3 gemv + 3 gemvT at n x k + one dense k x k solve)", inner)

    # ---- S1: selection (counted; the per-unit work of the selection is not priced)
    if blind:
        R.add("S1", None, "not priced: ev_select_calls = %d, ev_selected = %d (the scoring and sorting of the units is not replayed)" % (nn["ev_select_calls"], nn["ev_selected"]))

    # ---- V1: true energy evaluations (elastic energy of all elements + the contact value at the trial state) and validity
    te = nn["true_energy_evaluations"]
    cv = P.contact_mean("value")
    if te and gv is not None and ne:
        npairs = None
        if have:
            coll = [sum(fnum(v) for v in (r.get("cand") or {}).values()) for r in ccds if r.get("kind") == "collisions"]
            npairs = sum(coll) / len(coll) if coll else None
        t = te * ne * gv + (te * npairs * cv if (npairs and cv) else 0.0)
        R.add("V1", t, "counter true_energy_evaluations x (elements x value kernel" + (" + the mean collisions x the contact value kernel)" if npairs else "; the contact part is not included without the export)"), te)

    # ---- H1: assembly of the Newton chain (one projected Hessian and gradient per chain factorization)
    nchain = sum(1 for r in facts if r.get("kind") == "chain") if have else nn["fallback_iterations"]
    if nchain and gh is not None and ne:
        hpsd = P.contact_mean("hessian_psd")
        gcon = P.contact_mean("gradient")
        coll = []
        if have:
            coll = [sum(fnum(v) for v in (r.get("cand") or {}).values()) for r in ccds if r.get("kind") == "collisions"]
        mc = sum(coll) / len(coll) if coll else 0.0
        t = nchain * (ne * (proj or gh) + mc * ((hpsd or 0.0) + (gcon or 0.0)))
        R.add("H1", t, "chain iterations (export: chain factorizations; else the counter fallback_iterations) x (elements x projected-element kernel + mean collisions x contact gradient+projected Hessian)", nchain)
    return R, nn, ct, timers, blind


def measured_cpu(counts):
    """thread CPU time [s] that the export run itself measured around the same operations (rows of cl_counts.jsonl): factorizations, symbolic analyses, CCD / collision-set calls"""
    m = defaultdict(float)
    if counts is None:
        return m
    for r in counts["fact"]:
        m["F1"] += fnum(r.get("factor_cpu_ns")) * 1e-9
        m["F2"] += fnum(r.get("analyze_cpu_ns")) * 1e-9
    for r in counts["ccd"]:
        code = {"broad": "C1", "narrow": "C2", "collisions": "C3"}.get(r.get("kind"))
        if code:
            m[code] += fnum(r.get("cpu_ns")) * 1e-9
    return m


# ----------------------------------------------------------------------------------------------------------------------------- output
ORDER = ["F1", "F2", "F3", "C1", "C2", "C3", "G1", "G2", "E1", "E2", "A1", "A2", "S1", "V1", "H1"]
TIMER_OF = {"C1": ["time_broad_phase_ccd"], "C2": ["time_ccd"], "C3": ["time_constraint_set_update", "time_line_search_constraint_set_update"]}


def render(R, nn, ct, timers, blind, label, sim_path, prices_path, counts_path, mcpu):
    out = []
    out.append("estimate: %s" % label)
    out.append("  sim.json: %s" % sim_path)
    out.append("  unit prices: %s" % prices_path)
    out.append("  sizes: %s" % (counts_path if counts_path else "none (no export: mean prices and wall timers)"))
    out.append("  selection: %s; solves %d, iterations %d, iterations with B %d (path %d, line search %d, path failures %d), inner steps %d, factorizations %d (failed %d), back-substitutions %d, chain iterations %d" %
               ("blind" if blind else "energy split", nn["_solves"], nn["_iterations"], nn["path_iterations"] + nn["line_search_iterations"] + nn["path_failures"], nn["path_iterations"],
                nn["line_search_iterations"], nn["path_failures"], nn["inner_steps"], nn["factorizations"], nn["factorization_failures"], nn["back_substitutions"], nn["fallback_iterations"]))
    out.append("")
    out.append("%-4s %12s %14s %14s  %s" % ("item", "estimate [s]", "PolyFEM timer", "export run cpu", "basis"))
    tot = 0.0
    n_est = 0
    rows = []
    for it in ORDER:
        r = R.items.get(it)
        if r is None:
            continue
        # an item all of whose rows were not estimated has no number
        priced = [b for b in r["basis"] if not b.startswith("not estimated") and not b.startswith("not priced")]
        est = r["ns"] * 1e-9 if priced else None
        if est is not None:
            tot += est
            n_est += 1
        w = sum(timers.get(k, 0.0) for k in TIMER_OF.get(it, []))
        wtxt = "%.3f" % w if it in TIMER_OF else "-"
        mtxt = "%.3f" % mcpu[it] if it in mcpu else "-"
        out.append("%-4s %12s %14s %14s  %s" % (it, ("%.3f" % est) if est is not None else "-", wtxt, mtxt, "; ".join(sorted(r["basis"]))))
        for n in sorted(r["note"]):
            out.append("%-4s %12s %14s %14s    note: %s" % ("", "", "", "", n))
        if it == "A1" and r.get("alt"):
            a = r["alt"]
            out.append("%-4s %12s %14s %14s    A1 at the same points: per-block form %s s, merged form %s s" % ("", "", "", "", "%.3f" % (a["current_form_ns"] * 1e-9) if a["current_form_ns"] is not None else "-",
                                                                                                          "%.3f" % (a["merged_form_ns"] * 1e-9) if a["merged_form_ns"] is not None else "-"))
        rows.append((it, est, w if it in TIMER_OF else None, sorted(r["basis"])))
    out.append("")
    meas = timers.get("total_time", 0.0)
    out.append("sum of the estimated items: %.3f s (%d of %d items estimated; the items without a number are not in the sum)" % (tot, n_est, len(R.items)))
    out.append("measured total_time of the solves (sim.json, wall clock): %.3f s" % meas)
    ratio = tot / meas if meas > 0 else float("nan")
    out.append("estimate / measured: %.3f   (reference only: the estimate lacks the items without a number and everything outside the items, the measurement is the wall clock of the run)" % ratio)
    if mcpu:
        out.append("estimate / thread CPU time that the export run measured around the same operations (F1 F2 C1 C2 C3; the check that the unit prices reproduce the solver's own measurement):")
        out.append("  " + "  ".join("%s %.3f" % (it, R.items[it]["ns"] * 1e-9 / mcpu[it]) for it in ("F1", "F2", "C1", "C2", "C3") if it in mcpu and mcpu[it] > 0 and it in R.items and R.items[it]["ns"]))
    out.append("(the PolyFEM timer column is the solver's own timers x iterations; the broad phase, narrow phase and collision-set builds of the method's path are not inside them, so they are not a check of C1-C3)")
    return "\n".join(out), tot, meas, rows


# ----------------------------------------------------------------------------------------------------------------------------- category mode (full coverage)
# The export build measures, per solve, the exclusive thread CPU time of categories (the `cats` rows of cl_counts.jsonl: F1, A1, NT.smooth_hessian, form.elastic.value, ...) with their calls and work
# units. This mode prices every category: calls (or units) of the run x a unit price, and compares the sum with the thread CPU time of the solves minus the hook time of the export build
# (X.hook: the cost of writing the rows and files). The measured times are only the check; the prices come from
#   replay   F1 F2 F3 (CHOLMOD on the exported matrices), A1 (exported blocks), C1 C2 C3 (exported CCD calls and queries): unit_costs_cl.json
#   bench    the problem-side operations (assembly of the smooth part, energy and gradient of the forms, validity checks, exact-term evaluations, maps, selection): the `bench` rows of the bench run
#            (POLYFEM_CL_BENCH=<time step> of the same scene on the same machine: every operation repeated alone at a fixed state)
#   models   the dense algebra of the path (P.run, A2.*, G2.set, N.*): counts of gemv / gemvT / dot / spmv / axpy per unit from the code x the prices of the small-operation micro-benchmarks (L4e)
REPLAY_CATS = {"F1", "F1.chain", "F2", "F2.chain", "F3", "F3.chain", "F3.select", "A1", "C1.build", "C2.narrow", "C3.build"}
ITEM_OF = [  # (item, predicate on the category name), first match wins
    ("F1", lambda c: c in ("F1", "F1.chain")), ("F2", lambda c: c in ("F2", "F2.chain")), ("F3", lambda c: c.startswith("F3")),
    ("C1", lambda c: c in ("C1.build", "form.barrier-contact.ls_begin", "NL.line_search_begin", "NT.segment_begin")),
    ("C2", lambda c: c in ("C2.narrow", "NL.max_step_size") or c.endswith(".max_step")),
    ("C3", lambda c: c in ("C3.build", "NT.update", "form.barrier-contact.solution_changed", "NL.solution_changed")),
    ("C0", lambda c: c == "CC.displace"),
    ("G1", lambda c: c.startswith("NT.smooth_hessian") or c in ("NT.smooth_gradient", "N.assemble", "NT.exact_part")),
    ("S1", lambda c: c.startswith("S1")),
    ("A1", lambda c: c == "A1"),
    ("A2", lambda c: c.startswith("A2.") or c in ("P.run", "N.residual", "N.direction")),
    ("G2", lambda c: c.startswith("G2.") or c.endswith(".grad") or c in ("NL.gradient", "FNL.sum", "NT.gradient")),
    ("H1", lambda c: c.startswith("H1.") or c.startswith("NL.hessian") or ".hess" in c or c == "FNL.hess_sum"),
    ("V1", lambda c: c.endswith(".value") or c in ("NL.value", "NT.value", "V1.true_energy", "V1.jacobian_limit", "NT.smooth_valid_fraction", "NT.min_jacobian", "NT.min_jacobians", "NL.is_step_valid") or c.endswith(".step_valid")),
    ("E", lambda c: c.startswith("NT.") or c.startswith("P.")),
    ("M", lambda c: True),
]


def item_of(c):
    for name, f in ITEM_OF:
        if f(c):
            return name
    return "M"


def load_cats(counts):
    cats = defaultdict(lambda: {"ns": 0.0, "calls": 0.0, "units": 0.0})
    total = 0.0
    for r in counts.get("cats", []):
        total += fnum(r.get("total_cpu_ns"))
        for k, v in r["cats"].items():
            c = cats[k]
            c["ns"] += fnum(v.get("ns")); c["calls"] += fnum(v.get("calls")); c["units"] += fnum(v.get("units"))
    return cats, total


def load_bench(path):
    """-> ({category: {mode: {ns, calls, units}}}, number of rows); mode is hot or cold (rows without a mode count as hot); the selection repeats (S1.rep) are hot"""
    b = defaultdict(lambda: defaultdict(lambda: {"ns": 0.0, "calls": 0.0, "units": 0.0}))
    n = 0
    for line in open(path):
        r = json.loads(line)
        if r.get("t") == "bench":
            n += 1
            mode = r.get("mode", "hot")
            for k, v in r["cats"].items():
                c = b[k][mode]
                c["ns"] += fnum(v.get("ns")); c["calls"] += fnum(v.get("calls")); c["units"] += fnum(v.get("units"))
        elif r.get("t") == "cats":
            v = (r["cats"] or {}).get("S1.rep")
            if v:
                c = b["S1"]["hot"]
                c["ns"] += fnum(v.get("ns")); c["calls"] += fnum(v.get("calls")); c["units"] += fnum(v.get("units"))
    return b, n


def bench_price(modes, per):
    """ns per call (per = calls) or per unit (per = units) of a bench category: the geometric mean of the hot and the cold repeats (the one that exists if only one does); None if no price"""
    vals = []
    for m in ("hot", "cold"):
        v = modes.get(m)
        if v and v[per] > 0 and v["ns"] > 0:
            vals.append(v["ns"] / v[per])
    if not vals:
        return None
    return math.sqrt(vals[0] * vals[1]) if len(vals) == 2 else vals[0]


def estimate_categories(counts, bench, P, sim_nn, n_elements, export_dir=None):
    """-> (rows, total_measured_ns, hook_ns): rows = {category: dict(est_ns, meas_ns, calls, units, basis)}"""
    cats, total = load_cats(counts)
    hook = cats.get("X.hook", {}).get("ns", 0.0)
    facts, points, ccds, outers = counts["fact"], counts["point"], counts["ccd"], counts["outer"]
    nrow = fnum(facts[0].get("n")) if facts else 0.0
    prim = [r for r in facts if r.get("kind") != "chain"]
    chain = [r for r in facts if r.get("kind") == "chain"]
    kmean = (sum(fnum(p.get("k")) for p in points) / len(points)) if points else 1.0
    nnz = (sum(fnum(r.get("nnz")) for r in facts) / len(facts)) if facts else 0.0
    lnz_p = (sum(fnum(r.get("lnz")) for r in prim) / len(prim)) if prim else 0.0
    lnz_c = (sum(fnum(r.get("lnz")) for r in chain) / len(chain)) if chain else lnz_p
    inner = sum(fnum(o.get("inner_steps")) for o in outers)
    est = {}
    basis = {}

    def put(c, ns, b):
        est[c] = est.get(c, 0.0) + (ns or 0.0)
        basis[c] = b

    # ---- replay-priced
    put("F1", sum(P.factor(r) or 0.0 for r in prim), "replay: CHOLMOD time by flops, failed ones by the flops done")
    put("F1.chain", sum(P.factor(r) or 0.0 for r in chain), "replay: as F1")
    put("F2", sum((P.analyze(fnum(r.get("nnz")), fnum(r.get("n"))) or 0.0) for r in prim if r.get("pattern_changed")), "replay: analysis by nnz")
    put("F2.chain", sum((P.analyze(fnum(r.get("nnz")), fnum(r.get("n"))) or 0.0) for r in chain if r.get("pattern_changed")), "replay: analysis by nnz")
    f3 = cats.get("F3", {}).get("calls", 0.0)
    put("F3", f3 * (P.solve(lnz_p, min(1.0, len(prim) / f3) if f3 else 1.0) or 0.0), "replay: back-substitution by lnz x calls (the share right after a factorization with the factor in the cache, the rest cold)")
    put("F3.select", cats.get("F3.select", {}).get("calls", 0.0) * (P.solve(lnz_p, 0.0) or 0.0), "replay: back-substitution by lnz x calls (cold)")
    put("F3.chain", cats.get("F3.chain", {}).get("calls", 0.0) * (P.solve(lnz_c, 1.0) or 0.0), "replay: back-substitution by lnz x calls (right after its factorization)")
    cur = P.mode
    a1 = 0.0
    for p in points:
        if p.get("a1_form") == "global" or not p.get("nb"):
            continue
        a1 += (P.a1_merged(p) if p.get("a1_form") == "merged_tau0" else P.a1_current(p)) or 0.0
    put("A1", a1, "replay: the exported blocks, form of the run")
    put("C1.build", sum(P.ccd(r) or 0.0 for r in ccds if r.get("kind") == "broad"), "replay: broad phase by vertices and candidates")
    l4t = (get(P.uc, "L4t") or {}).get(export_dir) if export_dir else None
    if l4t:
        put("C2.narrow", sum(P.ccd(r, with_ti=False) or 0.0 for r in ccds if r.get("kind") == "narrow") + l4t["ti_ns_total"],
            "replay: certificate pass by candidates + replay of all the Tight Inclusion queries of the run (%d weighted)" % l4t["queries_weighted"])
    else:
        put("C2.narrow", sum(P.ccd(r) or 0.0 for r in ccds if r.get("kind") == "narrow"), "replay: certificate pass + Tight Inclusion by the per-query fit of the sampled calls")
    put("C3.build", sum(P.ccd(r) or 0.0 for r in ccds if r.get("kind") == "collisions"), "replay: collision-set builds by candidates (cached or with their own broad phase)")

    # ---- bench-priced: every other category that the bench measured
    bp = {}
    if bench:
        for c, v in bench.items():
            if c in REPLAY_CATS or c == "X.hook":
                continue
            bp[c] = v
    models_needed = []
    for c, v in cats.items():
        if c in REPLAY_CATS or c == "X.hook":
            continue
        b = bp.get(c)
        pu = bench_price(b, "units") if b else None
        pc = bench_price(b, "calls") if b else None
        if pu is not None and v["units"] > 0:
            put(c, v["units"] * pu, "bench: ns per unit of the isolated repeats (hot / cold geometric mean) x units")
        elif pc is not None:
            put(c, v["calls"] * pc, "bench: ns per call of the isolated repeats (hot / cold geometric mean) x calls")
        else:
            models_needed.append(c)

    # ---- models for the categories the bench cannot repeat (dense algebra of the path, selection)
    def small(it, n=None, k=None):
        return P.small(it, n, k) or 0.0
    gv, gt, dt, ax, sp = small("gemv", nrow, kmean), small("gemvT", nrow, kmean), small("dot", nrow), small("axpy", nrow), small("spmv", nrow)
    llt = small("dense_llt_solve", None, kmean)
    for c in models_needed:
        v = cats[c]
        calls, units = v["calls"], v["units"]
        def alg(item, k):
            h, cd = P.small(item, nrow, k), P.small(item + "_cold", nrow, k)
            if h and cd:
                # the basis Q (n x k doubles) stays in the cache between the inner steps only if it is small: above 4 MB (about the L2 cache) the cold replay is the price, below it the hot / cold mean
                return cd if nrow * max(k, 1) * 8 > 4e6 else math.sqrt(h * cd)
            return h or cd or 0.0
        keff = (lambda cnt: units / (nrow * cnt)) if nrow else (lambda cnt: kmean)
        if c == "P.run":
            rr_cols = cats.get("A2.col", {}).get("calls", 0.0) if cats.get("G2.set", {}).get("calls", 0.0) > 0 else 0.0
            ns, b = inner * alg("path_iter", keff(max(inner, 1))) + rr_cols * nnz * sp, "model: inner steps x the replay of the operation sequence of one inner step (8 products with the n x k basis, ~29 vector operations; hot / cold mean), with the re-read one more spmv with B per added column"
        elif c == "A2.col":
            ns, b = calls * alg("add_column", keff(max(calls, 1))), "model: calls x the replay of add_column (copy, spmv, k + 1 dots; hot / cold mean)"
        elif c == "A2.eval":
            ns, b = calls * alg("eval_deriv", keff(max(calls, 1))), "model: calls x the replay of the derivative part of the model evaluation (2 products, 3 vector operations; hot / cold mean)"
        elif c == "A2.eval0":
            ns, b = calls * kmean * kmean * dt * 0.5, "model: k x k"
        elif c == "A2.lm":
            ns, b = calls * llt * 2, "model: dense k x k factorizations (2 tries)"
        elif c in ("A2.alloc", "A2.delta"):
            ns, b = (units * ax * 3), "model: initialization of the basis arrays"
        elif c == "G2.set":
            ns, b = calls * alg("reread_set", keff(max(calls, 1))), "model: calls x the replay of the re-read correction (spmv, Q^T r, 5 vector operations; hot / cold mean)"
        elif c in ("N.residual", "H1.residual"):
            ns, b = units * sp + calls * 3 * nrow * ax, "model: one spmv and 3 vector operations"
        elif c == "N.assemble":
            hs = bench_price((bench or {}).get("FNL.hess_sum", {}), "units") if bench else None
            ns, b = units * 4 * sp, "model: the sparse sum of the two parts (parallel::spadd: about 4 spmv-equivalents per nonzero)"
        elif c == "S1":
            ns, b = None, "no bench price"
        else:
            ns, b = calls * 6 * nrow * ax, "model: 6 vector operations per call (glue)"
        if ns is not None:
            put(c, ns, b)
    rows = {}
    for c in set(list(cats.keys()) + list(est.keys())):
        if c == "X.hook":
            continue
        v = cats.get(c, {"ns": 0.0, "calls": 0.0, "units": 0.0})
        rows[c] = {"est_ns": est.get(c), "meas_ns": v["ns"], "calls": v["calls"], "units": v["units"], "basis": basis.get(c, "not priced")}
    return rows, total, hook


def render_categories(rows, total, hook, label, nmin=0.002):
    meas_total = total - hook
    out = ["category estimate: %s" % label,
           "  thread CPU in the solves %.3f s, of which the export hooks %.3f s; the run itself %.3f s (the reference of the check)" % (total * 1e-9, hook * 1e-9, meas_total * 1e-9)]
    items = defaultdict(lambda: {"est": 0.0, "meas": 0.0, "unpriced": 0.0, "cats": []})
    for c, r in rows.items():
        it = items[item_of(c)]
        it["meas"] += r["meas_ns"]
        if r["est_ns"] is None:
            it["unpriced"] += r["meas_ns"]
        else:
            it["est"] += r["est_ns"]
        it["cats"].append(c)
    out.append("")
    out.append("%-4s %10s %10s %7s %6s  categories" % ("item", "estimate s", "measured s", "share", "ratio"))
    tot_est = 0.0
    for name in ["F1", "F2", "F3", "C1", "C2", "C3", "C0", "G1", "S1", "A1", "A2", "G2", "H1", "V1", "E", "M"]:
        it = items.get(name)
        if not it:
            continue
        tot_est += it["est"]
        ratio = it["est"] / it["meas"] if it["meas"] > 0 else float("nan")
        flag = "" if (it["meas"] < 0.03 * meas_total or 0.8 <= ratio <= 1.25) else "  <-- outside [0.8, 1.25]"
        out.append("%-4s %10.3f %10.3f %6.1f%% %6.2f  %s%s%s" % (name, it["est"] * 1e-9, it["meas"] * 1e-9, 100 * it["meas"] / meas_total, ratio,
                                                           ",".join(sorted(it["cats"], key=lambda c: -rows[c]["meas_ns"])[:6]), "..." if len(it["cats"]) > 6 else "", flag))
        if it["unpriced"] > 0:
            out.append("     (not priced: %.3f s measured)" % (it["unpriced"] * 1e-9))
    out.append("")
    out.append("estimate %.3f s / measured %.3f s = %.3f   (coverage target >= 0.95; categories not priced count as missing)" % (tot_est * 1e-9, meas_total * 1e-9, tot_est / meas_total))
    out.append("")
    out.append("%-38s %9s %9s %6s %10s  basis" % ("category (>= 0.2% of the run)", "est s", "meas s", "ratio", "calls"))
    for c, r in sorted(rows.items(), key=lambda kv: -kv[1]["meas_ns"]):
        if r["meas_ns"] < nmin * meas_total:
            continue
        e = r["est_ns"]
        out.append("%-38s %9s %9.3f %6s %10d  %s" % (c, ("%.3f" % (e * 1e-9)) if e is not None else "-", r["meas_ns"] * 1e-9, ("%.2f" % (e / r["meas_ns"])) if (e is not None and r["meas_ns"] > 0) else "-", r["calls"], r["basis"]))
    return "\n".join(out), tot_est / meas_total, items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--counts")
    ap.add_argument("--n-elements", type=int)
    ap.add_argument("--l4c-mode", default="mid", choices=["hot", "cold", "mid"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--json")
    ap.add_argument("--export", help="the export directory of the run (default: <counts dir>/export): selects its Tight Inclusion replay in the unit price file")
    ap.add_argument("--bench", help="cl_counts.jsonl of the bench run of the same scene (POLYFEM_CL_BENCH): the prices of the problem-side operations; with it and `cats` rows in --counts the category mode is used (full coverage)")
    a = ap.parse_args()
    sim, solves = load_sim(a.sim)
    uc = json.load(open(a.prices))
    counts = load_counts(a.counts) if a.counts else None
    ne = a.n_elements if a.n_elements else int(fnum(sim.get("num_elements"), 0))
    P = Prices(uc, a.l4c_mode)
    if counts is not None and counts.get("cats"):
        bench = load_bench(a.bench)[0] if a.bench else None
        rows, total, hook = estimate_categories(counts, bench, P, None, ne, a.export or os.path.join(os.path.dirname(os.path.abspath(a.counts)), "export"))
        text, cov, items = render_categories(rows, total, hook, a.label or os.path.basename(os.path.dirname(os.path.abspath(a.sim))))
        print(text)
        if a.json:
            json.dump({"label": a.label, "coverage": cov, "measured_ns": total - hook, "hook_ns": hook, "items": {k: {"est_ns": v["est"], "meas_ns": v["meas"]} for k, v in items.items()},
                       "categories": rows}, open(a.json, "w"), indent=1)
        return
    R, nn, ct, timers, blind = estimate(sim, solves, counts, P, ne, a.l4c_mode)
    text, tot, meas, rows = render(R, nn, ct, timers, blind, a.label or os.path.basename(os.path.dirname(os.path.abspath(a.sim))), a.sim, a.prices, a.counts, measured_cpu(counts))
    print(text)
    if a.json:
        json.dump({"label": a.label, "blind": blind, "estimate_s": tot, "measured_total_time_s": meas, "ratio": tot / meas if meas > 0 else None,
                   "items": {it: {"estimate_s": est, "wall_timer_s": w, "basis": basis} for it, est, w, basis in rows}}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
