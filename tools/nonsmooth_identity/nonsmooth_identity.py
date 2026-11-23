#!/usr/bin/env python3
"""Same-output check for the NonsmoothNewton refactoring.

Runs a reference binary and a new binary on the same scenes with the same thread count and compares, run by run:
  - sol.bin (md5),
  - energy.csv without its first two lines (md5; the data row 0 holds an uninitialized subnormal),
  - sim.json: for every time step the solver_info entry (status, iterations, energy, gradient norm, ...) and every
    NonsmoothNewton counter, all keys except times and the uninitialized line_search_iterations.
A pair passes when all three agree exactly.

With more than one thread the reference does not reproduce itself (PolyFEM's threaded assembly sums in a
scheduler-dependent order), so no bitwise comparison is possible. There the reference is run twice and a pair passes
when both new and reference runs complete with the same number of steps and the new run differs from the reference in
the counters (iterations, inner steps, factorizations, retractions, path failures) by no more than the second reference
run does (sum of absolute differences; both given as a fraction of the reference's sum).

Examples
  nonsmooth_identity.py --ref ref/PolyFEM_bin --new new/PolyFEM_bin --out runs \\
      --scene-dir experiments/autonq/v1/scenes --scene-dir experiments/scenes/fric
  nonsmooth_identity.py --ref ref/PolyFEM_bin --out runs --scenes pin_12_c4R2   # reference runs only
  nonsmooth_identity.py --compare runs/ref/pin_12_c4R2_t1 runs/new/pin_12_c4R2_t1

Runs are kept in OUT/<ref|new>/<scene>_t<threads>; a run whose DONE marker holds the same binary md5 and steps is reused.
Only the standard library is used; nothing is tied to a platform except MKL_CBWR=COMPATIBLE, set on Linux only.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

# Scenes of the identity check (design section 5): name, time steps to run (0 = the scene's own length).
DEFAULT_SCENES = [
    ("mat3k_45_c4R2", 0),
    ("pin_12_c4R2", 0),
    ("bowl_19_it1000_c4R2", 0),
    ("trash_100_c4R2", 20),
    ("slide", 20),
    ("ball_rollers_ipc", 20),
]
PATH_KEYS = ("mesh", "common")  # keys whose relative values are paths next to the scene file
SKIP_INFO = ("line_search_iterations",)  # uninitialized in the solver info


def md5(path):
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def is_time(key):
    return "time" in key or "timer" in key or key.endswith("_seconds")


def energy_md5(d):
    p = os.path.join(d, "energy.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        lines = f.read().strip().splitlines()
    return hashlib.md5("\n".join(lines[2:]).encode()).hexdigest()


def solver_rows(d):
    """One flat dict per solver_info entry: info without times, NonsmoothNewton counters as 'NN.<key>'."""
    with open(os.path.join(d, "sim.json")) as f:
        j = json.load(f)
    rows = []
    for e in j.get("solver_info", []):
        info = e.get("info") or {}
        row = {"t": e.get("t"), "type": e.get("type")}
        for k, v in info.items():
            if is_time(k) or k in SKIP_INFO:
                continue
            if k == "NonsmoothNewton" and isinstance(v, dict):
                row.update({"NN." + a: b for a, b in v.items() if not is_time(a)})
            else:
                row[k] = v
        rows.append(row)
    return rows


def compare_dirs(a, b):
    """List of difference descriptions between two run directories (empty list = identical)."""
    diffs = []
    for name, fa, fb in (("sol.bin", md5(os.path.join(a, "sol.bin")), md5(os.path.join(b, "sol.bin"))),
                         ("energy.csv", energy_md5(a), energy_md5(b))):
        if fa is None or fb is None:
            diffs.append("%s missing (%s / %s)" % (name, fa, fb))
        elif fa != fb:
            diffs.append("%s differs" % name)
    try:
        ra, rb = solver_rows(a), solver_rows(b)
    except (OSError, ValueError) as e:
        return diffs + ["sim.json unreadable: %s" % e]
    if len(ra) != len(rb):
        diffs.append("solver_info has %d / %d entries" % (len(ra), len(rb)))
    for i, (x, y) in enumerate(zip(ra, rb)):
        for k in sorted(set(x) | set(y)):
            if x.get(k) != y.get(k):
                diffs.append("solver_info[%d] %s: %r / %r" % (i, k, x.get(k), y.get(k)))
    return diffs


def totals(d):
    """Summary of a run for the table: steps, outer iterations, NonsmoothNewton counts."""
    rows = solver_rows(d)
    t = {"steps": len(rows), "iterations": sum(r.get("iterations") or 0 for r in rows)}
    for key in ("active_iterations", "inner_steps", "factorizations", "path_failures", "retractions", "hand_backs"):
        t[key] = sum(r.get("NN." + key) or 0 for r in rows)
    return t


COUNT_KEYS = ("iterations", "NN.inner_steps", "NN.factorizations", "NN.retractions", "NN.path_failures", "NN.active_iterations")


def count_vector(d):
    rows = solver_rows(d)
    return [sum(r.get(k) or 0 for r in rows) for k in COUNT_KEYS], len(rows)


def count_distance(a, b):
    """Sum of absolute counter differences as a fraction of the sum of a's counters."""
    return sum(abs(x - y) for x, y in zip(a, b)) / max(1, sum(a))


def find_scene(name, scene_dirs):
    if os.path.isfile(name):
        return os.path.abspath(name)
    for d in scene_dirs:
        p = os.path.join(d, name if name.endswith(".json") else name + ".json")
        if os.path.isfile(p):
            return os.path.abspath(p)
    sys.exit("scene %s not found in %s" % (name, scene_dirs))


def prepare_scene(path, steps, out_dir):
    """The scene file to run: the scene itself, or, when it has to change, a copy in out_dir with absolute mesh/common
    paths. Changes: time.time_steps = steps (steps > 0); solver.nonlinear.solver = NonsmoothNewton and
    output.data.solution = sol.bin where the scene sets none (the scenes of experiments/scenes/fric do not)."""
    base = os.path.dirname(path)
    with open(path) as f:
        j = json.load(f)
    solver = j.setdefault("solver", {}).setdefault("nonlinear", {})
    data = j.setdefault("output", {}).setdefault("data", {})
    changed = steps > 0 or "solver" not in solver or "solution" not in data
    if not changed:
        return path

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in PATH_KEYS and isinstance(v, str) and not os.path.isabs(v):
                    o[k] = os.path.join(base, v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(j)
    solver.setdefault("solver", "NonsmoothNewton")
    data.setdefault("solution", "sol.bin")
    if steps > 0:
        j.setdefault("time", {})["time_steps"] = steps
    dst = os.path.join(out_dir, "scene.json")
    with open(dst, "w") as f:
        json.dump(j, f, indent=1)
    return dst


def run_one(binary, bin_md5, scene, steps, threads, out_dir, force, timeout):
    """Run one scene; returns True when the run directory holds a finished run."""
    done = os.path.join(out_dir, "DONE")
    tag = "%s %d" % (bin_md5, steps)
    if not force and os.path.exists(done) and open(done).read().strip() == tag:
        return True
    os.makedirs(out_dir, exist_ok=True)
    for f in ("DONE", "sol.bin", "sim.json", "energy.csv"):
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            os.remove(p)
    env = dict(os.environ)
    if platform.system() == "Linux":
        env["MKL_CBWR"] = "COMPATIBLE"
    cmd = [binary, "-j", prepare_scene(scene, steps, out_dir), "-o", out_dir, "--max_threads", str(threads), "--log_level", "warning"]
    with open(os.path.join(out_dir, "stdout.log"), "w") as log:
        try:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -1
    if rc != 0:
        print("  FAILED rc=%d: %s" % (rc, " ".join(cmd)), file=sys.stderr)
        return False
    with open(done, "w") as f:
        f.write(tag)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", help="reference PolyFEM binary")
    ap.add_argument("--new", help="new PolyFEM binary (omit to run the reference only)")
    ap.add_argument("--out", default="identity_runs", help="directory of the runs")
    ap.add_argument("--scene-dir", action="append", default=[], help="directory with the scene files (repeatable)")
    ap.add_argument("--scenes", nargs="*", help="scene names or files, NAME or NAME:STEPS; default: the check list")
    ap.add_argument("--threads", nargs="*", type=int, default=[1, 2, 8])
    ap.add_argument("--force", action="store_true", help="rerun even when a finished run exists")
    ap.add_argument("--timeout", type=float, default=None, help="seconds per run")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two run directories and exit")
    args = ap.parse_args()

    if args.compare:
        diffs = compare_dirs(*args.compare)
        print("identical" if not diffs else "\n".join(diffs))
        return 1 if diffs else 0
    if not args.ref:
        ap.error("--ref or --compare is required")

    scenes = DEFAULT_SCENES
    if args.scenes:
        scenes = []
        for s in args.scenes:
            name, _, n = s.partition(":")
            scenes.append((name, int(n) if n else 0))
    bins = {"ref": args.ref}
    if args.new:
        bins["new"] = args.new
    sums = {k: md5(v) for k, v in bins.items()}
    for k, v in bins.items():
        print("%s: %s  md5 %s" % (k, v, sums[k]))

    failed = 0
    results = []
    for name, steps in scenes:
        scene = find_scene(name, args.scene_dir)
        stem = os.path.splitext(os.path.basename(scene))[0]
        for th in args.threads:
            dirs = {}
            ok = True
            runs = dict(bins)
            if th > 1 and "new" in bins:
                runs["ref2"] = args.ref
            for k, b in runs.items():
                sums.setdefault(k, md5(b))
                d = os.path.join(args.out, k, "%s_t%d" % (stem, th))
                print("run %s %s threads=%d%s" % (k, stem, th, " steps=%d" % steps if steps else ""), flush=True)
                ok = run_one(b, sums[k], scene, steps, th, d, args.force, args.timeout) and ok
                dirs[k] = d
            if not ok:
                results.append({"scene": stem, "threads": th, "status": "run failed"})
                failed += 1
                continue
            res = {"scene": stem, "threads": th, "ref": totals(dirs["ref"])}
            if th > 1 and "new" in dirs:
                (cr, nr), (c2, n2), (cn, nn) = (count_vector(dirs[k]) for k in ("ref", "ref2", "new"))
                d_ref, d_new = count_distance(cr, c2), count_distance(cr, cn)
                good = nr == n2 == nn and d_new <= d_ref
                res = {"scene": stem, "threads": th, "ref": dict(zip(COUNT_KEYS, cr)), "ref2": dict(zip(COUNT_KEYS, c2)),
                       "new": dict(zip(COUNT_KEYS, cn)), "d_ref_ref": d_ref, "d_new_ref": d_new,
                       "status": "within" if good else "LARGER"}
                failed += not good
                results.append(res)
                print("  %s  new-ref %.4f, ref-ref %.4f  new %s  ref %s  ref2 %s" % (res["status"], d_new, d_ref, cn, cr, c2))
                continue
            if "new" in dirs:
                diffs = compare_dirs(dirs["ref"], dirs["new"])
                res["status"] = "identical" if not diffs else "DIFFERENT"
                res["diffs"] = diffs[:20]
                failed += bool(diffs)
            else:
                res["status"] = "ref only"
            results.append(res)
            print("  %s  %s" % (res["status"], res["ref"]))
            for x in res.get("diffs", []):
                print("    " + x)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"binaries": {k: {"path": v, "md5": sums[k]} for k, v in bins.items()}, "results": results}, f, indent=1)
    print("%d of %d pairs failed" % (failed, len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
