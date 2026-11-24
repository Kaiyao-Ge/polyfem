# prints the compile flags (defines, include paths, -O3 ...) of a source file of a PolyFEM build, or the link libraries of PolyFEM_bin, taken from `ninja -t commands`
# (reads the build tree only). usage: flags_from_build.py compile <file.cpp> | includes <file> | link   (includes: only the -I / -isystem paths, e.g. of a SuiteSparse source file)
# The build directory is $CL_BUILD_DIR (default: the export build next to this directory, ../../export/build).
import os, shlex, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.environ.get("CL_BUILD_DIR") or os.path.normpath(os.path.join(HERE, "..", "..", "export", "build"))
mode = sys.argv[1]
if mode in ("compile", "includes"):
    cmds = subprocess.run(["ninja", "-C", B, "-t", "commands"], capture_output=True, text=True).stdout.splitlines()
    line = next(l for l in cmds if (sys.argv[2] + ".o") in l and " -c " in l)
    t = shlex.split(line)
    out, i = [], 0
    while i < len(t):
        a = t[i]
        if a in ("-isystem", "-I"):
            out.append(a); out.append(t[i + 1]); i += 2; continue
        if mode == "includes":
            if a.startswith("-I") and len(a) > 2:
                out.append(a)
            i += 1
            continue
        if a.startswith("-isystem") or a.startswith("-I") or a.startswith("-D") or a in ("-O3", "-DNDEBUG", "-fPIC", "-fopenmp", "-fno-omit-frame-pointer", "-fno-optimize-sibling-calls") or a.startswith("-std="):
            out.append(a)
        i += 1
    print(" ".join(shlex.quote(x) for x in out))
else:
    line = subprocess.run(["ninja", "-C", B, "-t", "commands", "PolyFEM_bin"], capture_output=True, text=True).stdout.splitlines()[-1]
    t = shlex.split(line)
    out = []
    skip = ("libpolyfem", "libpolysolve", "liblinear_spec", "libnon_linear_spec", "libpython", ".o")
    for a in t:
        if a.startswith("-l") or a in ("-fopenmp",) or a.endswith(".a") or a.endswith(".so"):
            if any(s in a for s in skip):
                continue
            if not a.startswith("-") and not a.startswith("/"):
                a = B + "/" + a
            out.append(a)
    print(" ".join(shlex.quote(x) for x in out))
