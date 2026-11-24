#!/usr/bin/env python3
"""cats_report.py: the exclusive thread-CPU categories of an export run (the `cats` rows of cl_counts.jsonl, one per solve), summed over the solves, sorted by time.
usage: cats_report.py cl_counts.jsonl [N]"""
import collections, json, sys

tot = 0.0
cats = collections.defaultdict(lambda: [0.0, 0, 0.0])
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["t"] == "cats":
        tot += r["total_cpu_ns"]
        for k, v in r["cats"].items():
            c = cats[k]
            c[0] += v["ns"]; c[1] += v["calls"]; c[2] += v["units"]
s = sum(c[0] for c in cats.values())
print("total thread cpu in the solves %.3f s; sum of the categories %.3f s (outside any category %.3f s)" % (tot / 1e9, s / 1e9, (tot - s) / 1e9))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 45
for k, c in sorted(cats.items(), key=lambda kv: -kv[1][0])[:n]:
    print("%-36s %9.3f s %5.1f%% calls %9d units %.4g" % (k, c[0] / 1e9, 100 * c[0] / tot, c[1], c[2]))
