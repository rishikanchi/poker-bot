"""Compare two bot folders on the real starter-kit engine (the contest variant).

    python3 tools/ab_kit.py --poker-agent /path/to/venv/bin/poker-agent --a botA --b botB \
        --runs 16 --tables 10 --procs 4

Each run seats four copies of each bot at 8-player tables (seat order swapped
on alternate runs) and reads the points the kit reports. Prints B minus A in
points per table with a standard error across runs, and failed decisions.
"""
import argparse
import json
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor


def one(args):
    exe, a, b, tables, seed, swap = args
    order = [b, a] * 4 if swap else [a, b] * 4
    out = subprocess.run([exe, "play", *order, "--tables", str(tables), "--seed", str(seed), "--no-history"],
                         capture_output=True, text=True)
    txt = out.stdout
    data = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
    pts = {"a": [], "b": []}
    failed = {"a": 0, "b": 0}
    for i, (name, row) in enumerate(data["bots"].items()):
        who = "a" if order[i] == a else "b"
        pts[who].append(row["average_points"])
        failed[who] += row["failed_decisions"]
    return statistics.mean(pts["a"]), statistics.mean(pts["b"]), failed["a"], failed["b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poker-agent", default="poker-agent")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--runs", type=int, default=16)
    ap.add_argument("--tables", type=int, default=10)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    jobs = [(a.poker_agent, a.a, a.b, a.tables, a.seed + i, i % 2 == 1) for i in range(a.runs)]
    with ThreadPoolExecutor(a.procs) as ex:
        res = list(ex.map(one, jobs))
    diffs = [r[1] - r[0] for r in res]
    se = statistics.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else float("nan")
    print(json.dumps({
        "tables": a.runs * a.tables,
        "a_points": round(statistics.mean(r[0] for r in res), 2),
        "b_points": round(statistics.mean(r[1] for r in res), 2),
        "b_minus_a": round(statistics.mean(diffs), 2), "se": round(se, 2),
        "failed_decisions_a": sum(r[2] for r in res), "failed_decisions_b": sum(r[3] for r in res),
    }))


if __name__ == "__main__":
    main()
