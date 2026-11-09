"""Paired comparison of two VERSIONS OF THE BOT'S CODE on identical tables.

  python3 tools/ab_code.py --old /path/to/snapshot --old-pkg pb_m0 [--variant nlhe|plo4] [--tables 250]

Make the snapshot first with tools/snapshot_bot.sh DEST NAME (it copies bot/
and renames the package so both versions can be loaded in one process).

Scenarios (X = the version under test, O = the old version):
  strong7  X + the seven strong bots            (run once with X=new, once with X=old)
  field    X + the seven caricature bots        (same)
  mixed    X, X, O, O + TAG, NashICM, ValueReg, LAG   (same)
  h2h      4 new vs 4 old in one table, played twice with the seats swapped
           (same cards), so card luck cancels
Every scenario uses the same seeds, seat rotation and payouts for both versions,
and both versions use a fixed Monte Carlo sample count, so differences come
from the code. Reports mean difference (new - old) with its standard error.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PB_NAME = {"nlhe": "holdem", "plo4": "omaha4", "plo5": "omaha5", "plo6": "omaha6"}
PAYOUTS = [(50, 30, 20), (50, 30, 20), (45, 27, 18, 10)]
SCEN = {
    "strong7": ["X", "tag", "lag", "nashicm", "valuereg", "adaptive", "balanced", "bayes"],
    "field": ["X", "nit", "station", "maniac", "jammer", "minraiser", "random", "checkfold"],
    "mixed": ["X", "X", "O", "O", "tag", "nashicm", "valuereg", "lag"],
    "h2h": ["N", "N", "N", "N", "O", "O", "O", "O"],
}
_W = {}


def _init(old_dir, old_pkg, variant, fixed, new_dir, new_pkg):
    os.environ["PB_VARIANT"] = PB_NAME[variant]
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    for p in (ROOT, os.path.join(ROOT, "stub_sdk"), os.path.join(ROOT, "bot")):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    if old_dir not in sys.path:
        sys.path.append(old_dir)  # at the END: its agent.py must not shadow the current one

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    if new_dir not in sys.path:
        sys.path.append(new_dir)
    new_agent = load("_new_agent", os.path.join(new_dir, "agent.py"))
    new_brain = importlib.import_module(new_pkg + ".brain")
    new_settings = importlib.import_module(new_pkg + ".settings")
    old_agent = load("_old_agent", os.path.join(old_dir, "agent.py"))       # package <old_pkg>
    old_brain = importlib.import_module(old_pkg + ".brain")
    old_settings = importlib.import_module(old_pkg + ".settings")
    assert new_brain.GAME is not None and old_brain.GAME is not None and new_brain is not old_brain
    from sim import bots, engine, strong_bots
    Pn = copy.deepcopy(new_settings.P)
    Po = copy.deepcopy(old_settings.P)
    Pn["mc_fixed_samples"] = fixed
    Po["mc_fixed_samples"] = fixed
    _W.update(new=(new_agent, new_brain, Pn), old=(old_agent, old_brain, Po), bots=bots,
              engine=engine, strong=strong_bots, variant=variant)


def _bot(which):
    mod, brain, P = _W[which]

    class Fixed(mod.Agent):
        def act(self, state, memory, _b=brain, _P=P):
            import poker_agent_sdk
            return _b.decide(state, memory, poker_agent_sdk, None, params=_P)
    return Fixed()


def _agent(spec, x, seed):
    if spec == "X":
        return _bot(x)
    if spec == "N":
        return _bot("new")
    if spec == "O":
        return _bot("old")
    name, _, arg = spec.partition(":")
    cls = _W["bots"].BUILTIN_BOTS.get(name) or _W["strong"].STRONG_BOTS[name]
    return cls(seed=seed, aggression=float(arg)) if arg else cls(seed=seed)


def _play(job):
    scen, seed, x = job
    lineup = SCEN[scen]
    if scen == "h2h" and x == "old":  # mirrored seating: swap which version sits where
        lineup = [{"N": "O", "O": "N"}[t] for t in lineup]
    rot = seed % len(lineup)
    specs = lineup[rot:] + lineup[:rot]
    agents = [_agent(s, x, seed * 31 + i) for i, s in enumerate(specs)]
    res = _W["engine"].play_table(agents, seed=seed, timing=False, check_invariants=False,
                                  payouts=PAYOUTS[seed % len(PAYOUTS)], variant=_W["variant"])
    pts = {t: [res.points[i] for i, s in enumerate(specs) if s == t] for t in ("X", "N", "O")}
    fails = {}
    for i, s in enumerate(specs):
        if s in ("X", "N", "O"):
            key = x if s == "X" else ("new" if s == "N" else "old")
            fails[key] = fails.get(key, 0) + res.stats[i].failed
    return scen, seed, x, {k: (float(np.mean(v)) if v else None) for k, v in pts.items()}, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--old-pkg", required=True)
    ap.add_argument("--new", default=os.path.join(ROOT, "bot"), help="default: the live bot/ folder")
    ap.add_argument("--new-pkg", default="pb")
    ap.add_argument("--variant", default="nlhe")
    ap.add_argument("--tables", type=int, default=250)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--fixed-samples", type=int, default=250)
    ap.add_argument("--scenarios", default="strong7,field,mixed,h2h")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    scens = [s for s in a.scenarios.split(",") if s in SCEN]
    jobs = []
    for sc in scens:
        for i in range(a.tables):
            sd = a.seed * 1000 + i
            if sc == "h2h":
                jobs.append((sc, sd, "new"))
                jobs.append((sc, sd, "old"))  # same cards, seats swapped
            else:
                jobs.append((sc, sd, "new"))
                jobs.append((sc, sd, "old"))
    t = time.time()
    fixed = a.fixed_samples if a.variant == "nlhe" else min(a.fixed_samples, 120)
    with mp.get_context("spawn").Pool(a.procs, initializer=_init,
                                      initargs=(os.path.abspath(a.old), a.old_pkg, a.variant, fixed,
                                                os.path.abspath(a.new), a.new_pkg)) as pool:
        out = pool.map(_play, jobs, chunksize=2)
    res = {}
    fails = {"new": 0, "old": 0}
    for sc, sd, x, pts, fl in out:
        for k, v in fl.items():
            fails[k] += v
        d = res.setdefault(sc, {}).setdefault(sd, {})
        if sc == "h2h":
            d["new"] = d.get("new", 0.0) + pts["N"] / 2.0
            d["old"] = d.get("old", 0.0) + pts["O"] / 2.0
        else:
            d[x] = pts["X"]
    report = {}
    print(f"{a.label or 'new vs old'}  [{a.variant}, {a.tables} tables/scenario, {time.time() - t:.0f}s]")
    print(f"{'scenario':10s} {'new':>8s} {'old':>8s} {'diff':>8s} {'± se':>7s}")
    tot = []
    for sc in scens:
        rows = [(v["new"], v["old"]) for v in res[sc].values() if "new" in v and "old" in v]
        n_ = np.array([r[0] for r in rows])
        o_ = np.array([r[1] for r in rows])
        d = n_ - o_
        if sc == "h2h":
            d = d / 2.0  # one table's swing counts once (new up = old down)
        se = d.std(ddof=1) / np.sqrt(len(d))
        report[sc] = {"new": float(n_.mean()), "old": float(o_.mean()), "diff": float(d.mean()), "se": float(se)}
        tot.append(d)
        print(f"{sc:10s} {n_.mean():8.2f} {o_.mean():8.2f} {d.mean():+8.2f} {se:7.2f}")
    allm = np.mean([x.mean() for x in tot])
    allse = np.sqrt(sum((x.std(ddof=1) ** 2 / len(x)) for x in tot)) / len(tot)
    print(f"{'overall':10s} {'':8s} {'':8s} {allm:+8.2f} {allse:7.2f}   failed decisions: {fails}")
    report["overall"] = {"diff": float(allm), "se": float(allse)}
    report["fails"] = fails
    if a.label:
        path = os.path.join(ROOT, "tools", "ab_results.jsonl")
        with open(path, "a") as fh:
            fh.write(json.dumps({"label": a.label, "variant": a.variant, "tables": a.tables, "report": report}) + "\n")


if __name__ == "__main__":
    main()
