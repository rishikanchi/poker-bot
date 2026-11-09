"""Tune the bot's strategy parameters by simulation.

Method: cross-entropy search (sample parameter vectors around a mean, keep the
best few, move the mean toward them, shrink the spread). Every candidate in a
generation plays the SAME tables (same seeds, same seat rotation), so luck
mostly cancels when comparing them ("common random numbers").

What is tuned: strategy knobs only (open widths, push/fold threshold, ICM
strength, EQR priors, bet sizes, adaptation speed, ...). The opponent-model
priors stay at their equilibrium values (team design: start from equilibrium,
deviate on evidence).

Fitness = weighted average points of the candidate's seats over four fields:
  field   (x0.5) 1 candidate vs nit, station, maniac, jammer, min-raiser, random, check/fold
  strong7 (x1.5) 1 candidate vs the seven strong bots in sim/strong_bots.py
  vs_base (x1)   4 candidates vs 4 copies of the current settings
  mixed   (x1)   2 candidates + 2 current + TAG, NashICM, ValueReg, LAG

Usage:
  python3 tools/tune.py search --gens 10 --pop 12 --tables 16
  python3 tools/tune.py validate --params tools/tuned.json --tables 300
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "stub_sdk"), os.path.join(ROOT, "bot"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# name, low, high, integer?
SPACE = [
    ("open_scale", 0.6, 1.5, False),
    ("open_size_bb", 2.0, 3.0, False),
    ("push_fold_bb", 10.0, 20.0, False),
    ("threebet_jam_bb", 15.0, 40.0, False),
    ("fourbet_jam_bb", 25.0, 60.0, False),
    ("icm_scale", 0.0, 2.5, False),
    ("eqr_pre_ip", 0.75, 1.05, False),
    ("eqr_pre_oop", 0.60, 0.95, False),
    ("eqr_oop", 0.75, 1.00, False),
    ("multiway_call_margin", 0.0, 0.08, False),
    ("k_common", 5.0, 30.0, False),
    ("bluff_conf_z", 0.0, 1.5, False),
    ("trap_share", 0.0, 0.3, False),
    # plan: the k values, the mixing band and the all-in menu threshold are tuned too
    ("k_rare", 10.0, 60.0, False),
    ("tie_margin", 0.0, 0.06, False),
    ("tie_mix_prob", 0.0, 0.6, False),
    ("allin_spr", 1.5, 4.0, False),
    ("eqr_oop_turn", 0.80, 1.00, False),
]
# Bet sizes are fixed at 33/66/125% (team decision) and are not tuned.
NAMES = [s[0] for s in SPACE]
LO = np.array([s[1] for s in SPACE])
HI = np.array([s[2] for s in SPACE])

PAYOUTS = [(50, 30, 20), (50, 30, 20), (45, 27, 18, 10)]
SCENARIOS = {
    # name: (weight, lineup); "C" = candidate, "B" = current settings, else a sim bot
    "field": (0.5, ["C", "nit", "station", "maniac", "jammer", "minraiser", "random", "checkfold"]),
    "strong7": (1.5, ["C", "tag", "lag", "nashicm", "valuereg", "adaptive", "balanced", "bayes"]),
    "vs_base": (1.0, ["C", "C", "C", "C", "B", "B", "B", "B"]),
    "mixed": (1.0, ["C", "C", "B", "B", "tag", "nashicm", "valuereg", "lag"]),
}


def base_params():
    from pb import settings
    P = copy.deepcopy(settings.P)
    P["mc_fixed_samples"] = 250  # reproducible and fast
    return P


def current_vector():
    P = base_params()
    return np.array([P[n] for n in NAMES], float)


def make_params(x):
    P = base_params()
    for n, val in zip(NAMES, x):
        P[n] = float(val)
    return P


# ---------------------------------------------------------------- worker side
_W = {}


def _init_worker():
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    import agent as botmod  # noqa: F401  (compiles kernels once per worker)
    from sim import bots, engine
    _W["botmod"] = botmod
    _W["bots"] = bots
    _W["engine"] = engine


def _make_agent(spec, cand_P, base_P, seed):
    if spec in ("C", "B"):
        botmod = _W["botmod"]
        P = cand_P if spec == "C" else base_P

        class Tuned(botmod.Agent):
            def act(self, state, memory, _P=P):
                from pb import brain
                import poker_agent_sdk
                return brain.decide(state, memory, poker_agent_sdk, None, params=_P)
        return Tuned()
    bots = _W["bots"]
    name, _, arg = spec.partition(":")
    if name in bots.BUILTIN_BOTS:
        cls = bots.BUILTIN_BOTS[name]
    else:
        from sim import strong_bots
        cls = strong_bots.STRONG_BOTS[name]
    if arg:
        return cls(seed=seed, aggression=float(arg))
    return cls(seed=seed)


def _play(job):
    scen, seed, cand_P, base_P = job
    weight, lineup = SCENARIOS[scen]
    rot = seed % len(lineup)
    specs = lineup[rot:] + lineup[:rot]
    agents = [_make_agent(s, cand_P, base_P, seed * 31 + i) for i, s in enumerate(specs)]
    res = _W["engine"].play_table(agents, seed=seed, timing=False, check_invariants=False,
                                  payouts=PAYOUTS[seed % len(PAYOUTS)])
    c_pts = [res.points[i] for i, s in enumerate(specs) if s == "C"]
    fails = sum(st.failed for st, s in zip(res.stats, specs) if s in ("C", "B"))
    return scen, seed, float(np.mean(c_pts)), fails


def evaluate(pool, vectors, seeds_by_scen, base_P):
    """Mean points per scenario for each vector (same tables for all)."""
    jobs = []
    for vi, x in enumerate(vectors):
        P = make_params(x)
        for scen, seeds in seeds_by_scen.items():
            for sd in seeds:
                jobs.append((vi, (scen, sd, P, base_P)))
    out = pool.map(_play, [j for _, j in jobs], chunksize=1)
    res = [dict() for _ in vectors]
    fails = [0] * len(vectors)
    for (vi, _), (scen, sd, pts, f) in zip(jobs, out):
        res[vi].setdefault(scen, []).append(pts)
        fails[vi] += f
    return res, fails


def fitness(per_scen):
    tot, wsum = 0.0, 0.0
    for scen, pts in per_scen.items():
        w = SCENARIOS[scen][0]
        tot += w * float(np.mean(pts))
        wsum += w
    return tot / wsum


# ---------------------------------------------------------------- search
def search(args):
    rng = np.random.default_rng(args.seed)
    base_P = make_params(current_vector())
    x0 = current_vector()
    mean = (x0 - LO) / (HI - LO)
    sigma = np.full(len(SPACE), args.sigma)
    log_path = os.path.join(ROOT, "tools", "tune_log.jsonl")
    best = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.procs, initializer=_init_worker) as pool, open(log_path, "a") as log:
        for gen in range(args.gens):
            t = time.time()
            seeds = {s: [args.seed * 100000 + gen * 1000 + i for i in range(args.tables)] for s in SCENARIOS}
            pop = [np.clip(mean + sigma * rng.standard_normal(len(SPACE)), 0, 1) for _ in range(args.pop)]
            pop = [mean.copy(), (x0 - LO) / (HI - LO)] + pop  # current mean + the original settings
            vecs = [LO + p * (HI - LO) for p in pop]
            res, fails = evaluate(pool, vecs, seeds, base_P)
            fit = np.array([fitness(r) for r in res])
            order = np.argsort(-fit)
            elite = [pop[i] for i in order[: args.elite]]
            mean = 0.7 * np.mean(elite, 0) + 0.3 * mean
            sigma = np.maximum(0.7 * np.std(elite, 0) + 0.3 * sigma, 0.04)
            for i, (v, f, r, fl) in enumerate(zip(vecs, fit, res, fails)):
                rec = {"gen": gen, "idx": i, "fitness": round(float(f), 3), "fails": fl,
                       "params": dict(zip(NAMES, [round(float(a), 4) for a in v])),
                       "per_scen": {k: round(float(np.mean(p)), 2) for k, p in r.items()}}
                log.write(json.dumps(rec) + "\n")
                best.append(rec)
            log.flush()
            print(f"gen {gen}: best {fit[order[0]]:.2f}  mean-vector {fit[0]:.2f}  original {fit[1]:.2f}  "
                  f"fails {sum(fails)}  ({time.time() - t:.0f}s)", flush=True)
    final = dict(zip(NAMES, [round(float(a), 4) for a in LO + mean * (HI - LO)]))
    best.sort(key=lambda r: -r["fitness"])
    out = {"final_mean": final, "top": best[:5]}
    with open(os.path.join(ROOT, "tools", "tuned.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(final, indent=1))


# ---------------------------------------------------------------- validate
def validate(args):
    """Paired comparison: candidate vs current settings on fresh seeds."""
    data = json.load(open(args.params))
    cand = data.get("final_mean", data)
    x_c = np.array([cand[n] for n in NAMES], float)
    x_b = current_vector()
    base_P = make_params(x_b)
    seeds = {s: [987654 + i for i in range(args.tables)] for s in SCENARIOS}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.procs, initializer=_init_worker) as pool:
        res, fails = evaluate(pool, [x_c, x_b], seeds, base_P)
    report = {}
    for scen in SCENARIOS:
        a = np.array(res[0][scen])
        b = np.array(res[1][scen])
        d = a - b
        report[scen] = {"candidate": round(a.mean(), 2), "current": round(b.mean(), 2),
                        "diff": round(d.mean(), 2), "stderr": round(d.std(ddof=1) / np.sqrt(len(d)), 2)}
    report["fitness"] = {"candidate": round(fitness(res[0]), 2), "current": round(fitness(res[1]), 2)}
    report["fails"] = {"candidate": fails[0], "current": fails[1]}
    print(json.dumps(report, indent=1))
    with open(os.path.join(ROOT, "tools", "validation.json"), "w") as fh:
        json.dump({"candidate": cand, "report": report}, fh, indent=1)


# ---------------------------------------------------------------- ablation
# Each entry undoes ONE plan-alignment change (back to the earlier behaviour).
ABLATIONS = {
    # each entry switches ONE setting away from the current default
    "raise_bluff_prior_25pct": {"prior_raise_bluff_share": 0.25},
    "no_trapping": {"trap_share": 0.0},
    "no_multistreet_term": {"multistreet_disc": 0.0},
    "no_handcap_icm_boost": {"cap_rp_boost": 0.0},
    "no_pushfold_charts": {"use_pushfold_charts": False},
    "measured_eqr_on": {"use_measured_eqr": True},
    "jam_depths_plan_25_40": {"threebet_jam_bb": 25.0, "fourbet_jam_bb": 40.0},
    "threebet_eqr_0": {"threebet_eqr_adj": 0.0},
    "threebet_conf_z_0.5": {"threebet_conf_z": 0.5},
    "check_leak_0.05": {"range_leak_check": 0.05},
    "check_leak_0.35": {"range_leak_check": 0.35},
    "no_multiway_eqr": {"eqr_multiway": 0.0, "eqr_multiway_pos": 0.0},
    "no_postflop_icm_check": {"postflop_icm_allin": False},
    "cover_call_mult_1": {"cover_call_mult": 1.0},
}


def evaluate_P(pool, plist, seeds_by_scen, base_P):
    jobs = []
    for pi, P in enumerate(plist):
        for scen, seeds in seeds_by_scen.items():
            for sd in seeds:
                jobs.append((pi, (scen, sd, P, base_P)))
    out = pool.map(_play, [j for _, j in jobs], chunksize=2)
    res = [dict() for _ in plist]
    fails = [0] * len(plist)
    for (pi, _), (scen, sd, pts, f) in zip(jobs, out):
        res[pi].setdefault(scen, []).append(pts)
        fails[pi] += f
    return res, fails


def ablate(args):
    base_P = base_params()
    seeds = {s: [args.seed0 + i for i in range(args.tables)] for s in SCENARIOS}
    names = [n for n in ABLATIONS if not args.only or n in args.only.split(",")]
    plist = [base_P]
    for n in names:
        P = copy.deepcopy(base_P)
        P.update(copy.deepcopy(ABLATIONS[n]))
        plist.append(P)
    ctx = mp.get_context("spawn")
    t = time.time()
    with ctx.Pool(args.procs, initializer=_init_worker) as pool:
        res, fails = evaluate_P(pool, plist, seeds, base_P)
    base = res[0]
    rows = {}
    print(f"{'undo this plan change':32s} " + " ".join(f"{s:>14s}" for s in SCENARIOS) + "      overall")
    print(f"{'(plan-aligned, points)':32s} " + " ".join(f"{np.mean(base[s]):14.2f}" for s in SCENARIOS)
          + f"   {fitness(base):10.2f}")
    for n, r, fl in zip(names, res[1:], fails[1:]):
        cells = {}
        line = []
        for s in SCENARIOS:
            d = np.array(r[s]) - np.array(base[s])
            m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
            cells[s] = (round(m, 2), round(se, 2))
            line.append(f"{m:+7.2f} ±{se:4.2f}")
        tot = np.mean([np.array(r[s]) - np.array(base[s]) for s in SCENARIOS], axis=0)
        cells["overall"] = (round(float(tot.mean()), 2), round(float(tot.std(ddof=1) / np.sqrt(len(tot))), 2))
        rows[n] = {"delta": cells, "fails": fl}
        print(f"{n:32s} " + " ".join(f"{x:>14s}" for x in line) + f"   {cells['overall'][0]:+6.2f} ±{cells['overall'][1]:.2f}", flush=True)
    with open(os.path.join(ROOT, "tools", "ablation.json"), "w") as fh:
        json.dump({"tables_per_scenario": args.tables, "base": {s: float(np.mean(base[s])) for s in SCENARIOS},
                   "rows": rows}, fh, indent=1)
    print(f"done in {time.time() - t:.0f}s; positive delta = undoing the plan change scores MORE")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--gens", type=int, default=10)
    s.add_argument("--pop", type=int, default=12)
    s.add_argument("--elite", type=int, default=4)
    s.add_argument("--tables", type=int, default=16, help="tables per scenario per generation")
    s.add_argument("--sigma", type=float, default=0.22)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--procs", type=int, default=8)
    v = sub.add_parser("validate")
    v.add_argument("--params", default=os.path.join(ROOT, "tools", "tuned.json"))
    v.add_argument("--tables", type=int, default=300)
    v.add_argument("--procs", type=int, default=8)
    b = sub.add_parser("ablate")
    b.add_argument("--tables", type=int, default=250)
    b.add_argument("--procs", type=int, default=8)
    b.add_argument("--only", default="")
    b.add_argument("--seed0", type=int, default=555000)
    a = ap.parse_args()
    {"search": search, "validate": validate, "ablate": ablate}[a.cmd](a)


if __name__ == "__main__":
    main()
