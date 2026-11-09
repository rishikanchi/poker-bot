#!/usr/bin/env python3
"""Run many sit-and-go tables and print a JSON summary.

Examples:
  python3 sim/run.py --bot bot --opponents nit,station,maniac --tables 100 --seed 1
  python3 sim/run.py --bot bot --bot ../old_bot --opponents nit,jammer --procs 6
  python3 sim/run.py --bot bot --heads-up-test --tables 50

Entries (--bot and --opponents items) are built-in names (checkfold, minraiser,
station, maniac, nit, jammer, random) or paths to a folder containing agent.py
(or to a .py file) that defines ``class Agent``. The lineup is all --bot entries
followed by the opponents cycled to fill --players seats. Results are averaged
per entry label over every seat that entry occupied.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STUB = os.path.join(_REPO, "stub_sdk")
if _STUB not in sys.path:
    sys.path.insert(0, _STUB)
if _REPO not in sys.path:
    sys.path.insert(1, _REPO)

import numpy as np  # noqa: E402

from sim.bots import BUILTIN_BOTS, CheckOrFold  # noqa: E402
from sim.engine import play_table  # noqa: E402
from sim.evaluator import VARIANT_HOLE_CARDS, VARIANT_PB_NAME  # noqa: E402

DEFAULT_OPPONENTS = "nit,station,maniac,minraiser,jammer,random,checkfold"

# --------------------------------------------------------------------------- loading

_CLASS_CACHE: dict = {}

# stronger opponents (sim/strong_bots.py) are imported lazily, after PB_VARIANT is set
STRONG_LABELS = {"tag": "TAG", "lag": "LAG", "nashicm": "NashICM", "valuereg": "ValueReg",
                 "adaptive": "Adaptive", "balanced": "Balanced", "bayes": "BayesReader"}


def _builtin(key):
    if key in BUILTIN_BOTS:
        return BUILTIN_BOTS[key]
    if key in STRONG_LABELS:
        from sim import strong_bots
        return strong_bots.STRONG_BOTS[key]
    return None


def _resolve_path(spec: str) -> str:
    path = os.path.abspath(os.path.expanduser(spec))
    if not os.path.exists(path) and not os.path.isabs(os.path.expanduser(spec)):
        path = os.path.join(_REPO, spec)  # e.g. "--bot bot" from any working directory
    if os.path.isdir(path):
        path = os.path.join(path, "agent.py")
    if not os.path.isfile(path):
        raise SystemExit(f"error: {spec!r} is not a built-in bot ({', '.join(sorted(set(BUILTIN_BOTS) | set(STRONG_LABELS)))}) "
                         f"nor a folder with agent.py / a .py file")
    return path


def load_entry(spec: str):
    """(Agent class, is_builtin) for a built-in name or a bot folder / file path."""
    if spec in _CLASS_CACHE:
        return _CLASS_CACHE[spec]
    key = spec.strip().lower()
    if (key in BUILTIN_BOTS or key in STRONG_LABELS) and not os.path.exists(spec):
        out = (_builtin(key), True)
    else:
        path = _resolve_path(spec)
        folder = os.path.dirname(path)
        if folder not in sys.path:
            sys.path.insert(0, folder)
        modname = "_simbot_" + hashlib.sha1(path.encode()).hexdigest()[:12]
        mspec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(mspec)
        sys.modules[modname] = mod
        mspec.loader.exec_module(mod)
        if not hasattr(mod, "Agent"):
            raise SystemExit(f"error: {path} defines no class Agent")
        out = (mod.Agent, False)
    _CLASS_CACHE[spec] = out
    return out


def make_agent(spec: str, seed: int):
    cls, builtin = load_entry(spec)
    return cls(seed=seed) if builtin else cls()


def entry_labels(specs: list) -> dict:
    """spec -> short unique label."""
    labels = {}
    used = {}
    for spec in specs:
        if spec in labels:
            continue
        key = spec.strip().lower()
        if key in STRONG_LABELS and not os.path.exists(spec):
            base = STRONG_LABELS[key]
        elif key in BUILTIN_BOTS and not os.path.exists(spec):
            base = BUILTIN_BOTS[key].__name__
        else:
            p = os.path.abspath(os.path.expanduser(spec))
            base = os.path.basename(p[:-3] if p.endswith(".py") else p.rstrip("/"))
        lab = base
        k = 2
        while lab in used and used[lab] != spec:
            lab = f"{base}#{k}"
            k += 1
        used[lab] = spec
        labels[spec] = lab
    return labels


# --------------------------------------------------------------------------- worker

def _table_seed(base: int, t: int) -> int:
    return (base * 1_000_003 + t * 7919 + 12345) % (2 ** 31)


def _run_job(job: dict) -> dict:
    # our bot picks its variant plug-in from PB_VARIANT at import: set it before
    # the first make_agent() in this process (spawned workers inherit it anyway)
    var = job["options"].get("variant")
    if var in VARIANT_PB_NAME:
        os.environ["PB_VARIANT"] = VARIANT_PB_NAME[var]
    t = job["t"]
    seed = job["seed"]
    specs = job["seat_specs"]
    agents = [make_agent(spec, seed * 16 + s) for s, spec in enumerate(specs)]
    res = play_table(agents, seed, names=job["seat_labels"], record_hands=job.get("record", False),
                     **job["options"])
    hands = None
    if res.hands is not None:
        hands = [{
            "table": t, "seed": seed, "hand": h.hand_number, "seats": job["seat_labels"],
            "button": h.button, "sb": h.sb, "bb": h.bb,
            "blinds": [h.blinds.small, h.blinds.big, h.blinds.ante],
            "stacks_start": h.stacks_start, "hole": {str(k): list(v) for k, v in h.hole_cards.items()},
            "board": list(h.board), "stacks_end": h.stacks_end, "busted": h.busted,
            "events": [[e.kind, e.seat, e.amount, list(e.cards)] for e in h.events],
        } for h in res.hands]
    return {
        "hand_records": hands,
        "t": t, "labels": job["seat_labels"], "points": res.points, "places": res.places,
        "hands": res.hands_played, "ended_by": res.ended_by,
        "stats": [{
            "decisions": st.decisions, "failures": dict(st.failures),
            "times_ns": st.times_ns, "tracebacks": st.tracebacks,
            "illegal": st.illegal_examples, "min_bank_ms": st.min_bank_ms,
        } for st in res.stats],
    }


def _run_jobs(jobs: list, procs: int) -> list:
    if procs <= 1:
        return [_run_job(j) for j in jobs]
    chunk = max(1, len(jobs) // (procs * 8))
    with mp.get_context("spawn").Pool(procs) as pool:
        return list(pool.imap_unordered(_run_job, jobs, chunksize=chunk))


# --------------------------------------------------------------------------- summary

def _timing(times_ns) -> dict:
    if len(times_ns) == 0:
        return {}
    a = np.asarray(times_ns, dtype=np.float64) / 1e6
    return {
        "p50": round(float(np.percentile(a, 50)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
        "p99": round(float(np.percentile(a, 99)), 4),
        "p99.9": round(float(np.percentile(a, 99.9)), 4),
        "max": round(float(a.max()), 4),
        "mean": round(float(a.mean()), 4),
    }


def summarize(results: list, label_spec: dict, payouts, max_tb: int) -> dict:
    agg = {}
    for r in results:
        for s, lab in enumerate(r["labels"]):
            a = agg.setdefault(lab, {
                "points": [], "places": [], "decisions": 0,
                "failures": {"exception": 0, "illegal": 0, "timeout": 0},
                "times": [], "tracebacks": [], "illegal": [], "min_bank_ms": None})
            st = r["stats"][s]
            a["points"].append(r["points"][s])
            a["places"].append(r["places"][s])
            a["decisions"] += st["decisions"]
            for k, v in st["failures"].items():
                a["failures"][k] += v
            a["times"].extend(st["times_ns"])
            for tb in st["tracebacks"]:
                if len(a["tracebacks"]) < max_tb:
                    a["tracebacks"].append(f"table {r['t']}: {tb}")
            for il in st["illegal"]:
                if len(a["illegal"]) < max_tb:
                    a["illegal"].append(f"table {r['t']}: {il}")
            mb = st["min_bank_ms"]
            if a["min_bank_ms"] is None or mb < a["min_bank_ms"]:
                a["min_bank_ms"] = mb
    out = {}
    n_paid = len(payouts)
    for lab in sorted(agg, key=lambda k: -np.mean(agg[k]["points"])):
        a = agg[lab]
        pts = np.asarray(a["points"], dtype=np.float64)
        places = np.asarray(a["places"], dtype=np.float64)
        failed = sum(a["failures"].values())
        out[lab] = {
            "spec": label_spec.get(lab, lab),
            "seats": int(len(pts)),
            "avg_points": round(float(pts.mean()), 3),
            "points_stderr": round(float(pts.std(ddof=1) / np.sqrt(len(pts))), 3) if len(pts) > 1 else None,
            "avg_place": round(float(places.mean()), 3),
            "win_rate": round(float((places == 1).mean()), 4),
            "itm_rate": round(float((places <= n_paid).mean()), 4),
            "decisions": a["decisions"],
            "failed": failed,
            "failures": a["failures"],
            "failure_rate": round(failed / a["decisions"], 5) if a["decisions"] else 0.0,
            "timing_ms": _timing(a["times"]),
            "min_bank_ms": None if a["min_bank_ms"] is None else round(a["min_bank_ms"], 3),
            "tracebacks": a["tracebacks"],
            "illegal_examples": a["illegal"],
        }
    return out


def _write_hands(args, results):
    if not args.hands_out:
        return
    with open(args.hands_out, "w") as f:
        for r in results:
            for h in r.get("hand_records") or ():
                f.write(json.dumps(h, separators=(",", ":")) + "\n")


# --------------------------------------------------------------------------- modes

def table_options(args) -> dict:
    return {
        "starting_stack": args.starting_stack, "hands_per_level": args.hands_per_level,
        "hand_cap": args.hand_cap, "decision_ms": args.decision_ms,
        "time_bank_ms": args.time_bank_ms, "strict_timing": args.strict,
        "timing": not args.no_timing, "max_tracebacks": args.max_tracebacks,
        "variant": args.variant,
        **({"betting": args.betting} if args.betting else {}),
    }


def run_tournament(args) -> dict:
    bots = args.bot or []
    opps = [o.strip() for o in args.opponents.split(",") if o.strip()]
    n = args.players
    if len(bots) > n:
        raise SystemExit("error: more --bot entries than --players")
    if len(bots) < n and not opps:
        raise SystemExit("error: need --opponents to fill the table")
    lineup = list(bots) + [opps[i % len(opps)] for i in range(n - len(bots))]
    for spec in set(lineup):
        load_entry(spec)  # fail fast in the parent
    labels = entry_labels(lineup)
    label_spec = {v: k for k, v in labels.items()}
    opts = table_options(args)
    import random as _random
    jobs = []
    for t in range(args.tables):
        seed = _table_seed(args.seed, t)
        if args.seating == "rotate":
            seat_specs = [lineup[(s - t) % n] for s in range(n)]
        else:
            seat_specs = list(lineup)
            _random.Random(seed ^ 0x5EA7).shuffle(seat_specs)
        jobs.append({"t": t, "seed": seed, "seat_specs": seat_specs, "record": bool(args.hands_out),
                     "seat_labels": [labels[s] for s in seat_specs], "options": opts})
    t0 = time.perf_counter()
    results = _run_jobs(jobs, args.procs)
    elapsed = time.perf_counter() - t0
    results.sort(key=lambda r: r["t"])
    _write_hands(args, results)
    hands = [r["hands"] for r in results]
    return {
        "mode": "tournament", "tables": args.tables, "players": n, "seed": args.seed,
        "lineup": [labels[s] for s in lineup], "strict_timing": args.strict,
        "timing": not args.no_timing, "procs": args.procs,
        "elapsed_s": round(elapsed, 3), "tables_per_s": round(args.tables / elapsed, 2) if elapsed else None,
        "avg_hands_per_table": round(float(np.mean(hands)), 2) if hands else 0,
        "hand_cap_fraction": round(sum(r["ended_by"] == "hand_cap" for r in results) / max(1, len(results)), 4),
        "entries": summarize(results, label_spec, (50, 30, 20), args.max_tracebacks),
    }


def run_heads_up_test(args) -> dict:
    """Reproduce the upload check: bot vs CheckOrFold, heads-up, <= 20 hands, no time bank."""
    if not args.bot:
        raise SystemExit("error: --heads-up-test needs --bot")
    spec = args.bot[0]
    load_entry(spec)
    lab = entry_labels([spec])[spec]
    runs = args.tables
    jobs = []
    opts = table_options(args)
    opts.update({"hand_cap": args.hu_hands, "strict_timing": True, "timing": True})
    for t in range(runs):
        seed = _table_seed(args.seed, t)
        seat_specs = [spec, "checkfold"] if t % 2 == 0 else ["checkfold", spec]
        jobs.append({"t": t, "seed": seed, "seat_specs": seat_specs, "record": bool(args.hands_out),
                     "seat_labels": [lab if s == spec else "CheckOrFold" for s in seat_specs],
                     "options": opts})
    t0 = time.perf_counter()
    results = _run_jobs(jobs, args.procs)
    elapsed = time.perf_counter() - t0
    results.sort(key=lambda r: r["t"])
    _write_hands(args, results)
    dec = 0
    fails = {"exception": 0, "illegal": 0, "timeout": 0}
    times = []
    tbs = []
    illegal = []
    rejected = 0
    worst = 0.0
    for r in sorted(results, key=lambda r: r["t"]):
        s = r["labels"].index(lab)
        st = r["stats"][s]
        d = st["decisions"]
        f = sum(st["failures"].values())
        dec += d
        for k, v in st["failures"].items():
            fails[k] += v
        times.extend(st["times_ns"])
        rate = f / d if d else 0.0
        worst = max(worst, rate)
        if rate > 0.10:
            rejected += 1
        tbs.extend(f"run {r['t']}: {tb}" for tb in st["tracebacks"])
        illegal.extend(f"run {r['t']}: {x}" for x in st["illegal"])
    failed = sum(fails.values())
    return {
        "mode": "heads_up_test", "bot": lab, "spec": spec, "runs": runs,
        "hands_per_run": args.hu_hands, "decision_ms": args.decision_ms,
        "elapsed_s": round(elapsed, 3),
        "decisions": dec, "failed": failed, "failures": fails,
        "failure_rate": round(failed / dec, 5) if dec else 0.0,
        "worst_run_failure_rate": round(worst, 4),
        "runs_rejected": rejected,
        "would_be_rejected": rejected > 0,
        "timing_ms": _timing(times),
        "tracebacks": tbs[:args.max_tracebacks],
        "illegal_examples": illegal[:args.max_tracebacks],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bot", action="append", default=[],
                   help="bot folder with agent.py (or .py file, or built-in name); repeatable")
    p.add_argument("--opponents", default=DEFAULT_OPPONENTS,
                   help=f"comma list of built-in names / bot paths, cycled to fill seats (default {DEFAULT_OPPONENTS})")
    p.add_argument("--players", type=int, default=8)
    p.add_argument("--tables", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--strict", action="store_true", help="no time bank: any act() over decision_ms times out")
    p.add_argument("--no-timing", action="store_true", help="never time out (debugging)")
    p.add_argument("--procs", type=int, default=1, help="worker processes (timing is noisier with many)")
    p.add_argument("--heads-up-test", action="store_true",
                   help="upload check: --bot vs CheckOrFold heads-up, strict timing, --tables runs")
    p.add_argument("--hu-hands", type=int, default=20, help="hands per heads-up test run")
    p.add_argument("--seating", choices=("shuffle", "rotate"), default="shuffle",
                   help="per-table seat assignment: seeded shuffle (default) or rotation")
    p.add_argument("--starting-stack", type=int, default=2000)
    p.add_argument("--hands-per-level", type=int, default=25)
    p.add_argument("--hand-cap", type=int, default=200)
    p.add_argument("--decision-ms", type=float, default=1.0)
    p.add_argument("--time-bank-ms", type=float, default=100.0)
    p.add_argument("--max-tracebacks", type=int, default=3)
    p.add_argument("--variant", choices=sorted(VARIANT_HOLE_CARDS), default="nlhe",
                   help="nlhe (default), plo4, plo5, plo6")
    p.add_argument("--betting", choices=("no_limit", "pot_limit"), default=None,
                   help="default: no_limit for nlhe, pot_limit for plo*")
    p.add_argument("--out", help="also write the JSON summary to this file")
    p.add_argument("--hands-out", help="write every hand (cards, events, stacks) as JSON lines here")
    args = p.parse_args(argv)
    # set before load_entry() imports our bot in this process and before workers spawn
    os.environ["PB_VARIANT"] = VARIANT_PB_NAME[args.variant]

    summary = run_heads_up_test(args) if args.heads_up_test else run_tournament(args)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    return summary


if __name__ == "__main__":
    main()
