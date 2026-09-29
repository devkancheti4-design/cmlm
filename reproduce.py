"""Regenerate every table of the paper from the cmlm package.

    python3 reproduce.py            # all sections, writes results.json
    python3 reproduce.py t9 t10     # only the named sections

No API key and no network: the world is synthetic and seeded, the frontier is
scripted, training and recall are real. The exact memory (life/life.py from
living-fused, MIT) must be importable; set LIFE_PATH to its directory if it is
not beside this file. Wall times depend on the machine; every other number is
deterministic for a given seed.
"""
import asyncio, json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from cmlm.world import World, measure, measure_seeds, by_age
from cmlm.model import CMLM, features
from cmlm.frontier import est_tokens
from cmlm.toolmodel import KeyHead

OUT = os.environ.get("CMLM_RESULTS", os.path.join(HERE, "results.json"))
T975 = {4: 2.776, 19: 2.093, 39: 2.023}          # two-sided 95% t quantiles for n-1 degrees of freedom


def stat(xs):
    n = len(xs); m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    h = T975[n - 1] * sd / math.sqrt(n)
    return {"n": n, "mean": round(m, 4), "sd": round(sd, 4), "ci95": [round(m - h, 4), round(m + h, 4)],
            "min": round(min(xs), 4), "max": round(max(xs), 4)}


def per_seed(res):
    runs = res["runs"]
    return {"final": [r["final_recall"]["trained"] for r in runs],
            "old": [by_age(r["probes"])["old"]["trained"] for r in runs],
            "sent": [r["final"]["sent_tokens"] for r in runs]}


def summary(res, turns):
    out = {"final_recall": res["final_recall"], "by_age": res["by_age"], "recall": res["recall"],
           "final": res["final"], "wall_ms_per_turn": round(res["wall_ms_mean"] / turns, 1),
           "wall_s_per_session": round(res["wall_ms_mean"] / 1000, 1), "self_recall_last": res["self_recall_last"],
           "worst_final": res["worst_final"], "trained_beats_matching": res["trained_beats_matching"],
           "exact": res["exact"], "tree": res["tree"], "toolmodel": res["toolmodel"]}
    if res["cascade"]:
        out["cascade"] = {k: v for k, v in res["cascade"].items() if k != "per_level"}
    ps = per_seed(res)
    out["per_seed"] = {k: stat(v) for k, v in ps.items()}
    return out, ps


def paired(shape_ps, flat_ps):
    d = [a - b for a, b in zip(shape_ps["final"], flat_ps["final"])]
    s = stat(d); s["wins"] = sum(1 for x in d if x > 0); s["of"] = len(d)
    return s


def shapes_kw(shape):
    return {"flat": {}, "tree": {"tree": True, "theta": 0.0}, "recursive": {"recursive": True}}[shape]


async def run(name, seeds, **kw):
    t0 = time.time()
    res = await measure_seeds(seeds, **kw)
    print(f"  {name}: {len(seeds)} seeds in {time.time() - t0:.0f} s", flush=True)
    return res


async def t4():
    out = {}
    for label, kw in (("30 turns, k=4, paraphrase", dict(turns=30, k=4)),
                      ("30 turns, k=4, overlap", dict(turns=30, k=4, overlap=True)),
                      ("60 turns, k=4, paraphrase", dict(turns=60, k=4))):
        res = await run(label, range(1, 41), **kw)
        out[label], _ = summary(res, kw["turns"])
    return out


async def t5():
    out = {}
    for turns, k in ((60, 6), (90, 9), (120, 12)):
        for shape in ("flat", "tree", "recursive"):
            label = f"{turns} turns, k={k}, {shape}"
            res = await run(label, range(1, 21), turns=turns, k=k, **shapes_kw(shape))
            out[label], _ = summary(res, turns)
    return out


async def t6():
    out = {}
    for shape in ("flat", "tree", "recursive"):
        label = f"60 turns, k=6, {shape} + Life"
        res = await run(label, range(1, 21), turns=60, k=6, fused=True, **shapes_kw(shape))
        out[label], _ = summary(res, 60)
    return out


def key_head_curve(seeds=(1, 2, 3), points=(30, 60, 90, 120), test=30):
    rows = []
    for sd in seeds:
        w = World(seed=sd, turns=max(points) + test)
        facts = [(t["fact"], t["key"]) for t in w.turns]
        for n in points:
            h = KeyHead(seed=sd)
            for f, k in facts[:n]:
                h.learn(f, k)
            preds = [h.predict(f) for f, _ in facts[n:n + test]]
            ok = sum(1 for (p, c), (_, k) in zip(preds, facts[n:n + test]) if p == k)
            commit = sum(1 for p, c in preds if c >= 0.6)
            rows.append({"seed": sd, "examples": n, "correct": ok, "committed": commit, "of": test})
    curve = {n: round(sum(r["correct"] for r in rows if r["examples"] == n) / (len(seeds) * test), 3) for n in points}
    return {"rows": rows, "mean_accuracy": curve}


async def t7():
    out = {}
    res = await run("60 turns, tools every turn", range(1, 21), turns=60, k=6, fused=True, toolmodel=True, warmup=30, gate=10.0)
    out["tools every turn"], _ = summary(res, 60)
    res = await run("60 turns, permissive gate", range(1, 21), turns=60, k=6, fused=True, toolmodel=True, warmup=30, gate=0.0)
    out["permissive gate"], _ = summary(res, 60)
    res = await run("120 turns, prequential gate 0.85", range(1, 21), turns=120, k=12, fused=True, toolmodel=True, warmup=30)
    s, _ = summary(res, 120)
    s["switched"] = sum(1 for x in res["toolmodel"]["switched_at"] if x is not None)
    out["prequential gate 0.85, 120 turns"] = s
    out["key head alone"] = key_head_curve()
    return out


async def t9():
    out, ps = {}, {}
    for shape in ("flat", "tree", "recursive"):
        res = await run(f"60 turns, k=6, {shape} (per seed)", range(1, 21), turns=60, k=6, **shapes_kw(shape))
        out[shape], ps[shape] = summary(res, 60)
    for shape in ("tree", "recursive"):
        out[shape]["vs_flat"] = paired(ps[shape], ps["flat"])
    return out


def checkpoints(run, every=50):
    rows = []
    for t in range(every, run["turns"] + 1, every):
        ps = [p for p in run["probes"] if p["turn"] == t]
        n = sum(p["truth"] for p in ps)
        rec = sum(p["trained"] for p in ps) / n
        ceil = sum(min(p["k"], p["truth"]) for p in ps) / n
        rows.append({"turn": t, "recall": round(rec, 4), "ceiling": round(ceil, 4), "share": round(rec / ceil, 4)})
    return rows


async def t10():
    out, ps = {}, {}
    for shape in ("flat", "tree", "recursive"):
        res = await run(f"300 turns documents, {shape} + Life", range(1, 6), turns=300, k=12, bulk=250, fused=True, **shapes_kw(shape))
        out[shape], ps[shape] = summary(res, 300)
        r1 = res["runs"][0]
        out[shape]["seed1_checkpoints"] = checkpoints(r1)
        # one log entry per teaching turn; number them by world turn (the log's own "t" counts every question asked)
        out[shape]["seed1_per_turn"] = [{"t": i + 1, "sent": x["sent_tokens"], "transcript": x["world_transcript_tokens"]}
                                        for i, x in enumerate(r1["turn_log"])]
    for shape in ("tree", "recursive"):
        out[shape]["vs_flat"] = paired(ps[shape], ps["flat"])
    return out


def vector_hop():
    w = World(seed=1, turns=60)
    teach = [(t["question"], t["fact"]) for t in w.turns[:60]]
    probes = [t["question"] for t in w.turns[:60]] + ["deadline for the review", "who signed the northern lease",
                                                      "amount on the last invoice"]
    a, b = CMLM(seed=0), CMLM(seed=0)
    for n, (q, f) in enumerate(teach):
        a.teach([(q, f)], n)
        b.teach([(features(q), f)], n)
    same = sum(1 for p in probes if [r["fact"] for r in a.readout(p, 4)] == [r["fact"] for r in b.readout(p, 4)])
    diff = max(abs(x["score"] - y["score"]) for p in probes for x, y in zip(a.readout(p, 4), b.readout(p, 4)))
    return {"weights_identical": bool(np.array_equal(a.Wq, b.Wq) and np.array_equal(a.Wf, b.Wf)),
            "readouts_identical": same, "probes": len(probes), "max_score_diff": float(diff),
            "word_hop_tokens": sum(est_tokens(q) for q, _ in teach), "vector_hop_tokens": 0}


def real_session(path=os.path.join(HERE, "data", "real_session_sizes.csv"), R=164, K=140, J=30, r=0.1, w=1.25):
    """The cost model of the paper applied to the per-turn sizes of a real session."""
    import csv
    rows = list(csv.DictReader(open(path)))
    prev, D, out = 0, 0, []
    for x in rows:
        Q, cum = int(x["question_tokens"]), int(x["cumulative_tokens"])
        P = prev
        out.append({"turn": int(x["turn"]), "cold": P + Q, "warm": r * (P - D) + w * D + Q, "cmlm": 2 * R + 2 * Q + K + J})
        D, prev = cum - prev, cum
    tot = {k: round(sum(o[k] for o in out)) for k in ("cold", "warm", "cmlm")}
    return {"turns": len(out), "totals": tot, "per_turn": out,
            "warm_cheaper_than_cmlm_on_turns": [o["turn"] for o in out if o["warm"] < o["cmlm"]]}


async def ablations():
    """The alternatives the paper reports as measured worse: the tree's self-recall trigger, and the cascade's
    normalised merge and hand-down on failure, each against the setting used in the paper, 60 turns, k = 6, 20 seeds."""
    out, ps = {}, {}
    for label, kw in (("tree, cap only", {"tree": True, "theta": 0.0}),
                      ("tree, self-recall trigger 0.9", {"tree": True, "theta": 0.9}),
                      ("cascade, rank interleave", {"recursive": True}),
                      ("cascade, normalised merge", {"recursive": True, "merge": "z"}),
                      ("cascade, hand down on failure", {"recursive": True, "demote_failing": True})):
        res = await run(label, range(1, 21), turns=60, k=6, **kw)
        out[label], ps[label] = summary(res, 60)
    for a, b in (("tree, self-recall trigger 0.9", "tree, cap only"), ("cascade, normalised merge", "cascade, rank interleave"),
                 ("cascade, hand down on failure", "cascade, rank interleave")):
        out[a]["vs_paper_setting"] = paired(ps[a], ps[b])
    return out


async def sweep():
    """The learning rate and temperature grid, 30 turns, k = 4, eight seeds (41 to 48, disjoint from the reported seeds)."""
    rows = []
    for lr in (0.02, 0.05, 0.1, 0.2):
        for tau in (0.05, 0.1, 0.2, 0.5):
            vals = []
            for sd in range(41, 49):
                r = await measure(seed=sd, turns=30, k=4, cmlm=CMLM(seed=sd, lr=lr, tau=tau))
                vals.append(r["final_recall"]["trained"])
            rows.append({"lr": lr, "tau": tau, "final_recall_mean": round(sum(vals) / len(vals), 4)})
            print(f"  sweep lr={lr} tau={tau}: {rows[-1]['final_recall_mean']}", flush=True)
    best = max(rows, key=lambda r: r["final_recall_mean"])
    return {"rows": rows, "best": best, "range": [min(r["final_recall_mean"] for r in rows), max(r["final_recall_mean"] for r in rows)]}


SECTIONS = {"t4": t4, "t5": t5, "t6": t6, "t7": t7, "t9": t9, "t10": t10, "ablations": ablations, "sweep": sweep}


async def main(names):
    try:
        results = json.load(open(OUT))
    except (OSError, ValueError):
        results = {}
    for n in names:
        t0 = time.time()
        print(f"[{n}]", flush=True)
        if n == "vector_hop":
            results[n] = vector_hop()
        elif n == "real_session":
            results[n] = real_session()
        else:
            results[n] = await SECTIONS[n]()
        results[n + "_seconds"] = round(time.time() - t0, 1)
        json.dump(results, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    names = sys.argv[1:] or ["vector_hop", "real_session", "t4", "t5", "t6", "t7", "t9", "t10", "ablations", "sweep"]
    asyncio.run(main(names))
