"""
THE FAKE WORLD -- a scripted session with a known truth, and a scripted
frontier model that pulls, answers and teaches the way the loop expects.

Ten topics. FACT words and CUE words are disjoint within a topic and across
topics, so in the paraphrase world a probe shares no word with the fact it
needs; a test asserts it. With overlap=True the cue words ARE the fact words:
matching should suffice there, and the record says whether training bought
anything. The frontier model's answers are SCRIPTED -- the world is the
answerer -- because what is measured is the CMLM and the loop around it,
not the frontier model. Token counts are chars/4, labelled FAKE; the model
name is "fake-frontier", so nothing here is ever priced.
"""
import json, random
from types import SimpleNamespace as NS
from .model import CMLM
from .frontier import Session, PatternTeacher, est_tokens, _est_request, K_DEFAULT

TOPICS = [
    (["lamp", "carton", "courier", "dispatched", "acme", "warehouse"],
     ["light", "fixture", "purchase", "parcel", "supplier", "shipping"]),
    (["lease", "flat", "landlord", "deposit", "tenancy", "rent"],
     ["apartment", "agreement", "owner", "bond", "letting", "monthly"]),
    (["visa", "embassy", "passport", "biometrics", "consulate", "stamp"],
     ["travel", "permit", "interview", "document", "fingerprints", "mission"]),
    (["invoice", "accounts", "payable", "remittance", "ledger", "vendor"],
     ["bill", "finance", "settlement", "transfer", "books", "contractor"]),
    (["server", "outage", "incident", "rollback", "deploy", "cluster"],
     ["machine", "downtime", "failure", "revert", "release", "fleet"]),
    (["dentist", "molar", "filling", "crown", "clinic", "hygienist"],
     ["tooth", "cavity", "repair", "cap", "surgery", "cleaning"]),
    (["thesis", "advisor", "chapter", "defense", "committee", "draft"],
     ["dissertation", "supervisor", "section", "viva", "panel", "manuscript"]),
    (["garden", "seedlings", "compost", "trellis", "tomatoes", "planter"],
     ["yard", "sprouts", "mulch", "frame", "vegetables", "pot"]),
    (["insurance", "premium", "claim", "adjuster", "policy", "deductible"],
     ["cover", "fee", "request", "assessor", "terms", "excess"]),
    (["conference", "keynote", "abstract", "registration", "venue", "badge"],
     ["summit", "talk", "summary", "signup", "hall", "pass"]),
]
FILLER = ("then also while after before since just still quite rather really mostly fairly "
          "nearly around again maybe perhaps probably anyway meanwhile somewhat usually often "
          "certainly generally roughly likewise otherwise").split()
VALUES = [
    lambda r: f"{r.randint(1, 9)},{r.randint(100, 999)} rupees",
    lambda r: f"{r.choice(['Monday', 'Tuesday', 'Thursday', 'Friday'])} at {r.randint(8, 18)}:{r.choice(['00', '15', '30', '45'])}",
    lambda r: f"reference {r.randint(10000, 99999)}",
    lambda r: f"{r.randint(2, 9)} of {r.randint(10, 40)}",
    lambda r: f"due on the {r.randint(1, 28)}th",
]
KINDS = ["amount", "schedule", "reference", "count", "deadline"]     # one per VALUES entry: the fact's exact key is topic.kind
PROBE = "Tell me again about the"
EXACT = "What exactly is"           # an exact probe: the frontier asks recall with the key


class World:
    def __init__(self, seed=1, turns=30, overlap=False, bulk=0):
        """bulk: extra filler sentences appended to every answer, so an
        answer is a document, not a line -- the deep-search shape, where
        one turn's result is thousands of tokens and holds a fact or two.
        Drawn after the turn's own words, so bulk=0 worlds are unchanged."""
        self.rng = random.Random(seed)
        self.overlap, self.bulk = overlap, bulk
        self.topics = [(f, f if overlap else c) for f, c in TOPICS]
        order = [i % len(TOPICS) for i in range(turns)]
        self.rng.shuffle(order)
        self.turns = []
        for i in order:
            fw, cw = self.topics[i]
            fw3 = self.rng.sample(fw, 3)
            kind = self.rng.randrange(len(VALUES))               # the same draw rng.choice made, in the same order: worlds are unchanged
            fact = f"The {' '.join(fw3)} is {VALUES[kind](self.rng)}."
            q = f"What about the {' '.join(self.rng.sample(cw, 3))}?"
            fill = [(" ".join(self.rng.sample(FILLER, 7))).capitalize() + "." for _ in range(2)]
            self.turns.append({"topic": i, "question": q, "fact": fact, "key": f"topic{i}.{KINDS[kind]}",
                               "answer": " ".join([fill[0], fact, fill[1]])})
        if bulk:
            for t in self.turns:
                more = [(" ".join(self.rng.sample(FILLER, 7))).capitalize() + "." for _ in range(bulk)]
                t["answer"] = t["answer"] + " " + " ".join(more)

    @property
    def answers(self):
        return [t["answer"] for t in self.turns]

    def probe(self, topic):
        return f"{PROBE} {' '.join(self.rng.sample(self.topics[topic][1], 3))}."

    def truth(self, topic, upto):
        return [t["fact"] for t in self.turns[:upto] if t["topic"] == topic]

    @property
    def keys(self):
        return [t["key"] for t in self.turns]

    def newest(self, upto):
        """key -> the fact stated last under it: the truth an exact probe wants.
        A key stated twice is a natural revision: the same topic and kind, a new value."""
        out = {}
        for t in self.turns[:upto]:
            out[t["key"]] = t["fact"]
        return out

    def revised(self, upto):
        seen, rev = {}, set()
        for t in self.turns[:upto]:
            if t["key"] in seen and seen[t["key"]] != t["fact"]:
                rev.add(t["key"])
            seen[t["key"]] = t["fact"]
        return rev

    @staticmethod
    def exact_probe(key):
        return f"{EXACT} {key}?"

    def transcript_tokens(self, upto, rules):
        return est_tokens(rules) + sum(est_tokens(t["question"]) + est_tokens(t["answer"]) for t in self.turns[:upto])

    def window(self, upto, tokens):
        text = "\n".join(f"Q: {t['question']}\nA: {t['answer']}" for t in self.turns[:upto])
        return text[-4 * tokens:]


class FakeFrontier:
    """A scripted frontier model behind client.beta.messages.create.
    First call of a turn (one user message): recall(question, k). Second
    call, after the readout: the answer -- the world's scripted answer on a
    teaching turn, the readout quoted back on a probe -- plus one teach
    call with the pairs the pattern teacher would write, and none on a
    probe. It never sees the world's truth."""

    def __init__(self, answers, k=K_DEFAULT, keys=None):
        self.answers, self.k = list(answers), k
        self.keys = list(keys) if keys else None           # with keys, it teaches each fact under its key
        self.calls, self.teaching_turn, self._n = [], 0, 0
        self.teacher = PatternTeacher()
        self.beta = NS(messages=NS(create=self.create))

    async def create(self, **kw):
        self.calls.append(kw)
        msgs = kw["messages"]
        q = msgs[0]["content"]
        sent = _est_request(kw["system"], msgs)
        self._n += 1
        if not kw.get("tools"):
            # a recovered turn: the readout is in a <context> block above the question; no tools, one answer
            ctx, _, question = q.rpartition("</context>\n\n")
            question = question or q
            if question.startswith(PROBE) or question.startswith(EXACT):
                text = "From the context model:\n" + ctx.replace("<context>\n", "", 1)
            else:
                text = self.answers[self.teaching_turn % len(self.answers)]
                self.teaching_turn += 1
            return self._msg([NS(type="text", text=text)], "end_turn", sent)
        if len(msgs) == 1:
            key = q[len(EXACT):].strip(" ?") if q.startswith(EXACT) else ""
            return self._msg([NS(type="tool_use", id=f"tu{self._n}", name="recall", input={"query": q, "k": self.k, "key": key})], "tool_use", sent)
        readout = msgs[-1]["content"][0]["content"]
        if q.startswith(PROBE) or q.startswith(EXACT):
            return self._msg([NS(type="text", text="From the context model:\n" + readout),
                              NS(type="tool_use", id=f"tu{self._n}", name="teach", input={"pairs": []})], "tool_use", sent)
        i = self.teaching_turn
        text = self.answers[i % len(self.answers)]
        self.teaching_turn += 1
        key = self.keys[i % len(self.keys)] if self.keys else ""
        pairs = [{"cue": c, "fact": f, "key": key} for c, f in self.teacher.teach(q, text)]
        return self._msg([NS(type="text", text=text),
                          NS(type="tool_use", id=f"tu{self._n}", name="teach", input={"pairs": pairs})], "tool_use", sent)

    @staticmethod
    def _msg(content, stop, sent):
        out = sum(est_tokens(getattr(b, "text", None) or json.dumps(getattr(b, "input", None) or {})) for b in content)
        return NS(content=content, stop_reason=stop, stop_details=None, model="fake-frontier",
                  usage=NS(input_tokens=sent, cache_read_input_tokens=0, cache_creation_input_tokens=0,
                           output_tokens=out, iterations=None))


# ------------------------------------------------------------ measurement
async def measure(seed=1, turns=30, k=K_DEFAULT, every=5, window=320, overlap=False, cmlm=None,
                  tree=False, recursive=False, cap=30, theta=0.9, fan=None, merge="rank", demote_failing=False,
                  fused=False, toolmodel=False, warmup=30, bulk=0, gate=None):
    """One fake session through the real loop, probed every `every` turns on
    every topic taught so far. Recall@k for the trained CMLM (through the
    recall tool, scored on the frontier's answer), the same model at step 0
    (matching), and a transcript window of `window` tokens; the full
    transcript recalls everything at P tokens, by construction."""
    w = World(seed=seed, turns=turns, overlap=overlap, bulk=bulk)
    if cmlm is None:
        if tree:
            from .tree import Tree
            cmlm = Tree(seed=seed, cap=cap, theta=theta, fan=fan)
        elif recursive:
            from .cascade import Recursive
            cmlm = Recursive(seed=seed, cap=cap, merge=merge, demote_failing=demote_failing)
        else:
            cmlm = CMLM(seed=seed)
        if fused:
            from .fused import Fused
            cmlm = Fused(cmlm)
    tm = None
    if toolmodel:
        from .toolmodel import ToolModel
        tm = ToolModel(seed=seed, warmup=warmup) if gate is None else ToolModel(seed=seed, warmup=warmup, theta=gate)
    s = Session(FakeFrontier(w.answers, k=k, keys=w.keys if fused else None), cmlm, k=k, toolmodel=tm)
    probes, exacts, log = [], [], []
    for t, turn in enumerate(w.turns):
        x = await s.ask(turn["question"])
        used = x.taught[0][2] if x.taught and len(x.taught[0]) > 2 else ""
        log.append({**x.to_dict(), "teaching_turn": t + 1, "world_transcript_tokens": w.transcript_tokens(t + 1, s.cmlm.rules),
                    "true_key": turn["key"], "key_used": used, "key_ok": int(used == turn["key"])})
        if (t + 1) % every == 0 or t + 1 == turns:
            first = {}
            for j, y in enumerate(w.turns[:t + 1]):
                first.setdefault(y["topic"], j)
            win = w.window(t + 1, window)
            for topic in sorted(first):
                truth = set(w.truth(topic, t + 1))
                q = w.probe(topic)
                got0 = {r["fact"] for r in s.cmlm.readout(q, k, trained=False)}
                route2 = None
                if hasattr(s.cmlm, "route"):             # a tree: is the child that holds each fact among the parent's first two?
                    order = s.cmlm.route(q)[:2]
                    route2 = sum(1 for f in truth if s.cmlm.where(f) in order)
                p = await s.ask(q)                       # through the tool, scored on the answer
                probes.append({"turn": t + 1, "topic": topic, "age": t + 1 - first[topic], "truth": len(truth), "k": k,
                               "trained": sum(1 for f in truth if f in p.answer),
                               "matching": len(got0 & truth),
                               "window": sum(1 for f in truth if f in win),
                               "route2": route2,
                               "sent_tokens": p.sent_tokens, "round_trips": p.round_trips})
            if fused:
                # exact probes: every key taught so far, the truth is the fact stated last under it;
                # and three keys never taught, where the only right answer is ABSTAIN
                newest, rev = w.newest(t + 1), w.revised(t + 1)
                for key, fact in newest.items():
                    p = await s.ask(w.exact_probe(key))
                    first = p.answer.split("\n")[1] if "\n" in p.answer else ""
                    exacts.append({"turn": t + 1, "key": key, "kind": "known", "revised": key in rev,
                                   "hit": int(fact in first), "abstained": int("ABSTAIN" in first),
                                   "sent_tokens": p.sent_tokens})
                for key in (f"nonesuch.{KINDS[j]}" for j in range(3)):
                    p = await s.ask(w.exact_probe(key))
                    first = p.answer.split("\n")[1] if "\n" in p.answer else ""
                    exacts.append({"turn": t + 1, "key": key, "kind": "unknown", "revised": False,
                                   "hit": 0, "abstained": int("ABSTAIN" in first), "sent_tokens": p.sent_tokens})
    last = log[-1]
    return {"seed": seed, "turns": turns, "k": k, "overlap": overlap, "window": window,
            "probes": probes, "recall": _recall(probes), "exacts": exacts, "exact": _exact(exacts),
            "final_recall": _recall(probes, lambda p: p["turn"] == turns),
            "final": {"transcript_tokens": last["world_transcript_tokens"], "sent_tokens": last["sent_tokens"],
                      "sent_max": max(x["sent_tokens"] for x in log),
                      "round_trips": sum(x["round_trips"] for x in log) / len(log)},
            "model": s.cmlm.state(), "toolmodel": (tm.state_() if tm else None), "turn_log": log}


def _exact(exacts):
    """Exact probes by key: hits on known keys, ABSTAIN on unknown ones, and
    the newest value on keys stated more than once."""
    if not exacts:
        return None
    known = [e for e in exacts if e["kind"] == "known"]
    unknown = [e for e in exacts if e["kind"] == "unknown"]
    rev = [e for e in known if e["revised"]]
    r = lambda xs, f: (round(sum(f(x) for x in xs) / len(xs), 3) if xs else None)
    return {"known": len(known), "hit": r(known, lambda e: e["hit"]),
            "unknown": len(unknown), "abstain": r(unknown, lambda e: e["abstained"]),
            "false_abstain": r(known, lambda e: e["abstained"]),
            "revised": len(rev), "newest": r(rev, lambda e: e["hit"])}


def _recall(probes, key=None):
    ps = [p for p in probes if key is None or key(p)]
    n = sum(p["truth"] for p in ps)
    out = {m: (round(sum(p[m] for p in ps) / n, 3) if n else None) for m in ("trained", "matching", "window")}
    # recall@k cannot exceed min(k, truth)/truth: with more facts per topic than k, the ceiling is under 1
    out.update(probes=len(ps), facts=n, ceiling=(round(sum(min(p["k"], p["truth"]) for p in ps) / n, 3) if n else None))
    if ps and ps[0].get("route2") is not None:
        out["route2"] = round(sum(p["route2"] for p in ps) / n, 3) if n else None
    return out


def by_age(probes, split=15):
    return {"old": _recall(probes, lambda p: p["age"] > split), "recent": _recall(probes, lambda p: p["age"] <= split)}


async def measure_seeds(seeds, **kw):
    out = [await measure(seed=sd, **kw) for sd in seeds]
    allp = [p for r in out for p in r["probes"]]
    tree = None
    if "children" in out[0]["model"]:
        closes = [c for r in out for c in r["model"]["closed"]]
        tree = {"children_mean": round(sum(r["model"]["children"] for r in out) / len(out), 2),
                "closed": len(closes), "why": {w: sum(1 for c in closes if c["why"] == w) for w in ("forgetting", "cap")},
                "self_recall_at_close": round(sum(c["self_recall"] for c in closes) / len(closes), 3) if closes else None}
    cascade = None
    if "levels" in out[0]["model"]:
        ms = [r["model"] for r in out]
        cascade = {"levels_mean": round(sum(m["levels"] for m in ms) / len(ms), 2),
                   "demoted": sum(m["demoted"] for m in ms),
                   "why": {w: sum(m["why"][w] for m in ms) for w in ("forgetting", "cap")},
                   "per_level": ms[0]["per_level"]}
    alle = [e for r in out for e in r["exacts"]]
    tmodel = None
    if out[0].get("toolmodel"):
        logs = [x for r in out for x in r["turn_log"]]
        tools = [x for x in logs if x["mode"] == "tools"]; rec = [x for x in logs if x["mode"] == "recovered"]
        mean = lambda xs, f: (round(sum(f(x) for x in xs) / len(xs), 3) if xs else None)
        tmodel = {"switched_at": [r["toolmodel"]["switched_at"] for r in out],
                  "state_prequential": round(sum(r["toolmodel"]["state_prequential"] for r in out) / len(out), 3),
                  "ask_prequential": round(sum(r["toolmodel"]["ask_prequential"] for r in out) / len(out), 3),
                  "turns_tools": len(tools), "turns_recovered": len(rec),
                  "sent_tools": mean(tools, lambda x: x["sent_tokens"]), "sent_recovered": mean(rec, lambda x: x["sent_tokens"]),
                  "trips_tools": mean(tools, lambda x: x["round_trips"]), "trips_recovered": mean(rec, lambda x: x["round_trips"]),
                  "key_ok_recovered": mean(rec, lambda x: x["key_ok"]), "keyed_recovered": mean(rec, lambda x: int(bool(x["key_used"]))),
                  "recall_keyed": mean([x for x in rec if x["recalls"]], lambda x: int(bool(x["recalls"][0]["key"])))}
    return {"seeds": len(out), "runs": out, "recall": _recall(allp), "by_age": by_age(allp), "tree": tree, "cascade": cascade,
            "exact": _exact(alle), "life": (out[0]["model"].get("life_keys") is not None), "toolmodel": tmodel,
            "self_recall_last": round(sum(r["turn_log"][-1]["train"]["self_recall"] for r in out) / len(out), 3),
            "final_recall": _recall(allp, lambda p: p["turn"] == out[0]["turns"]),
            "worst_final": min((r["final_recall"]["trained"] or 0) for r in out),
            "trained_beats_matching": sum(1 for r in out if (r["final_recall"]["trained"] or 0) > (r["final_recall"]["matching"] or 0)),
            "wall_ms_mean": round(sum(r["model"]["wall_ms"] for r in out) / len(out), 1),
            "final": {kk: round(sum(r["final"][kk] for r in out) / len(out), 1) for kk in ("transcript_tokens", "sent_tokens", "sent_max", "round_trips")}}


# ------------------------------------------------------------ report
def report_session(res):
    k = res["k"]
    print(f"\n  CMLM  seed {res['seed']}, {res['turns']} turns, recall k={k}"
          f"   [FAKE world: scripted frontier, chars/4 token estimates, REAL training, REAL recall]")
    print(f"\n  {'turn':>4s}  {'transcript':>10s}  {'sent':>6s}  {'trips':>5s}  {'pairs':>5s}  {'train ms':>8s}  {'by':8s} question")
    for x in res["turn_log"]:
        print(f"  {x['teaching_turn']:>4d}  {x['world_transcript_tokens']:>10,}  {x['sent_tokens']:>6,}  {x['round_trips']:>5d}  "
              f"{x['pairs']:>5d}  {x['train']['ms']:>8.1f}  {x['taught_by'] or '-':8s} {x['question']}")
    f = res["final"]
    print(f"\n  the last turn: the transcript path would have sent {f['transcript_tokens']:,} tokens in one call;"
          f" this path sent {f['sent_tokens']:,} over {f['round_trips']:.0f} round trips (max over the session {f['sent_max']:,}).")
    r, fr = res["recall"], res["final_recall"]
    print(f"\n  PROBES  recall@{k} through the recall tool, scored on the frontier's answer.  {r['probes']} probes, {r['facts']} facts needed:")
    print(f"    {'':28s} {'all probes':>10s} {'last turn':>10s}")
    print(f"    {'trained CMLM':28s} {r['trained']:>10.3f} {fr['trained']:>10.3f}")
    print(f"    {'the same model at step 0':28s} {r['matching']:>10.3f} {fr['matching']:>10.3f}   (matching: a random projection of the words)")
    print(f"    {'last %d tokens of transcript' % res['window']:28s} {r['window']:>10.3f} {fr['window']:>10.3f}")
    print(f"    {'the full transcript':28s} {'1.000':>10s} {'1.000':>10s}   at {f['transcript_tokens']:,} tokens, growing every turn")
    print(f"    recall@{k} cannot exceed {r['ceiling']:.3f} over all probes and {fr['ceiling']:.3f} at the last turn: k against the facts a topic holds")
    if res.get("exact"):
        e = res["exact"]
        print(f"    LIFE  exact by key: {e['hit']:.3f} of {e['known']} known keys (false ABSTAIN {e['false_abstain']:.3f});"
              f" ABSTAIN on {e['abstain']:.3f} of {e['unknown']} unknown keys; newest value on {e['revised']} revised keys: "
              f"{e['newest'] if e['newest'] is not None else 'n/a'}")
    a = by_age(res["probes"])
    print(f"\n  FORGETTING  facts taught over 15 turns ago: trained {a['old']['trained']}   recent: {a['recent']['trained']}")
    m = res["model"]
    print(f"  MODEL  {m['params']:,} weights, {m['facts']} facts stored, {m['pairs']} pairs, {m['steps']} steps, "
          f"{m['wall_ms']:.0f} ms of training in total, last loss {m['loss_last']:.3f}, bound to rules {m['rules_sha']}")


def report_seeds(res, k, label):
    r, fr, a, f = res["recall"], res["final_recall"], res["by_age"], res["final"]
    print(f"\n  {label}: {res['seeds']} seeds, recall@{k} over {r['probes']} probes")
    print(f"    {'':10s} {'all probes':>10s} {'last turn':>10s}")
    print(f"    {'trained':10s} {r['trained']:>10.3f} {fr['trained']:>10.3f}   worst seed at the last turn {res['worst_final']:.3f};"
          f" beats matching in {res['trained_beats_matching']} of {res['seeds']} seeds")
    print(f"    {'matching':10s} {r['matching']:>10.3f} {fr['matching']:>10.3f}")
    print(f"    {'window':10s} {r['window']:>10.3f} {fr['window']:>10.3f}")
    print(f"    {'ceiling':10s} {r['ceiling']:>10.3f} {fr['ceiling']:>10.3f}   (k against the facts a topic holds; recall@{k} cannot exceed it)")
    print(f"    old facts (taught >15 turns ago): trained {a['old']['trained']}  matching {a['old']['matching']}  window {a['old']['window']}"
          f"   recent: trained {a['recent']['trained']}")
    print(f"    the model's own self-recall@1 on its cues at the last turn: {res['self_recall_last']:.3f}"
          + (f"; the truth's child among the parent's first two: {fr.get('route2')}" if fr.get("route2") is not None else ""))
    if res.get("exact"):
        e = res["exact"]
        print(f"    LIFE  exact by key: {e['hit']:.3f} of {e['known']} known keys (false ABSTAIN {e['false_abstain']:.3f});"
              f" ABSTAIN on {e['abstain']:.3f} of {e['unknown']} unknown keys; newest value on {e['revised']} revised keys: "
              f"{e['newest'] if e['newest'] is not None else 'n/a'}")
    if res.get("toolmodel"):
        t = res["toolmodel"]
        sw = [x for x in t["switched_at"] if x is not None]
        print(f"    TOOL MODEL  switched on {len(sw)} of {res['seeds']} seeds" + (f" at session turn {min(sw)}..{max(sw)}" if sw else "")
              + f"; {t['turns_recovered']} recovered teaching turns of {t['turns_tools'] + t['turns_recovered']};"
              f" prequential accuracy at the end: keys from facts {t['state_prequential']}, keys from questions {t['ask_prequential']}")
        print(f"      sent per turn: tools {t['sent_tools']} over {t['trips_tools']} round trips -> recovered {t['sent_recovered']} over {t['trips_recovered']};"
              f" recovered teach keys right {t['key_ok_recovered']} (keyed {t['keyed_recovered']}); recall keyed {t['recall_keyed']}")
    if res.get("cascade"):
        c = res["cascade"]
        print(f"    CASCADE  {c['levels_mean']} levels on average; {c['demoted']} facts handed down: {c['why']['forgetting']} for forgetting,"
              f" {c['why']['cap']} at the cap; facts per level in seed 1: {c['per_level']}")
    if res.get("tree"):
        t = res["tree"]
        print(f"    TREE  {t['children_mean']} children on average; {t['closed']} closes: {t['why']['forgetting']} for forgetting,"
              f" {t['why']['cap']} at the cap; self-recall at close {t['self_recall_at_close']}")
    print(f"    last turn, mean over seeds: transcript {f['transcript_tokens']:,.0f} tokens in one call; sent {f['sent_tokens']:,.0f}"
          f" over {f['round_trips']:.1f} round trips (max {f['sent_max']:,.0f}); training {res['wall_ms_mean']:.0f} ms per session")
