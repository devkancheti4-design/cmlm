"""
LIVE BY HAND -- Claude in a session as the frontier, no key needed.

ENCODER.md's precedent: packets produced by Claude reading the same
messages in a session, which is the same model the API path would call.
Here the frontier's two tool calls are written by hand, per turn, from the
question and the answer text; then the probes are answered from the
readouts alone, and scored against a truth the frontier never sees.

    python3 sandbox/live_by_hand.py show   SEED TURNS            # phase 1: the turns to write calls for
    python3 sandbox/live_by_hand.py probe  SEED TURNS calls.json # phase 2: run the calls, print readouts to answer
    python3 sandbox/live_by_hand.py score  SEED TURNS calls.json answers.json
"""
import asyncio, json, sys
sys.path.insert(0, ".")
from cmlm.world import World, PROBE
from cmlm.frontier import Session, RULES
from cmlm.model import CMLM
from cmlm.fused import Fused
from types import SimpleNamespace as NS
from cmlm.frontier import _est_request, est_tokens


class HandFrontier:
    """Replays calls a person (or a Claude in a session) wrote: per teaching
    turn a recall (query, k, key) and a teach (pairs). Probes: recall with
    the written query/key, then the readout is shown for a hand answer."""
    def __init__(self, calls, k=4):
        self.calls, self.k = calls, k
        self.beta = NS(messages=NS(create=self.create)); self.n = 0; self.readouts = {}

    async def create(self, **kw):
        msgs = kw["messages"]; q = msgs[0]["content"]; self.n += 1
        c = self.calls.get(q, {})
        sent = _est_request(kw["system"], msgs)
        if len(msgs) == 1:
            rc = c.get("recall", {"query": q, "k": self.k, "key": ""})
            return self._msg([NS(type="tool_use", id=f"h{self.n}", name="recall",
                                 input={"query": rc.get("query", q), "k": int(rc.get("k", self.k)), "key": rc.get("key", "")})], "tool_use", sent)
        self.readouts[q] = msgs[-1]["content"][0]["content"]
        content = [NS(type="text", text=c.get("answer", ""))]
        pairs = c.get("teach", [])
        content.append(NS(type="tool_use", id=f"h{self.n}", name="teach", input={"pairs": pairs}))
        return self._msg(content, "tool_use", sent)

    @staticmethod
    def _msg(content, stop, sent):
        out = sum(est_tokens(getattr(b, "text", None) or json.dumps(getattr(b, "input", None) or {})) for b in content)
        return NS(content=content, stop_reason=stop, stop_details=None, model="claude-by-hand",
                  usage=NS(input_tokens=sent, cache_read_input_tokens=0, cache_creation_input_tokens=0, output_tokens=out, iterations=None))


def main(argv):
    cmd, seed, turns = argv[0], int(argv[1]), int(argv[2])
    w = World(seed=seed, turns=turns)
    if cmd == "show":
        print(json.dumps([{"turn": i + 1, "question": t["question"], "answer": t["answer"]} for i, t in enumerate(w.turns)], indent=1))
        return 0
    calls = json.load(open(argv[3]))
    hf = HandFrontier(calls)
    s = Session(hf, Fused(CMLM(seed=seed)), k=4)
    async def run():
        for t in w.turns:
            await s.ask(t["question"])
        probes = [(topic, w.probe(topic)) for topic in range(len(w.topics))]
        out = []
        for topic, q in probes:
            rc = calls.get(q, {}).get("recall", {})
            rows, text = s.recall(rc.get("query", q), 4, rc.get("key", ""))
            out.append({"topic": topic, "probe": q, "readout": text})
        return out
    out = asyncio.run(run())
    if cmd == "probe":
        st = s.cmlm.state()
        print(json.dumps({"taught": [{"turn": x.t, "pairs": x.taught, "recalls": x.recalls} for x in s.turns],
                          "life_keys": sorted(s.cmlm.life.store), "life_sha": st["life_sha"], "probes": out}, indent=1))
        return 0
    answers = json.load(open(argv[4]))
    hits = total = abst = wrong_abst = 0; rows = []
    for o in out:
        truth = w.truth(o["topic"], turns); a = answers.get(o["probe"], "")
        h = sum(1 for f in truth if f in a); miss = [f for f in truth if f not in o["readout"]]
        said_abstain = "ABSTAIN" in a.upper() or "not held" in a.lower() or "does not hold" in a.lower()
        hits += h; total += len(truth)
        if said_abstain and h == 0 and len(miss) == len(truth): abst += 1
        if said_abstain and h == 0 and len(miss) < len(truth): wrong_abst += 1
        rows.append({"topic": o["topic"], "truth": len(truth), "in_readout": len(truth) - len(miss), "in_answer": h, "abstained": said_abstain})
    print(json.dumps({"seed": seed, "turns": turns, "recall_in_answer": round(hits / total, 3), "recall_in_readout": round(sum(r["in_readout"] for r in rows) / total, 3),
                      "facts": total, "abstained_rightly": abst, "abstained_wrongly": wrong_abst, "rows": rows,
                      "sent_per_turn_mean": round(sum(x.sent_tokens for x in s.turns) / len(s.turns)),
                      "life_keys": len(s.cmlm.life.store), "exact_by_own_keys": sum(1 for k in s.cmlm.life.store if s.cmlm.life.recall(k))}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
