"""
A CMLM FOR CMLMs -- recursion against forgetting.

A CMLM forgets as it fills. At 60 turns the flat model returned 0.525 of
the facts taught over fifteen turns earlier against 0.817 of recent ones
(CMLM.md): training on new pairs moves the weights that held the old ones.
The recursive form stops that at the source. A child CMLM that has begun
to forget is CLOSED -- its weights never move again -- and a new child
opens; a parent CMLM, itself a CMLM whose "facts" are the children, learns
which child a question should be asked of. Recall asks the parent to rank
the children, asks the first `fan` of them (all, by default) for k facts
each, and interleaves by rank. The frontier sees the same two tools; the
recursion is inside.

Forgetting is SELF-MEASURED, at zero tokens and with no truth: after each
teach the open child re-asks itself every cue it was taught and counts how
often the fact it was taught comes back first (self-recall@1). Under
`theta` it closes. It also closes at `cap` facts, a ceiling read off the
flat model's record, because a model that fits its own cues may not notice
that it has stopped answering paraphrases of them -- the record shows which
of the two fired, and how often.

Not built: saving a tree to a file. It is an experiment against the flat
model, measured in the same world with the same probes.
"""
from .model import CMLM, words


class Tree:
    def __init__(self, rules="", seed=0, cap=30, theta=0.9, fan=None, **kw):
        self.seed, self.kw = seed, kw
        self.cap, self.theta, self.fan = cap, theta, fan
        self.parent = CMLM(rules=rules, seed=seed, **kw)          # its facts are child labels
        self.children = [CMLM(rules=rules, seed=seed, **kw)]      # one seed: every child shares W0, so matching scores compare
        self.closed = []            # the record: turn, child, facts, self_recall, why
        self.turns = 0
        self.wall_ms = 0.0

    # Session reads and may set .rules; the parent holds them for the tree.
    @property
    def rules(self):
        return self.parent.rules

    @rules.setter
    def rules(self, v):
        self.parent.rules = v
        for c in self.children:
            c.rules = v

    @property
    def rules_sha(self):
        return self.parent.rules_sha

    @property
    def open(self):
        return self.children[-1]

    @property
    def facts(self):
        return [f for c in self.children for f in c.facts]

    def where(self, fact):
        """Which child holds a fact; None if none does."""
        fact = " ".join(str(fact).split())
        return next((i for i, c in enumerate(self.children) if fact in c.facts), None)

    # ------------------------------------------------------------ training
    def teach(self, pairs, turn):
        i = len(self.children) - 1
        c = self.open
        rec = c.teach(pairs, turn)
        prec = self.parent.teach([(cue, f"child {i}") for cue, fact in pairs if words(cue) and words(fact)], turn)
        sr = rec["self_recall"]
        why = None
        if rec["pairs"]:
            if sr < self.theta:
                why = "forgetting"
            elif len(c.facts) >= self.cap:
                why = "cap"
        if why:
            self.closed.append({"turn": int(turn), "child": i, "facts": len(c.facts), "self_recall": sr, "why": why})
            self.children.append(CMLM(rules=self.rules, seed=self.seed, **self.kw))
        ms = rec["ms"] + prec["ms"]
        self.wall_ms += prec["ms"]
        return {**rec, "ms": round(ms, 2), "children": len(self.children), "closed": why}

    # ------------------------------------------------------------ readout
    def route(self, query, trained=True):
        """Every child, in the parent's order; a child the parent was never
        taught about (fresh and empty) comes last."""
        n = len(self.children)
        if n == 1:
            return [0]
        order = []
        for r in self.parent.readout(query, n, trained):
            j = int(r["fact"].split()[1])
            if j not in order:
                order.append(j)
        return order + [j for j in range(n) if j not in order]

    def readout(self, query, k=4, trained=True):
        if not trained:
            # the matching baseline is flat: every fact under the shared step-0 matrix
            rows = [r for c in self.children for r in c.readout(query, k, trained=False)]
            rows.sort(key=lambda r: -r["score"])
            return rows[:k]
        order = self.route(query)[: self.fan or len(self.children)]
        lists = [self.children[j].readout(query, k) for j in order]
        out, seen, rank = [], set(), 0
        while len(out) < k and any(rank < len(l) for l in lists):
            for l in lists:
                if rank < len(l) and len(out) < k and l[rank]["fact"] not in seen:
                    out.append(l[rank]); seen.add(l[rank]["fact"])     # a fact taught twice may sit in two children
            rank += 1
        return out

    def state(self):
        cs = self.children
        return {"rules_sha": self.rules_sha, "facts": sum(len(c.facts) for c in cs), "pairs": sum(len(c.cues) for c in cs),
                "params": self.parent.params() + sum(c.params() for c in cs),
                "steps": self.parent.steps + sum(c.steps for c in cs), "turns": self.turns,
                "wall_ms": round(self.wall_ms + sum(c.wall_ms for c in cs), 1),
                "loss_last": self.open.loss[-1] if self.open.loss else None,
                "children": len(cs), "closed": list(self.closed)}
