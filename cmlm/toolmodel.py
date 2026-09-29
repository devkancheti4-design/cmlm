"""
A TOOL-RECOVERING MODEL -- an empty CMLM beside the frontier that learns
the frontier's tool calls and then makes them itself.

The cascade removed the transcript. What it could not remove: the tool
definitions, about 380 tokens re-sent on every call, and the recall round
trip before every answer. Both exist only because the frontier decides
what to ask and what to teach. This model recovers those decisions. It has
the CMLM's architecture -- two towers over hashed words -- and its "facts"
are labels: which key a question asks for, which key a stated sentence
belongs under. It starts EMPTY. While the frontier still calls the tools,
every call is a training pair. When the model can recover the calls itself
-- at least `warmup` supervised turns, and self-recall on what it was
taught at or above `theta` on both heads -- the loop switches:

    recovered turn:  question ---> the tool model: the key this asks for (or none)
                     readout fetched BEFORE the frontier is called, with the question as the query
                     frontier answers under short rules, no tools, ONE round trip
                     the tool model recovers the teach: the answer's sentences that carry a
                     value, under the key it predicts for each -> the CMLM and the Life

The frontier's own tool calls remain the teacher whenever the loop is in
tool mode. Keys are recovered by dotted part -- one head per position, so
`topic3.amount` is a topic head and a kind head -- and a part is used only
when its confidence clears `accept`; otherwise the key is empty and the
readout is the map's alone. A key written literally into a question is
taken as written: an explicit key is exact by construction.

What it kills, and what it risks. Killed: the tool definitions, the recall
round trip, the tool-call JSON in the output. Risked: a wrong key writes a
fact under the wrong name in the Life, and a frontier without tools cannot
ask again in other words. Both are measured, not assumed.
"""
import math, re
from .model import CMLM, words
from .frontier import PatternTeacher, sentences

_KEY = re.compile(r"\b([a-z0-9_-]+(?:\.[a-z0-9_-]+)+)\b", re.I)


class KeyHead:
    """text -> a dotted key, one CMLM per dotted position. Labels are the
    CMLM's 'facts'; a prediction is the top label with its softmax share."""

    def __init__(self, seed=0, tau=0.1, min_seen=2, **kw):
        self.seed, self.tau, self.min_seen, self.kw = seed, tau, min_seen, kw
        self.heads = []            # one CMLM per key part
        self.seen = []             # per head: label -> how many examples taught it
        self.n = 0

    def _head(self, i):
        while len(self.heads) <= i:
            self.heads.append(CMLM(seed=self.seed + len(self.heads), **self.kw)); self.seen.append({})
        return self.heads[i]

    def learn(self, text, key, turn=0):
        parts = str(key).strip().split(".")
        for i, part in enumerate(parts):
            self._head(i).teach([(text, part)], turn)
            self.seen[i][part] = self.seen[i].get(part, 0) + 1
        self.n += 1

    def predict(self, text):
        """(key or "", confidence): each part's top label, if its share of
        the softmax over all labels of that head clears nothing here --
        the caller applies `accept` -- and the whole key's confidence is
        the weakest part's."""
        if not self.heads or not self.heads[0].facts:
            return "", 0.0
        parts, conf = [], 1.0
        for h, seen in zip(self.heads, self.seen):
            rows = h.readout(text, k=max(1, len(h.facts)))
            if not rows:
                return "", 0.0
            m = max(r["score"] for r in rows)
            z = [math.exp((r["score"] - m) / self.tau) for r in rows]
            p = z[0] / sum(z)
            if seen.get(rows[0]["fact"], 0) < self.min_seen:      # a label taught once is a guess, not a prediction
                p = 0.0
            parts.append(rows[0]["fact"]); conf = min(conf, p)
        return ".".join(parts), conf

    def self_recall(self):
        if not self.heads or not self.heads[0].cues:
            return 0.0
        return min(h.self_recall() for h in self.heads)

    def labels(self):
        return [list(h.facts) for h in self.heads]


class ToolModel:
    def __init__(self, seed=0, warmup=30, theta=0.85, accept=0.6, **kw):
        self.ask = KeyHead(seed, **kw)          # question -> key
        self.state = KeyHead(seed + 100, **kw)  # a stated sentence -> key
        self.warmup, self.theta, self.accept = warmup, theta, accept
        self.teacher = PatternTeacher()
        self.supervised = 0; self.recovered = 0; self.switched_at = None; self.restated = 0
        # PREQUENTIAL accuracy: every supervised example is predicted BEFORE it is learned, and scored
        # against the frontier's key. Self-recall on taught cues is blind to generalisation (it sits at
        # 1.0 while unseen-fact accuracy is 0.57); predict-then-learn is the honest estimate, and it is
        # what readiness is judged on.
        self.preq = {"ask": [0, 0], "state": [0, 0]}   # [scored, right]
        self.asked = 0; self.asked_keyed = 0; self.stated = 0; self.stated_keyed = 0

    # ------------------------------------------------------------ learning from the frontier
    def learn_from(self, question, recalls, taught, turn):
        """One tool-mode turn: every keyed recall and every keyed teach pair
        is a training pair. Unkeyed calls teach nothing."""
        for r in recalls:
            if r.get("key"):
                self._score("ask", question, r["key"])
                self.ask.learn(question, r["key"], turn)
        for p in taught:
            if len(p) > 2 and p[2]:
                self._score("state", p[1], p[2])
                self.state.learn(p[1], p[2], turn)
        self.supervised += 1

    def _score(self, head, text, key):
        h = self.ask if head == "ask" else self.state
        if h.n < 5:                                   # nothing to predict with yet; not a miss
            return
        pk, conf = h.predict(text)
        c = self.preq[head]; c[0] += 1; c[1] += int(pk == str(key).strip())

    def prequential(self, head):
        n, ok = self.preq[head]
        return (ok / n) if n else 0.0

    def ready(self):
        """Enough KEYED examples on both heads -- probes and unkeyed calls
        teach nothing and do not count -- and both heads still return what
        they were taught."""
        return (min(self.ask.n, self.state.n) >= self.warmup
                and min(self.preq["ask"][0], self.preq["state"][0]) >= 10
                and self.prequential("ask") >= self.theta and self.prequential("state") >= self.theta)

    # ------------------------------------------------------------ recovering the calls
    def recover_recall(self, question):
        """The key this question asks for: written literally, or predicted
        with confidence over `accept`; else none."""
        self.asked += 1
        m = _KEY.search(question)
        if m:
            self.asked_keyed += 1
            return m.group(1), 1.0, "literal"
        key, conf = self.ask.predict(question)
        if key and conf >= self.accept:
            self.asked_keyed += 1
            return key, conf, "predicted"
        return "", conf, "none"

    def recover_teach(self, question, answer, turn, context="", known=()):
        """The teach the frontier would have made: the answer's sentences
        that carry a value, each under the key the state head predicts for
        it when confident, else unkeyed. Only what was stated ANEW: a
        sentence that restates the context the frontier was sent, or a
        fact the store already holds, is not a new fact -- a frontier that
        quotes its readout would otherwise teach it back, prefixed, as
        hundreds of near-duplicates (measured: recall fell to 0.08 before
        this line existed). Returns triples and the record."""
        pairs, rec = [], []
        known = [k for k in known if k]
        for cue, fact in self.teacher.teach(question, answer):
            if (context and fact in context) or any(k in fact for k in known):
                self.restated += 1
                continue
            key, conf = self.state.predict(fact)
            keyed = bool(key) and conf >= self.accept
            pairs.append((cue, fact, key if keyed else ""))
            rec.append({"fact": fact, "key": key if keyed else "", "conf": round(conf, 3)})
            self.stated += 1; self.stated_keyed += int(keyed)
        self.recovered += 1
        return pairs, rec

    def state_(self):
        return {"supervised": self.supervised, "recovered": self.recovered, "switched_at": self.switched_at,
                "ask_self_recall": round(self.ask.self_recall(), 3), "state_self_recall": round(self.state.self_recall(), 3),
                "ask_prequential": round(self.prequential("ask"), 3), "state_prequential": round(self.prequential("state"), 3),
                "prequential_n": {"ask": self.preq["ask"][0], "state": self.preq["state"][0]},
                "asked": self.asked, "asked_keyed": self.asked_keyed, "stated": self.stated, "stated_keyed": self.stated_keyed,
                "restated_skipped": self.restated,
                "labels": {"ask": [len(l) for l in self.ask.labels()], "state": [len(l) for l in self.state.labels()]}}
