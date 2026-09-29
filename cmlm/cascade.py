"""
A RECURSIVE CMLM -- the CMLM is the frontier of its own CMLM.

Claude keeps no transcript: it recalls from a CMLM and teaches it. The
recursive form makes that relation the same at every level. A CMLM keeps
at most `cap` facts of its own; for the rest it is a frontier to a CMLM of
its own, which it recalls from and teaches. After every teach it hands
down what it cannot hold -- the oldest facts, until at most `cap` remain
here; with demote_failing=True, first any fact that no longer comes back
for the cue it was taught with, its own forgetting, self-measured, which
measured worse (CMLM.md) and is off by default. Its context does the same,
and so on down; the depth grows with the session, and every level holds a
bounded store, so every level replays all of its own pairs on every turn
and none of them forgets within itself.

Recall asks this level first, then its context, and so on down, and the
readout interleaves the levels by rank, nearest level first, each fact
marked with the depth it came from (merge="z" instead compares scores by
their distance from each level's own mean, in that level's spread; it
measured worse where levels are equal and facts are spread evenly). Between
two CMLMs the hop is not words: the cue travels as the feature vector this
level already computed, and only the fact travels as the string the store
must keep for the frontier. The hop into Claude is words, because the API
takes tokens.

Not built: saving a cascade to a file (vector-taught cues have no string
to save). It is an experiment against the flat model and the tree,
measured in the same world with the same probes.
"""
from .model import CMLM, features


class Recursive:
    def __init__(self, rules="", seed=0, cap=30, depth=0, merge="rank", demote_failing=False, **kw):
        self.seed, self.kw, self.cap, self.depth = seed, kw, cap, depth
        self.merge, self.demote_failing = merge, demote_failing
        self.model = CMLM(rules=rules, seed=seed, **kw)    # one seed at every level: a shared W0, so matching compares
        self.context = None                                # its own CMLM, made when first needed
        self.turns = 0
        self.demoted = []                                  # the record: turn, fact, why, to depth

    # Session reads and may set .rules; they propagate down.
    @property
    def rules(self):
        return self.model.rules

    @rules.setter
    def rules(self, v):
        self.model.rules = v
        if self.context is not None:
            self.context.rules = v

    @property
    def rules_sha(self):
        return self.model.rules_sha

    @property
    def facts(self):
        return self.model.facts + (self.context.facts if self.context else [])

    def levels(self):
        return 1 + (self.context.levels() if self.context else 0)

    def where(self, fact):
        """The depth a fact sits at; None if nowhere."""
        fact = " ".join(str(fact).split())
        if fact in self.model.facts:
            return self.depth
        return self.context.where(fact) if self.context else None

    # ------------------------------------------------------------ training
    def teach(self, pairs, turn):
        rec = self.model.teach(pairs, turn)
        moved = self._hand_down(turn)
        return {**rec, "demoted": moved, "levels": self.levels()}

    def _hand_down(self, turn):
        """What this level cannot hold goes to its context: the facts it is
        forgetting first, then the oldest, until at most `cap` remain."""
        m = self.model
        failing = m.failing() if self.demote_failing else []
        keep = [i for i in range(len(m.facts)) if i not in set(failing)]
        keep.sort(key=lambda i: m.taught[i])              # oldest first
        over = max(0, len(keep) - self.cap)
        victims = [(i, "forgetting") for i in failing] + [(i, "cap") for i in keep[:over]]
        if not victims:
            return 0
        if self.context is None:
            self.context = Recursive(rules=self.rules, seed=self.seed, cap=self.cap, depth=self.depth + 1,
                                     merge=self.merge, demote_failing=self.demote_failing, **self.kw)
        down = []
        for i, why in victims:
            for vec in m.cue_vectors(i):                   # the cue as a vector: not words
                down.append((vec, m.facts[i]))
            self.demoted.append({"turn": int(turn), "fact": m.facts[i], "why": why, "to": self.depth + 1})
        self.context.teach(down, turn)
        m.forget([i for i, _ in victims])
        return len(victims)

    # ------------------------------------------------------------ readout
    def readout(self, query, k=4, trained=True):
        return self.readout_vec(features(query), k, trained)

    def levels_readout(self, v, k, trained):
        """Every level's own top-k for this vector, with the level's score
        statistics over all of its facts, nearest level first."""
        rows = [{**r, "depth": self.depth} for r in self.model.readout_vec(v, k, trained)]
        stats = self.model.score_stats(v, trained)
        out = [(rows, stats)]
        if self.context is not None:
            out += self.context.levels_readout(v, k, trained)
        return out

    def readout_vec(self, v, k=4, trained=True):
        levels = self.levels_readout(v, k, trained)
        if len(levels) == 1:
            return levels[0][0][:k]
        if not trained:
            # the matching baseline is flat: every fact under the shared step-0 matrix
            rows = sorted((r for rows, _ in levels for r in rows), key=lambda r: -r["score"])
            return rows[:k]
        out, seen = [], set()
        if self.merge == "z":
            # scores from different weights do not compare; a score's distance from its own level's
            # mean, in that level's spread, does. A level too small to have a spread borrows the largest.
            floor = max((sd for _, (mu, sd, n) in levels if n >= 5), default=1e-6) or 1e-6
            ranked = sorted((((r["score"] - mu) / max(sd, floor), r) for rows, (mu, sd, n) in levels for r in rows),
                            key=lambda t: -t[0])
            for _, r in ranked:
                if len(out) < k and r["fact"] not in seen:
                    out.append(r); seen.add(r["fact"])
            return out
        rank = 0
        while len(out) < k and any(rank < len(rows) for rows, _ in levels):
            for rows, _ in levels:
                if rank < len(rows) and len(out) < k and rows[rank]["fact"] not in seen:
                    out.append(rows[rank]); seen.add(rows[rank]["fact"])
            rank += 1
        return out

    def state(self):
        st = self.model.state()
        below = self.context.state() if self.context else None
        out = {"rules_sha": self.rules_sha, "turns": self.turns, "levels": self.levels(),
               "facts": st["facts"] + (below["facts"] if below else 0),
               "pairs": st["pairs"] + (below["pairs"] if below else 0),
               "params": st["params"] + (below["params"] if below else 0),
               "steps": st["steps"] + (below["steps"] if below else 0),
               "wall_ms": round(st["wall_ms"] + (below["wall_ms"] if below else 0), 1),
               "loss_last": st["loss_last"],
               "demoted": len(self.demoted) + (below["demoted"] if below else 0),
               "why": {w: sum(1 for d in self.demoted if d["why"] == w) + (below["why"][w] if below else 0)
                       for w in ("forgetting", "cap")},
               "per_level": [st["facts"]] + (below["per_level"] if below else [])}
        return out
