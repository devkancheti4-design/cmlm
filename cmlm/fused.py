"""
LIVING-FUSED ON A CMLM -- the exact organ beside the trained map.

A CMLM is gradient-trained: it maps the words a person asks with to facts
stated in other words, and it is fuzzy where a map must be. Beside it,
bolted on, sits a Life from living-fused (life/life.py, MIT, by the same
author; https://github.com/devkancheti4-design/living-fused): an exact-key,
recency-dominant, integer count table. The two fail in opposite directions,
which is why they fuse:

    teach(cue, fact, key)   the CMLM trains on (cue, fact); the Life learns key -> fact, exactly
    recall(query, k, key)   the Life answers the key verbatim or ABSTAINS; its counts are blended
                            into the CMLM's readout at read time with w = t/(t+C); the CMLM
                            answers the paraphrase, keyed or not

What the Life makes a CMLM. ONLINE WITHOUT GRADIENTS: a closed child of a
tree or a deep level of a cascade never trains again, but a fact of theirs
can still be revised in the Life -- newest wins, history kept -- with no
weight moving. LOAD-BEARING: an exact key returns the verbatim fact or a
structural ABSTAIN, never a guess, and the store has a byte-exact identity
(sha) that the float weights cannot have. SAVED where the weights cannot be:
a cascade's vector-taught levels have no file; the Life beside them does.

INTEGRATION.md's six mistunes, applied here. (1) The gate is the Life's own
w = t/(t+C) with C = 0.25: one assertion by the frontier is one fact, and it
speaks (w = 0.8 after one write). (2) Only teach writes the Life; recall
never does. (3) Writes are recency-dominant, so a revision wins. (4) Keys are
exact strings, disjoint by construction; the hashed collisions live in the
CMLM's weights, not here. (5) Counts are Python ints and cannot overflow.
(6) The store is integer; the float blend happens at read time only, so the
Life's sha is exact while the CMLM's weights are not.

What it does not add: paraphrase -- the Life scores 0 on it by design and
that is the CMLM's job -- and tokens: the frontier reads the same readout.
The Life is found at LIFE_PATH, in this repository's life/ (an MIT copy of
living-fused's life/life.py), or in a living-fused checkout. living-fused's
root is AGPL-3.0; only life/life.py (MIT) is used.
"""
import math, os, sys


def find_life():
    """The Life class from the user's living-fused checkout, and where it was found."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [os.environ.get("LIFE_PATH"),
             os.path.join(here, "life"),
             os.path.join(here, "..", "living-fused", "life"),
             os.path.expanduser("~/living-fused/life"),
             os.path.expanduser("~/hybrid/living-fused/life")]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "life.py")):
            c = os.path.abspath(c)
            if c not in sys.path:
                sys.path.insert(0, c)
            from life import Life
            return Life, os.path.join(c, "life.py")
    raise ImportError("living-fused not found. Set LIFE_PATH to the directory holding life.py "
                      "(https://github.com/devkancheti4-design/living-fused, the life/ directory, MIT).")


class Fused:
    """Any CMLM -- flat, tree or cascade -- with a Life beside it."""

    def __init__(self, cmlm, life=None, life_file=None):
        Life, self.life_path = find_life()
        self.cmlm = cmlm
        self.life = life if life is not None else Life(life_file)
        self.life_file = life_file
        self.exact_hits = 0; self.abstains = 0; self.revisions = 0

    # the Session reads these from whatever it is given
    @property
    def rules(self):
        return self.cmlm.rules

    @rules.setter
    def rules(self, v):
        self.cmlm.rules = v

    @property
    def rules_sha(self):
        return self.cmlm.rules_sha

    @property
    def turns(self):
        return self.cmlm.turns

    @turns.setter
    def turns(self, v):
        self.cmlm.turns = v

    @property
    def facts(self):
        return self.cmlm.facts

    # ------------------------------------------------------------ training
    def teach(self, pairs, turn):
        """pairs of (cue, fact) or (cue, fact, key). The map trains on every
        pair; the Life learns every keyed one, exactly, newest winning."""
        plain, keyed = [], 0
        for p in pairs:
            cue, fact = p[0], p[1]
            key = str(p[2]).strip() if len(p) > 2 and p[2] is not None else ""
            plain.append((cue, fact))
            if key:
                value = " ".join(str(fact).split())
                before = self.life.recall(key)
                if before is not None and before != value:
                    self.revisions += 1
                self.life.learn(key, value)
                keyed += 1
        rec = self.cmlm.teach(plain, turn)
        return {**rec, "keyed": keyed, "life_keys": len(self.life.store), "life_sha": self.life.sha()}

    # ------------------------------------------------------------ readout
    def readout(self, query, k=4, trained=True, key="", guessed=False):
        """Without a key: the CMLM's readout. With one: the Life's exact
        answer or a structural ABSTAIN, and the Life's counts blended into
        the CMLM's top-k as a distribution -- the fused readout. A GUESSED
        key (a tool model's prediction, not an explicit ask) earns one slot:
        the Life's current value first, the map's readout after it, and no
        ABSTAIN line, since nobody asked for that key by name."""
        rows = self.cmlm.readout(query, k, trained)
        key = str(key or "").strip()
        if not key or not trained:
            return rows
        exact = self.life.recall(key)
        if guessed:
            if exact is None:
                return rows
            self.exact_hits += 1
            head = {"fact": exact, "score": 0.0, "turn": self._turn_of(exact), "index": None, "exact": True,
                    "key": key, "confidence": round(self.life.confidence(key), 3), "guessed": True}
            return [head] + [r for r in rows if r["fact"] != exact][: max(0, k - 1)]
        if exact is None:
            self.abstains += 1
            return [{"fact": None, "abstain": True, "key": key, "score": 0.0, "turn": None, "index": None}] + rows[:k]
        probs = {}
        if rows:
            m, tau = max(r["score"] for r in rows), 0.1
            z = [math.exp((r["score"] - m) / tau) for r in rows]
            s = sum(z)
            for r, zi in zip(rows, z):
                probs[r["fact"]] = probs.get(r["fact"], 0.0) + zi / s
        fused = self.life.blend(key, probs)          # w = t/(t+C); the exact value takes its share
        conf = round(self.life.confidence(key), 3)
        by_fact = {r["fact"]: r for r in rows}
        out = []
        for fact, p in sorted(fused.items(), key=lambda kv: -kv[1]):
            r = by_fact.get(fact) or {"fact": fact, "score": 0.0, "turn": self._turn_of(fact), "index": None}
            out.append({**r, "fused": round(p, 4), "exact": fact == exact, "key": key, "confidence": conf})
        self.exact_hits += 1
        return out[:k]

    def _turn_of(self, fact):
        facts = getattr(self.cmlm, "facts", [])
        taught = getattr(self.cmlm, "taught", None)
        if taught is not None and fact in facts:
            return taught[facts.index(fact)]
        return None

    def state(self):
        return {**self.cmlm.state(), "life_keys": len(self.life.store), "life_sha": self.life.sha(),
                "exact_hits": self.exact_hits, "abstains": self.abstains, "revisions": self.revisions,
                "life_path": self.life_path}

    # ------------------------------------------------------------ the files
    def save(self, path):
        """The Life always saves, atomically, beside the CMLM file; the CMLM
        saves when it can, and the record says when it could not."""
        life_file = self.life_file or (path + ".life.json")
        self.life.save(life_file)
        saved = None
        try:
            saved = self.cmlm.save(path)
        except (NotImplementedError, AttributeError) as e:
            saved = f"not saved: {e}"
        return {"life": life_file, "cmlm": saved}
