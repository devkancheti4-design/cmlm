"""
THE CMLM -- the model. Two towers over hashed word features sharing a small
space:

    score(query, fact) = (phi(query) Wq) . (phi(fact) Wf)

At step 0 both towers are the SAME random matrix, so the untrained model is
MATCHING -- a random projection of the words' cosine -- and every comparison
in the record is the same model before and after training. `teach` pulls
each cue the frontier wrote onto the fact it was written for and away from
the other facts in the batch (a softmax over the batch), replaying old pairs
with every new turn: an actively training model forgets, and replay is the
only thing here that resists it.

What is where. Fact STRINGS live in a store, because a hashed vector cannot
be decoded back into words. The WEIGHTS hold what a store cannot: which
facts a question the model has never seen should bring back. A CMLM is one
file: the two towers, the step-0 matrix, and a manifest with the rules it
is bound to, the facts, the cues, and the record of its training.
"""
import hashlib, json, math, os, random, re, time, zlib

try:
    import numpy as np
except ImportError as e:                    # pragma: no cover
    raise ImportError("cmlm needs numpy (python3 -m pip install numpy): "
                      "the context model is a trained matrix, not a table") from e

D = 1 << 12            # hashed feature slots; no vocabulary, so any word has a slot
M = 64                 # the shared space both towers project into
_WORD = re.compile(r"[a-z0-9]+(?:[',.][a-z0-9]+)*")


def words(s):
    return _WORD.findall(str(s).lower())


def features(text):
    """Hashed unigrams and bigrams with signs, L2-normalised. A word never
    seen before still lands in a fixed slot, so the model can be asked
    anything; two words can share a slot, and the record would show it as
    a fact that came back for the wrong question."""
    v = np.zeros(D, dtype=np.float32)
    ws = words(text)
    for g in ws + [a + " " + b for a, b in zip(ws, ws[1:])]:
        h = zlib.crc32(g.encode())
        v[h & (D - 1)] += 1.0 if (h >> 12) & 1 else -1.0
    n = float(np.linalg.norm(v))
    return v / n if n else v


class _Adam:
    def __init__(self, shape, lr):
        self.m = np.zeros(shape, np.float32); self.v = np.zeros(shape, np.float32)
        self.t = 0; self.lr = lr

    def step(self, W, g):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g * g
        mh = self.m / (1 - 0.9 ** self.t); vh = self.v / (1 - 0.999 ** self.t)
        W -= self.lr * mh / (np.sqrt(vh) + 1e-8)


class CMLM:
    VERSION = 1

    def __init__(self, rules="", seed=0, m=M, lr=0.05, tau=0.1, replay=48, steps=20):
        # lr, tau, replay, steps: the best of a 16-point sweep over 8 seeds (final-turn recall
        # 0.60..0.70; the spread is the world's ceiling, not the optimizer's). About 24 ms of
        # training per turn on one CPU core.
        self.rules, self.seed = rules, seed
        self.lr, self.tau, self.replay, self.steps_per_turn = lr, tau, replay, steps
        rng = np.random.default_rng(seed)
        W = (rng.standard_normal((D, m)) / math.sqrt(D)).astype(np.float32)
        self.W0, self.Wq, self.Wf = W.copy(), W.copy(), W.copy()
        self._oq, self._of = _Adam(W.shape, lr), _Adam(W.shape, lr)
        self.facts, self.taught = [], []          # fact strings, the turn each was taught
        self.cues, self.pair_fact = [], []        # the curriculum: cue string -> fact index
        self._F, self._C = [], []                 # feature caches
        self.steps = 0; self.wall_ms = 0.0; self.loss = []; self.turns = 0
        self._rng = random.Random(seed)

    @classmethod
    def new(cls, rules, seed=0, **kw):
        """Empty, bound to these rules."""
        return cls(rules=rules, seed=seed, **kw)

    @property
    def rules_sha(self):
        return hashlib.sha256(self.rules.encode()).hexdigest()[:16]

    # ------------------------------------------------------------ the store
    def add_fact(self, fact, turn):
        fact = " ".join(str(fact).split())
        if fact in self.facts:
            return self.facts.index(fact)
        self.facts.append(fact); self.taught.append(int(turn)); self._F.append(features(fact))
        return len(self.facts) - 1

    @property
    def F(self):
        return np.stack(self._F) if self._F else np.zeros((0, D), np.float32)

    @property
    def C(self):
        return np.stack(self._C) if self._C else np.zeros((0, D), np.float32)

    def params(self):
        return int(self.Wq.size + self.Wf.size)

    # ------------------------------------------------------------ training
    def teach(self, pairs, turn):
        """(cue, fact) pairs from one turn. Adds the facts, keeps the pairs,
        and takes steps_per_turn gradient steps over this turn's pairs plus
        a replay sample of old ones. Returns the record of the turn."""
        new = []
        for cue, fact in pairs:
            vec = isinstance(cue, np.ndarray)          # a cue may arrive as a vector from another CMLM: no words
            if not (vec or words(cue)) or not words(fact):
                continue
            j = self.add_fact(fact, turn)
            self.cues.append(None if vec else " ".join(str(cue).split())); self.pair_fact.append(j)
            self._C.append(cue.astype(np.float32) if vec else features(cue))
            new.append(len(self.cues) - 1)
        t0 = time.perf_counter()
        losses = []
        if new:
            pool = [i for i in range(len(self.cues)) if i not in set(new)]
            old = self._rng.sample(pool, min(self.replay, len(pool)))
            for _ in range(self.steps_per_turn):
                losses.append(self._step(new + old))
        ms = (time.perf_counter() - t0) * 1000
        self.wall_ms += ms
        if losses:
            self.loss.append(losses[-1])
        return {"pairs": len(new), "steps": len(losses), "ms": round(ms, 2),
                "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
                "self_recall": round(self.self_recall(), 3)}

    def self_recall(self):
        """The model asking itself what it was taught: the fraction of its
        own cues for which the fact taught with them comes back first. Zero
        tokens and no truth needed -- and blind to paraphrase, since it can
        only re-ask in the words it was taught."""
        if not self.cues:
            return 1.0
        S = (self.C @ self.Wq) @ (self.F @ self.Wf).T
        return float(np.mean(S.argmax(axis=1) == np.array(self.pair_fact)))

    def _step(self, idxs):
        C = self.C[idxs]                                          # b x D  cues
        ys = [self.pair_fact[i] for i in idxs]
        cand = sorted(set(ys))                                    # the facts in play this step
        extra = [i for i in range(len(self.facts)) if i not in set(cand)]
        cand += self._rng.sample(extra, min(32, len(extra)))      # more negatives from the store
        pos = {i: j for j, i in enumerate(cand)}
        Y = np.array([pos[i] for i in ys])
        Fn = self.F[cand]                                         # n x D
        Q, Kf = C @ self.Wq, Fn @ self.Wf                          # b x m, n x m
        S = (Q @ Kf.T) / self.tau
        S -= S.max(axis=1, keepdims=True)
        P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
        loss = max(0.0, float(-np.log(P[np.arange(len(ys)), Y] + 1e-9).mean()))
        dS = P.copy(); dS[np.arange(len(ys)), Y] -= 1.0; dS /= len(ys)
        dQ, dK = (dS @ Kf) / self.tau, (dS.T @ Q) / self.tau
        self._oq.step(self.Wq, C.T @ dQ); self._of.step(self.Wf, Fn.T @ dK)
        self.steps += 1
        return loss

    # ------------------------------------------------------------ readout
    def readout(self, query, k=4, trained=True):
        """The k facts this query brings back, best first. trained=False reads
        the same model at step 0: matching, the baseline."""
        return self.readout_vec(features(query), k, trained)

    def readout_vec(self, v, k=4, trained=True):
        """The same, from a feature vector another model already computed."""
        if not self.facts:
            return []
        Wq, Wf = (self.Wq, self.Wf) if trained else (self.W0, self.W0)
        q = v @ Wq
        s = (self.F @ Wf) @ q
        order = np.argsort(-s)[:max(1, int(k))]
        return [{"fact": self.facts[i], "score": float(s[i]), "turn": self.taught[i], "index": int(i)}
                for i in order]

    def score_stats(self, v, trained=True):
        """Mean and spread of this model's scores for a vector over all of
        its facts, and how many facts: what another level needs to compare
        a score of this model with one of its own."""
        if not self.facts:
            return (0.0, 0.0, 0)
        Wq, Wf = (self.Wq, self.Wf) if trained else (self.W0, self.W0)
        s = (self.F @ Wf) @ (v @ Wq)
        return (float(s.mean()), float(s.std()), len(self.facts))

    def failing(self):
        """The facts this model is forgetting: those with a cue for which
        they no longer come back first. Self-measured, zero tokens."""
        if not self.cues:
            return []
        S = (self.C @ self.Wq) @ (self.F @ self.Wf).T
        top = S.argmax(axis=1)
        return sorted({j for p, j in enumerate(self.pair_fact) if top[p] != j})

    def cue_vectors(self, i):
        return [self._C[p] for p, j in enumerate(self.pair_fact) if j == i]

    def forget(self, indices):
        """Drop facts and their pairs from the store. The weights are left
        as they are: they still carry the association, harmlessly; the
        store no longer offers the fact."""
        drop = set(int(i) for i in indices)
        keep = [i for i in range(len(self.facts)) if i not in drop]
        remap = {i: n for n, i in enumerate(keep)}
        self.facts = [self.facts[i] for i in keep]; self.taught = [self.taught[i] for i in keep]
        self._F = [self._F[i] for i in keep]
        kp = [p for p, j in enumerate(self.pair_fact) if j in remap]
        self.cues = [self.cues[p] for p in kp]; self._C = [self._C[p] for p in kp]
        self.pair_fact = [remap[self.pair_fact[p]] for p in kp]

    def state(self):
        return {"rules_sha": self.rules_sha, "facts": len(self.facts), "pairs": len(self.cues),
                "params": self.params(), "steps": self.steps, "turns": self.turns,
                "wall_ms": round(self.wall_ms, 1), "loss_last": self.loss[-1] if self.loss else None}

    # ------------------------------------------------------------ the file
    def manifest(self):
        return {"version": self.VERSION, "rules": self.rules, "rules_sha": self.rules_sha, "seed": self.seed,
                "m": int(self.Wq.shape[1]), "lr": self.lr, "tau": self.tau, "replay": self.replay,
                "steps_per_turn": self.steps_per_turn, "facts": self.facts, "taught": self.taught,
                "cues": self.cues, "pair_fact": self.pair_fact, "steps": self.steps, "turns": self.turns,
                "wall_ms": self.wall_ms, "loss": self.loss[-50:]}

    def save(self, path):
        """The whole CMLM as one file. The optimizer's moments are not kept:
        training resumes with fresh moments, which the record shows as a
        slightly higher first loss on the next turn."""
        if any(c is None for c in self.cues):
            raise NotImplementedError("a CMLM taught with vector cues (from another CMLM) cannot be saved yet")
        with open(path, "wb") as f:
            np.savez(f, Wq=self.Wq, Wf=self.Wf, W0=self.W0, manifest=np.array(json.dumps(self.manifest())))
        return os.path.getsize(path)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        man = json.loads(str(z["manifest"]))
        if man.get("version") != cls.VERSION:
            raise ValueError(f"{path}: CMLM file version {man.get('version')}, this code reads {cls.VERSION}")
        m = cls(rules=man["rules"], seed=man["seed"], m=man["m"], lr=man["lr"], tau=man["tau"],
                replay=man["replay"], steps=man["steps_per_turn"])
        m.W0, m.Wq, m.Wf = z["W0"].astype(np.float32), z["Wq"].astype(np.float32), z["Wf"].astype(np.float32)
        for fact, turn in zip(man["facts"], man["taught"]):
            m.facts.append(fact); m.taught.append(turn); m._F.append(features(fact))
        for cue, j in zip(man["cues"], man["pair_fact"]):
            m.cues.append(cue); m.pair_fact.append(j); m._C.append(features(cue))
        m.steps, m.turns, m.wall_ms, m.loss = man["steps"], man["turns"], man["wall_ms"], list(man["loss"])
        return m
