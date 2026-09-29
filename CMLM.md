# cmlm: a context mapping language model in place of a context window

    python3 -m cmlm --fake                       # one fake session: real training, real recall, estimated tokens
    python3 -m cmlm --fake --seeds 40            # measured over seeds, both worlds
    python3 -m cmlm --fake --file me.cmlm        # run it twice: the second run loads the file and keeps training
    python3 -m cmlm --inspect me.cmlm            # what the file holds
    python3 -m cmlm --file me.cmlm               # live, questions on stdin: unmeasured here, no key

If the frontier model is Claude, its context here is not a one-million-token
window. It is a small model that starts **empty**, is **bound to the
frontier's rules**, and **fills by training** as the person works. The
frontier keeps no transcript. It reaches the CMLM through two tools:
`recall`, which asks the CMLM what it holds about something and gets facts
back, model to model; and `teach`, with which the frontier trains the CMLM
on what the turn established. What survives a turn is in the CMLM's
weights and store, and the CMLM is **a file**: it is carried between
sessions and between frontier models, and it depends on no model's cache.

    turn:  question ---> frontier   (system = the rules; tools = recall, teach; no history)
                         recall(query, k) ---> the CMLM's readout, as a tool result      (model to model)
                         ... the answer ...
                         teach(pairs)     ---> gradient steps on the CMLM                (the frontier trains it)

The limit on what it can hold is training, not tokens. What the frontier
did not teach is gone. The record shows what came back and what did not,
per probe, against a truth the CMLM cannot see. As of this writing the
**live path is unmeasured**: there is no key on this machine, and the
numbers below come from a fake world with a scripted frontier.

## The claim, with its conditions attached

> Per turn the frontier model reads *system + readout + question*, bounded,
> instead of *system + the whole transcript + question*, growing --
> **provided** (1) what the question needs was taught, which is the
> teacher's choice and has no theoretical answer; (2) the CMLM brings it
> back for words it has never seen, which is the recall measured below;
> (3) `k` is large enough for what the question needs -- with six facts on
> a topic and four asked for, recall cannot exceed 0.667, and the record
> prints that ceiling; and (4) the saving is counted against a *cached*
> transcript, where the window path reads history at a tenth of the price.

Condition (4) decides where the saving is. Under the fake world's sizes
(rules 164 tokens, a turn 50 tokens, a readout of four facts, two round
trips per turn because the frontier pulls before it answers):

| situation | transcript path, input | CMLM path, input |
|---|---|---|
| warm session, 30 turns (1,584 tokens of transcript) | 0.10 x 1,534 read + 1.25 x 50 written = about 216 token-equivalents | 426 tokens, uncached |
| warm session, 10,000 tokens of transcript | about 1,060 | 426 |
| new session, or the cache expired (1,584 tokens) | 1,584 read at full price, 1,980 if it is also written | 426 |
| past the context window | not possible | 426 |

So inside one short warm session this path costs **more** input, not less:
two round trips re-send the rules, and the rules plus tool definitions
are about 544 tokens by the chars/4 estimate, at the edge of Opus 5's
512-token cache minimum (they were 460 before the `key` fields, under it).
Whether that block caches is decided by the API's tokenizer, and the usage
fields will say; if it does, round trip two and every later turn read it
at a tenth of the price. It wins past a few thousand tokens of transcript, across sessions,
after any gap over the TTL, and past the window -- which is the case the
vision names: the person comes back. Output tokens are unchanged, plus the
pairs the frontier writes into `teach`. `chorus measure`'s finding applies
here too: output, not input, is most of a bill.

## Measured, in the fake world, over 40 seeds

Ten topics. Each teaching turn asks about one topic in **cue** words and is
answered with one fact in **fact** words between two filler sentences. In
the *paraphrase* world cue words and fact words are disjoint, so a probe
shares no word with the fact it needs; in the *overlap* world the cue words
are the fact words. Probes run every five turns on every topic taught so
far, **through the recall tool**, and are scored on the frontier's answer.
The scripted frontier quotes the readout back, so this is the CMLM's recall
as the frontier would see it. The same model at step 0 is *matching* -- a
random projection of the words' cosine -- and the *window* is the last 320
tokens of transcript.

**30 turns, three facts per topic, k = 4 (ceiling 1.000 at the last turn):**

| recall@4 at the last turn | paraphrase | overlap |
|---|---|---|
| trained CMLM | **0.662** (worst seed 0.567; beats matching in 40 of 40) | **0.612** (worst 0.433; 38 of 40) |
| the same model at step 0 (matching) | 0.135 | 0.403 |
| last 320 tokens of transcript | 0.228 | 0.225 |
| the full transcript | 1.000 at 1,584 tokens | 1.000 at 1,591 tokens |

Facts taught more than fifteen turns earlier: trained 0.698, matching
0.172, window 0.208 (paraphrase). Sent per turn: 426 tokens over two round
trips, flat from turn five on (max 433). Training: 671 ms per session on
one CPU core, about 22 ms per turn.

**60 turns, six facts per topic, k = 4 (ceiling 0.667 at the last turn):**

| recall@4 at the last turn | paraphrase | overlap |
|---|---|---|
| trained CMLM | **0.416** of the truth, 62% of what k allowed (worst 0.317; 40 of 40) | **0.399** (worst 0.300; 39 of 40) |
| matching | 0.060 | 0.256 |
| window | 0.114 | 0.113 |
| the full transcript | 1.000 at 2,996 tokens | 1.000 at 3,015 tokens |

Facts taught more than fifteen turns earlier: trained 0.525 against 0.817
for recent ones. **That gap is forgetting, and it grows with the session.**
Replay of old pairs with every new turn is the only thing here that resists
it; the record shows what it did not hold. Sent per turn stayed at 426.

**Where the words overlap, training still helped.** A probe's three words
are drawn from a topic's six fact words and a fact holds three of them, so
matching gets part of the way (0.403) and training adds 0.2; in the
paraphrase world it adds 0.5. The gain is smaller where matching already
had it, not zero, and the record says how much.

**Persistence.** A CMLM taught 20 turns of one world and saved, then loaded
and taught 20 turns of another, still returned 13 of the first world's 20
facts; before the second world it returned 16. The file is 3.1 MB: two
towers of 4096 x 64 and the step-0 matrix, a fixed size, plus a manifest
with the facts and cues. A 30-turn transcript is about 6 KB. **The saving
is in what the frontier reads per turn, not in bytes on disk.**

## What a CMLM is, exactly

Two towers over hashed word features sharing a 64-dimensional space:
`score(query, fact) = (phi(query) Wq) . (phi(fact) Wf)`. Hashed unigrams and
bigrams, no vocabulary, so any word has a slot. At step 0 both towers are
the *same* random matrix, so the untrained model is matching and every
comparison above is the same model before and after training. `teach`
pulls each cue onto the fact it was written for and away from the other
facts in the batch (a softmax over the batch, temperature 0.1), with the
new pairs and 48 replayed old ones, 20 steps of Adam per turn. The learning
rate, temperature, replay and steps are the best of a 16-point sweep over
8 seeds; the sweep moved the last-turn recall between 0.60 and 0.70, so the
ceiling is the world's, not the optimizer's.

Fact **strings** live in a store, because a hashed vector cannot be decoded
back into words. The **weights** hold what a store cannot: which facts a
question the model has never seen should bring back. A generative memory,
whose weights would hold the words themselves, was not built: it can
invent a fact, and a retrieved string is either there or not. The
substrate sits behind two calls, `teach(pairs, turn)` and `readout(query,
k)`, so a heavier model can replace it and be measured in the same world;
torch is installed on this machine and nothing here uses it yet.

## Decisions, and why

**Pull, not push.** The frontier decides when to ask, in what words, and
how many facts to bring back; a readout chosen before the frontier saw the
question would decide for it. The price is a second round trip per turn,
counted above.

**The frontier teaches.** The `teach` tool is how the frontier trains the
CMLM, in the same turn, with no second model. If the frontier does not
call it, a pattern teacher writes the pairs -- the question as cue, the
sentences that carry a number as facts -- and the record says `pattern`,
not `frontier`. It is free and degraded: it cannot read a negation and does
not know which sentence mattered.

**A teach-only message ends the turn.** Every `recall` is answered and
called back; a message whose only tool call is `teach` is the last one,
taught and not called back. The turn's messages are dropped, so nothing
dangles.

**No transcript is kept for the model, and it is counted anyway.** Every
turn computes what the transcript path would have sent, so the two paths
sit side by side in the record. It is never in a request.

**Bound to the rules.** A new CMLM takes the frontier's rules; the file
carries them and their hash. A file made under other rules is used, and
`rules_match` is false in the record -- flagged, not refused, because the
person may mean it.

**Strict tools, fallbacks on, adaptive thinking.** Both tools are
`strict: true` with closed schemas, so the frontier can only ever produce a
query, a count, or (cue, fact) strings. `fallbacks: "default"` re-runs a
declined request server-side; a refusal ends the turn and teaches nothing.
`thinking` is not set: Opus 5 runs adaptive thinking by default, and
`effort` is the dial.

**The fake frontier is scripted and says so.** Its model name is
`fake-frontier`, which prices as unpriced; every token count in fake mode
is chars/4 and the report says FAKE.

## A CMLM for CMLMs: recursion against forgetting

    python3 -m cmlm --fake --tree --seeds 20 --turns 90 -k 9      # children close, a parent routes

The flat model forgets as it fills, and the record above shows it. The
recursive form, `--tree`, is expedition's move applied to memory: a child
CMLM that has filled is **closed** -- its weights never move again -- and a
new child opens; a **parent** CMLM, whose "facts" are the children, learns
from the same cues which child a question should be asked of. Recall asks
the parent to rank the children, asks each for k facts, and interleaves by
rank. The frontier sees the same two tools; the recursion is inside.

Two triggers close a child, and the record says which fired. **Self-recall:**
after every teach the child re-asks itself every cue it was taught and
counts how often the fact taught with it comes back first; under 0.9 it
closes. **The cap:** at 30 facts, the size at which the flat record was
still at 0.66, it closes regardless.

Measured over 20 seeds, paraphrase world, recall at the last turn:

| | flat | tree, self-recall 0.9 + cap 30 | tree, cap 30 only |
|---|---|---|---|
| 60 turns, k = 4 (ceiling 0.667) | 0.411 | 0.446 | |
| 60 turns, k = 6 | 0.479 | 0.558 | **0.578** |
| 90 turns, k = 9 | 0.435 | **0.572** | |
| old facts, 60 turns, k = 6 | 0.606 | 0.615 | 0.638 |
| old facts, 90 turns, k = 9 | 0.558 | 0.638 | |
| recent facts, 90 turns, k = 9 | 0.928 | 0.924 | |

**It works where k lets it.** With k at least the facts a topic holds, the
tree adds 0.08 to 0.14 at the last turn, in 20 of 20 seeds. At k = 4 with
six facts per topic the interleave spreads four slots over three or four
children and the gain is 0.035.

**It pins old facts at the closed child's level, not at the recent level.**
Old facts came back at 0.64, which is where a 30-fact flat model sits
(0.662), not the 0.92 of recent facts -- recent facts live in a small open
child with few distractors. Recursion stops the decline past the cap; it
does not make old facts as good as new ones.

**A model cannot see its own forgetting.** The flat model's self-recall on
the cues it was taught stayed at 0.97 at 60 turns while its recall of old
*paraphrases* had fallen to 0.53: it fits its own words and cannot re-ask
in other words. The self-measured trigger closed children early and did
slightly worse than the cap alone (0.558 against 0.578), and with the cap
alone the parent ranked the right child first or second every time (1.0
against 0.87). The cap is read off the record rather than chosen by the
model, and it turned out to be the better rule. A live teacher writing a
second cue per fact, held out, would give the model a paraphrase to test
itself on; not built.

**Cost.** Training time doubles -- 2.8 s against 1.4 s per 60-turn session,
about 45 ms per turn, since the parent trains too. Sent tokens are the
same. A tree is not saved to a file yet.

## The CMLM as the frontier of its own CMLM

    python3 -m cmlm --fake --recursive --seeds 20 --turns 120 -k 12

The tree adds a router. The recursive form, `--recursive`, adds nothing:
it makes the relation between Claude and its CMLM the same at every level.
A CMLM keeps at most `cap` facts of its own. For the rest it is a frontier
to a CMLM of its own: it recalls from it and teaches it. After every teach
it hands down the oldest facts it cannot hold, its context does the same,
and the depth grows with the session. Every level holds a bounded store,
so every level replays all of its own pairs on every turn and none of them
forgets within itself. Recall asks this level, then its context, and so on
down, and interleaves the levels by rank, nearest first, each fact marked
with the depth it came from. Between two CMLMs the hop is **not words**:
the cue travels down as the feature vector the level already computed,
and only the fact travels as the string the store must keep for the
frontier. The hop into Claude is words, because the API takes tokens.

Measured over 20 seeds, paraphrase world, recall at the last turn, k equal
to the facts a topic holds so the ceiling is 1:

| | flat | tree (cap 30) | recursive (cap 30) |
|---|---|---|---|
| 60 turns, k = 6 | 0.479 | 0.578 | 0.550 |
| 90 turns, k = 9 | 0.435 | 0.591 | 0.545 |
| 120 turns, k = 12 | 0.384 | | 0.532 |
| old facts, 120 turns | 0.514 | | 0.616 |
| recent facts, 120 turns | 0.950 | | 0.947 |

**Recall stops falling with the session.** The flat model goes 0.479,
0.435, 0.384 as the session doubles; the recursive one goes 0.550, 0.545,
0.532, and its old facts hold at 0.62 across all three lengths, where the
flat model's fall to 0.51. That is the claim the recursion was built for,
and in this world it holds. Levels in seed 1 at 120 turns: 30, 30, 30, 30.

**The tree is a little higher, and the record says why.** Its closed
children were trained only on the facts they hold and then frozen; the
recursive top level keeps training, so its weights still carry the facts
it handed down, and at 60 turns that costs 0.03. The recursive form buys
with that 0.03 a structure that needs no router, orders itself by age, and
speaks to its own context in vectors. Whether that trade is right depends
on what the heavier substrate does with a frozen weight set.

**Two things measured worse, and are off by default.** Handing down a
fact the moment its own cue stops returning it first (self-measured
forgetting) created a tiny third level early and cost 0.05 at 60 turns,
the same lesson as the tree's trigger. Merging levels by a score
normalised within each level, instead of by rank, cost 0.06 here: two full
levels hold three facts each of every topic, and a three-plus-three
interleave matches that exactly. Where levels are unequal or facts are
skewed the normalised merge may win; `merge="z"` is kept for that test.

**Cost.** Training time grows with depth -- 6.9 s for a 120-turn session
against 3.2 s flat, about 58 ms per turn -- because a hand-down trains
the level below. Sent tokens are the same. A cascade is not saved to a
file yet: a level taught in vectors has no cue strings to save, and
`save()` says so rather than writing a file it could not load.

## living-fused bolted on: the exact organ beside the trained map

    python3 -m cmlm --fake --fused --seeds 20 --turns 60 -k 6          # any shape: add --tree or --recursive
    python3 -m cmlm --fake --fused --file me.cmlm                       # me.cmlm and me.cmlm.life.json beside it

A CMLM is gradient-trained: it maps the words a person asks with to facts
stated in other words, and it is fuzzy where a map must be. Beside it,
`--fused` bolts on a **Life** from
[living-fused](https://github.com/devkancheti4-design/living-fused)
(`life/life.py`, MIT, by the same author; imported from the checkout via
`LIFE_PATH`, not copied): an exact-key, recency-dominant, integer count
table. The two fail in opposite directions, which is why they fuse. `teach`
now carries a key per fact, `subject.attribute`, and the Life learns it
verbatim while the map trains on the cue; `recall` now carries a key, and
the Life answers it verbatim or **ABSTAINS**, with its counts blended into
the map's readout at read time under its own gate, w = t/(t+C). The
frontier sees the same two tools with one more field each.

What the Life makes a CMLM. **Online without gradients:** a closed child of
the tree or a deep level of the cascade never trains again, but a fact of
theirs can still be revised in the Life -- newest wins, history kept -- with
no weight moving. **Load-bearing:** an exact key returns the fact or a
structural ABSTAIN, never a guess, and the store has a byte-exact identity
(`sha`) that float weights cannot have. **Saved where the weights cannot
be:** the cascade's vector-taught levels have no file; the Life beside them
does, atomically.

Measured over 20 seeds, every key taught probed at every checkpoint, three
keys never taught probed beside them:

| 60 turns, k = 6 | flat + Life | tree + Life | cascade + Life |
|---|---|---|---|
| exact by key, 5,665 probes | 1.000 | 1.000 | 1.000 |
| false ABSTAIN on a known key | 0.000 | 0.000 | 0.000 |
| ABSTAIN on 720 unknown keys | 1.000 | 1.000 | 1.000 |
| newest value on 1,711 revised-key probes | 1.000 | 1.000 | 1.000 |
| paraphrase recall at the last turn | 0.479 | 0.578 | 0.550 |
| paraphrase recall without the Life | 0.479 | 0.578 | 0.550 |

**The map is unchanged.** Paraphrase recall is identical with and without
the Life, in every shape, and so are the tokens sent, to within the three
tokens the key field adds to a tool call (429 against 426 at 30 turns). The
Life adds exactness, revision and abstention; it adds no paraphrase, which
it scores 0 on by design, and no tokens.

**Revisions land in frozen children.** Fourteen of the 39 keys in a 60-turn
world are stated twice with a new value -- the same topic and kind, a
natural revision. In the tree, the earlier value often sits in a closed
child that will never train again; the Life returned the newest on every
one of the 1,711 revised-key probes. Without it the map holds both
sentences and ranks them by the words, not by which came last.

**INTEGRATION.md's six mistunes, as applied.** (1) The gate is the Life's
own, C = 0.25: one assertion by the frontier is one fact, and it speaks at
w = 0.8. (2) Only `teach` writes the Life; `recall` never does, so a
question cannot poison a row. (3) Writes are recency-dominant, which is
what makes revision work. (4) Keys are exact strings, disjoint by
construction; the hashed collisions live in the map's weights, not here.
(5) Counts are Python ints and cannot overflow. (6) The store is integer and
the float blend happens at read time only, so the Life's `sha` is exact
across machines while the map's weights are not.

**Licensing.** living-fused's root is AGPL-3.0; only `life/life.py`, which
is MIT, is imported, and nothing from the root -- the same discipline
NOTICE.md keeps between `neos/` and `mailagent/`.

**Unmeasured.** Whether Claude writes good keys into `teach` and asks with
them in `recall` -- the scripted frontier always does both.

## A tool-recovering model: killing the overhead, and what it cost

    python3 -m cmlm --fake --fused --toolmodel --seeds 20 --turns 60 -k 6

The cascade removed the transcript. What it could not remove: the tool
definitions, about 380 tokens re-sent on every call, and the recall round
trip before every answer. Both exist only because the frontier decides
what to ask and what to teach. `--toolmodel` puts an **empty model with
the CMLM's architecture** beside the frontier whose "facts" are labels:
which key a question asks for, which key a stated sentence belongs under.
While the frontier calls the tools, every call is a training pair. When
the model can recover the calls itself, the loop switches: the readout is
fetched before the frontier is called, the frontier answers under short
rules with **no tools in one round trip**, and the teach is recovered from
its answer, the sentences that carry a value under the key it predicts.

**With a permissive gate it took over, and here is what that bought and
cost** (20 seeds, 60 turns, switch after 30 keyed examples):

| per turn | tools every turn | recovered |
|---|---|---|
| sent to the frontier | 449 tokens | 175 tokens |
| round trips | 2 | 1 |
| paraphrase recall at the last turn | 0.479 | 0.478 |
| Life exact by key | 1.000 | 0.734 |
| false ABSTAIN on known keys | 0.000 | 0.111 |
| newest value on revised keys | 1.000 | 0.638 |

Sixty-one percent of the tokens and the second round trip gone, the map
untouched, and a quarter of the facts filed under the wrong name: the
recovered teach keys were right 0.46 of the time while the model committed
to a key 0.935 of the time. Tested alone on facts it had not seen, the key
head is right on 16 to 18 of 30 after 30 examples and 24 to 27 of 30 after
120, kind learned faster than topic. It is a learner with a curve, and its
softmax confidence said nothing about that curve: it committed on 28 to 30
of 30 predictions at every accuracy.

**The honest gate never switched.** Readiness is now judged
**prequentially**: every supervised example is predicted before it is
learned and scored against the frontier's key, the same lesson as the
tree's trigger, since self-recall on taught cues sits at 1.0 while unseen
accuracy is 0.57. With that gate at 0.85, over 120 turns and 20 seeds, the
model took over on 0 of 20 seeds: keys-from-facts ended at 0.663 prequential
accuracy, keys-from-questions at 0.948. So under an honest bar the overhead
was not killed in this world. What would clear it: more keyed examples than
a 120-turn session gives, or a key learner better than a hashed two-tower.
Recovering only `recall` and leaving `teach` to the frontier is the cheaper
half and is one flag away.

**Two findings that changed the design.** A frontier that quotes its
readout would teach it back: before the recovered teacher skipped any
sentence that restates the context sent or a fact already held, it re-filed
1,866 restated sentences per session as new facts and recall fell to 0.08.
And a guessed key earns one slot in the readout, the Life's current value
first, not the full blend an explicit ask earns: a wrong guess then costs
one row, not the readout.

## At the scale it is for: a million tokens deep

    python3 -m cmlm --fake --fused --recursive --seeds 5 --turns 300 --bulk 250 -k 12

The CMLM is for deep search, not for a chat under the window. `--bulk 250`
makes every answer a document of about 3,300 tokens holding one fact, so
300 turns put the transcript path at **983,286 tokens**, past the window,
while the map is taught one fact per document. Five seeds, k = 12, and
each topic ends with thirty facts, so recall@12 cannot exceed 0.400 at the
last turn; the record reads recall as a **share of that ceiling** as the
ceiling falls through the session.

| at the last turn, 300 turns | flat + Life | tree + Life | recursive + Life |
|---|---|---|---|
| recall@12 (ceiling 0.400) | 0.237 | **0.324** | **0.302** |
| share of the ceiling, every 50 turns, seed 1 | 0.68, 0.45, 0.47, 0.46, 0.53, 0.48 | 0.78, 0.60, 0.65, 0.71, 0.79, 0.82 | 0.76, 0.57, 0.60, 0.72, 0.70, 0.67 |
| old facts, taught over 15 turns earlier | 0.320 | 0.436 | 0.408 |
| self-recall on its own cues | 0.814 | 0.993 | 0.994 |
| Life: exact by key, 12,440 probes | 1.000 | 1.000 | 1.000 |
| Life: ABSTAIN on 900 unknown keys · newest on 10,330 revised | 1.000 · 1.000 | 1.000 · 1.000 | 1.000 · 1.000 |
| sent to the frontier per turn | 560 | 558 | 560 |
| the transcript path would send | 983,286 | 983,286 | 983,286 |
| levels or children at the end | 1 | 11 | 10 |
| training per 300-turn session | 10.4 s | 15.0 s | 40.3 s |

**The expanding shapes hold; the flat one falls.** As a share of what k
allows, the flat map slides from 0.68 to 0.48 over the session, and at 300
facts it can see its own forgetting even on the cues it was taught
(self-recall 0.814). The tree holds at 0.8 and the cascade at 0.6 to 0.7,
because a level never holds more than thirty facts and replays all of
them. Their old facts come back at 0.44 and 0.41 against the flat 0.32.
That is the design: the model keeps expanding so recall does not fall, and
between levels nothing is tokenized.

**The window is zero and the transcript is impossible.** A 320-token window
holds none of these documents, so the last-tokens baseline recalls
nothing, and a million-token transcript does not fit a request at all. The
frontier read about 560 tokens a turn throughout; the warm-cache caveat
from the top of this file does not arise here, since there is no
transcript path to cache.

**The Life does not care how deep it is.** Exact on every one of 12,440
keyed probes, ABSTAIN on all 900 unknown keys, newest on all 10,330
revised-key probes, in every shape.

**This session, real.** The conversation that built this ran to
243,167 estimated tokens over 34 of the person's
turns, one turn alone 90,579 tokens; a pattern teacher would have kept
79 numbered sentences from it. The 3D page draws that curve as the
wall beside the run.

**Live, by hand.** With no key on this machine, Claude in a session played
the frontier for 12 turns of world seed 1, writing the recall queries and
the keyed teach pairs, then answering ten probes from the readouts alone:
11 of 12 facts came back and were answered, none abstained wrongly, 12
keys exact, 418 tokens a turn. Recorded in `sandbox/live_by_hand_seed1.json`;
the cue writing was not blind, the answering was.

## What it does not do

It cannot send, execute, or read a credential; the only file it writes is
its own, through `save()`, enforced by test. It does not summarise: what
the teacher skipped is not compressed, it is gone. It does not carry
weights into the frontier: Claude's API takes tokens, so the last hop
turns the readout into words, and everything before that hop stays in
weights. It does not measure the frontier: the scripted one always asks
once with the question's words and quotes the readout; a real one may ask
again in other words and do better, or ask badly and do worse.

## Unmeasured

The live path has not run. Whether Claude calls `recall` with good queries,
writes good pairs into `teach`, and says plainly when the readout does not
hold what the question needs, is untested here. Forgetting on real
sessions is untested; the fake world's 0.525 for old facts at 60 turns is
the only number. The token arithmetic above uses chars/4; the API's
`usage` fields are the truth. Run it with a key before repeating any
number in this file.

    python3 tests/test_cmlm.py    # 21 -- no transcript, bounded input, trained vs matching over 20 seeds, the file, the tree, the cascade, the Life, the tool model, a million tokens deep
