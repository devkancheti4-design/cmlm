# CMLM: context as a trained model

A frontier language model served through an API remembers nothing between calls, so a long session
re-sends its whole transcript every turn until the context window ends it. This repository replaces the
transcript with a **context-mapping language model (CMLM)**: a small model that starts empty, is bound to
the frontier's rules, and trains as the person works.

The frontier keeps no history. Before answering it calls **recall** and gets back the few taught facts that
best match the question; after answering it calls **teach**, and the CMLM trains on the new cue-and-fact
pairs locally, at no token cost. A CMLM holds at most 30 facts and hands its oldest down, as feature
vectors, to a CMLM of its own (the **cascade**), or closes into a frozen child under a routing parent (the
**tree**), so recall holds as the session grows. Beside every level an exact-key integer memory (the
**Life**, from [living-fused](https://github.com/devkancheti4-design/living-fused)) returns keyed facts
verbatim, newest value first, and abstains on unknown keys.

## Quick start

Python 3.10+ and NumPy. No API key, no network, no GPU.

```bash
python3 -m cmlm --fake                                            # one synthetic session, real training and recall
python3 -m cmlm --fake --fused --recursive --seeds 5 --turns 300 --bulk 250 -k 12   # a million tokens deep
python3 tests/test_cmlm.py                                        # 22 tests
python3 reproduce.py t9 t10                                       # regenerate the central tables into results.json
```

A live run with a real frontier needs `ANTHROPIC_API_KEY` and the `anthropic` SDK; it has not been measured.

## Measured, in a seeded synthetic world with a scripted frontier

| | flat CMLM | tree | cascade |
|---|---|---|---|
| Recall at the last turn, 60 turns, k = 6, 20 seeds | 0.479 | 0.578 | 0.550 |
| Recall@12 at 300 turns of 3,300-token documents, 5 seeds (ceiling 0.400) | 0.237 | 0.324 | 0.301 |
| Tokens per question at 300 turns: transcript path / sent to the frontier | 983,286 / 560 | | |
| Life exact on keyed probes, 300 turns | 12,440 of 12,440 | | |

On the per-turn sizes of a real 34-turn session of 243,167 tokens, the cost model gives 17,816 input tokens
for the CMLM path against 831,269 token-equivalents for a cached transcript. Token counts are characters
divided by four. Every number above is produced by `reproduce.py` and stored in `results.json`.

## Layout

    cmlm/            the package: model, loop (frontier.py), tree, cascade, fused Life, tool model, world
    life/            life.py from living-fused (MIT, life/LICENSE)
    tests/           the test suite
    sandbox/         a 12-turn run with Claude as the frontier, and its record
    data/            per-turn sizes of the real session (no message text)
    reproduce.py     regenerates every table of the paper
    results.json     the results used in the paper
    CMLM.md          the full design and measurement record
    paper/           the manuscript prepared for PeerJ Computer Science, its tables, figures and build scripts

## Status and limits

All recall results come from a synthetic world with a scripted frontier; the live path with a real frontier
through an API has not been run. The CMLM keeps only the facts the frontier teaches: what was never taught
cannot be recalled. The paper in `paper/` is a draft under preparation and has not been published.

## Licence

Code: Apache License 2.0 (`LICENSE`). `life/life.py`: MIT (`life/LICENSE`).
Built with Claude Code under the author's direction; see the acknowledgements in the paper.
