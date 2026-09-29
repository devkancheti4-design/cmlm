"""
    python3 -m cmlm --fake [--seed N] [--turns N]      one fake session through the loop
    python3 -m cmlm --fake --seeds 40 [--turns N]      measured over seeds, paraphrase and overlap worlds
    python3 -m cmlm --fake --file me.cmlm              the same, persisted: run it twice
    python3 -m cmlm --sources notes.txt "question" [--fake]
    python3 -m cmlm --inspect me.cmlm
    python3 -m cmlm [--file me.cmlm]                   live, questions on stdin (needs a key; unmeasured here)
"""
import argparse, asyncio, json, os, sys
from .model import CMLM
from .frontier import Session, RULES, K_DEFAULT, EFFORTS
from .world import World, FakeFrontier, measure, measure_seeds, report_session, report_seeds

CANNED = [
    "The short answer is yes, with one condition: the second call pays full price unless the first has begun. "
    "Check the usage block on the second call, a non-zero cache read is the only proof.",
    "No, or not the way it is usually described. Two calls with the same input share work only if the first "
    "has started answering before the second is sent; fire them together and both write, neither reads.",
]


def has_credentials():
    try:
        from chorus.server import has_credentials as h
        return h()
    except Exception:                                   # noqa: BLE001
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _open(path, seed, fused=False):
    if path and os.path.exists(path):
        m = CMLM.load(path)
        print(f"  loaded {path}: {m.state()['facts']} facts, {m.state()['turns']} turns so far, rules {m.rules_sha}")
    else:
        m = CMLM(seed=seed)
    if fused:
        from .fused import Fused
        m = Fused(m, life_file=(path + ".life.json") if path else None)
        st = m.state()
        print(f"  life beside it: {st['life_keys']} keys, sha {st['life_sha']}  ({st['life_path']})")
    return m


def _save(m, path):
    if path:
        r = m.save(path)
        st = m.state()
        if isinstance(r, dict):
            print(f"  saved {path} and {r['life']}: {st['facts']} facts, {st['life_keys']} keys (sha {st['life_sha']}), {st['turns']} turns")
        else:
            print(f"  saved {path}: {r:,} bytes, {st['facts']} facts, {st['turns']} turns, {st['steps']} steps")


async def _fake_file(path, seed, turns, k, fused=False):
    """A fake session on a persisted CMLM: the turns of world `seed` are
    taught, then every topic is probed, then the file is saved. Run it
    again with another --seed: the earlier facts still come back."""
    m = _open(path, seed, fused)
    w = World(seed=seed, turns=turns)
    s = Session(FakeFrontier(w.answers, k=k, keys=w.keys if fused else None), m, k=k)
    if not s.rules_match:
        print("  NOTE the file was created under other rules; used anyway, and the record says so")
    for t in w.turns:
        await s.ask(t["question"])
    hits = total = 0
    for topic in range(len(w.topics)):
        truth = w.truth(topic, turns)
        p = await s.ask(w.probe(topic))
        hits += sum(1 for f in truth if f in p.answer); total += len(truth)
    print(f"  taught {turns} turns of world {seed}; probes on this world: {hits} of {total} facts came back through the tool")
    _save(m, path)
    return 0


async def _sources(path, question, fake, k, effort, file, seed, fused=False):
    if not fake and not has_credentials():
        print("cmlm: no API credentials in this shell. Add --fake to answer with canned text.", file=sys.stderr)
        return 2
    units = [u.strip() for u in open(path).read().split("\n\n") if u.strip()]
    m = _open(file, seed, fused)
    if fake:
        client = FakeFrontier(CANNED, k=k)
    else:
        import anthropic
        client = anthropic.AsyncAnthropic()
    s = Session(client, m, k=k, effort=effort)
    for u in units:
        await s.learn(u)
    st = m.state()
    print(f"\n  TAUGHT  {len(units)} units from {path}: {st['facts']} facts, {st['pairs']} pairs, {st['wall_ms']:.0f} ms, 0 tokens (pattern teacher)")
    rows, text = s.recall(question, k)
    print(f"\n  READOUT the CMLM would give for {question!r}:")
    for line in text.splitlines():
        print(f"    {line}")
    t = await s.ask(question)
    print(f"\n  ANSWER  ({t.served}{', FAKE' if fake else ''})  {t.round_trips} round trips, recalled {[r['query'] for r in t.recalls]},"
          f" sent ~{t.sent_tokens:,} tokens; api in {t.usage['input']:,} out {t.usage['output']:,}; taught by {t.taught_by}")
    for line in (t.answer or "").splitlines():
        print(f"     {line}")
    _save(m, file)
    return 0


async def _live(file, seed, k, effort, fused=False):
    if not has_credentials():
        print("cmlm: no API credentials in this shell. Add --fake to run the fake world.", file=sys.stderr)
        return 2
    import anthropic
    m = _open(file, seed, fused)
    s = Session(anthropic.AsyncAnthropic(), m, k=k, effort=effort)
    print("  a session: one question per line. The frontier model gets the CMLM, never a transcript. ^D ends it.")
    for line in sys.stdin:
        q = line.strip()
        if not q:
            continue
        t = await s.ask(q)
        print(f"\n{t.answer}\n")
        print(f"  {t.served}  {t.round_trips} round trips, recalled {[r['query'] for r in t.recalls]}; sent ~{t.sent_tokens:,} tokens"
              f" (transcript path: ~{t.transcript_tokens:,}); api in {t.usage['input']:,} cache rd {t.usage['cache_read']:,}"
              f" out {t.usage['output']:,}; taught {t.pairs} pairs by {t.taught_by}, {t.train.get('ms', 0):.0f} ms\n")
        _save(m, file)
    return 0


def _inspect(path):
    m = CMLM.load(path)
    st = m.state()
    print(f"\n  {path}: {os.path.getsize(path):,} bytes")
    print(f"  bound to rules {m.rules_sha}{'  (the current RULES)' if m.rules == RULES else '  (NOT the current RULES)'}")
    print(f"  {st['params']:,} weights, {st['facts']} facts, {st['pairs']} pairs, {st['turns']} turns, {st['steps']} steps, "
          f"{st['wall_ms']:.0f} ms of training, last loss {st['loss_last']}")
    for f, t in list(zip(m.facts, m.taught))[-8:]:
        print(f"    [taught turn {t}] {f}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="cmlm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="?", help="with --sources: the question to answer over the taught units")
    p.add_argument("--fake", action="store_true", help="the fake world and a scripted frontier; no key, no network")
    p.add_argument("--seeds", type=int, default=0, help="measure over this many seeds (fake)")
    p.add_argument("--turns", type=int, default=30)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("-k", type=int, default=K_DEFAULT, help="facts per recall")
    p.add_argument("--effort", default="low", choices=list(EFFORTS))
    p.add_argument("--window", type=int, default=320, help="the transcript-window baseline, in tokens")
    p.add_argument("--bulk", type=int, default=0, help="filler sentences appended to every answer: a document per turn, the deep-search shape")
    p.add_argument("--file", help="the CMLM file: loaded if it exists, saved after")
    p.add_argument("--sources", help="a text file, one unit per paragraph, taught before the question")
    p.add_argument("--inspect", metavar="FILE", help="print what a CMLM file holds")
    p.add_argument("--tree", action="store_true", help="a CMLM for CMLMs: children close when they forget, a parent routes")
    p.add_argument("--recursive", action="store_true", help="the CMLM is the frontier of its own CMLM: it hands down what it cannot hold")
    p.add_argument("--fused", action="store_true", help="living-fused bolted on: an exact Life beside the CMLM (needs the checkout; LIFE_PATH)")
    p.add_argument("--toolmodel", action="store_true", help="a tool-recovering model learns the frontier's calls and then makes them: no tools, one round trip")
    p.add_argument("--warmup", type=int, default=30, help="--toolmodel: supervised turns before it may take over")
    p.add_argument("--cap", type=int, default=30, help="--tree / --recursive: facts a level holds before it closes or hands down")
    p.add_argument("--theta", type=float, default=0.9, help="--tree: a child closes under this self-recall@1")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    shape = dict(tree=a.tree, recursive=a.recursive, cap=a.cap, theta=a.theta, fused=a.fused, toolmodel=a.toolmodel, warmup=a.warmup, bulk=a.bulk)
    if a.inspect:
        return _inspect(a.inspect)
    if a.sources:
        if not a.question:
            p.error("--sources needs a question")
        return asyncio.run(_sources(a.sources, a.question, a.fake, a.k, a.effort, a.file, a.seed, a.fused))
    if not a.fake:
        return asyncio.run(_live(a.file, a.seed, a.k, a.effort, a.fused))
    if a.file:
        return asyncio.run(_fake_file(a.file, a.seed, a.turns, a.k, a.fused))
    if a.seeds:
        seeds = list(range(1, a.seeds + 1))
        para = asyncio.run(measure_seeds(seeds, turns=a.turns, k=a.k, window=a.window, **shape))
        over = asyncio.run(measure_seeds(seeds, turns=a.turns, k=a.k, window=a.window, overlap=True, **shape))
        if a.json:
            print(json.dumps({"paraphrase": {kk: v for kk, v in para.items() if kk != "runs"},
                              "overlap": {kk: v for kk, v in over.items() if kk != "runs"}}, indent=1))
            return 0
        print(f"\n  CMLM{' TREE' if a.tree else ''}{' RECURSIVE' if a.recursive else ''}{' + LIFE' if a.fused else ''}{' + TOOL MODEL' if a.toolmodel else ''}  measured over {a.seeds} seeds, {a.turns} turns each   [FAKE world: real training, real recall through the tool]")
        report_seeds(para, a.k, "PARAPHRASE world (a probe shares no word with the fact it needs)")
        report_seeds(over, a.k, "OVERLAP world (the probe's words are the fact's words)")
        print("\n  Training buys what matching cannot: the map from the words people ask with to the words facts\n"
              "  are stated in. Where the words overlap, matching already has part of it and the gain from training\n"
              "  is smaller; the record says how much. The live path is unmeasured: no key here.\n")
        return 0
    res = asyncio.run(measure(seed=a.seed, turns=a.turns, k=a.k, window=a.window, **shape))
    if a.json:
        print(json.dumps(res, indent=1)); return 0
    report_session(res)
    print("\n  Every question went to the frontier model with no history. It pulled what it needed through recall and\n"
          "  trained the CMLM through teach. What it did not teach is gone; the probes are the record. No key here.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
