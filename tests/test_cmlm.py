"""
Tests for cmlm: a context mapping language model in place of a context
window. No network, no key. The fake world is a scripted session with a
known truth and a scripted frontier that pulls, answers and teaches. The
tests prove that the frontier model gets no transcript, that what it sends
stays bounded while the transcript grows, that training moves recall above
matching where the question's words are not the fact's words -- measured
over seeds -- that a CMLM survives as a file, and that the record shows
what did not come back.
python3 tests/test_cmlm.py
"""
import asyncio, contextlib, io, json, os, pathlib, re, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from types import SimpleNamespace as NS
from cmlm.model import CMLM, features, words
from cmlm.frontier import (Session, PatternTeacher, RULES, RECALL, TEACH, FALLBACK_BETA, MODEL, est_tokens, sentences)
from cmlm.world import World, FakeFrontier, TOPICS, FILLER, PROBE, measure, measure_seeds
import cmlm.__main__ as cli

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES = []
def case(fn):
    CASES.append(fn); return fn

_SEEDS = {}
def seeds(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _SEEDS:
        _SEEDS[key] = asyncio.run(measure_seeds(range(1, 21), **kw))
    return _SEEDS[key]


@case
def test_the_frontier_gets_no_transcript_and_pulls_through_the_tool():
    """Every request is this turn only: the question, then the recall call
    and its readout. No earlier question, answer or filler sentence ever
    rides along. Every fact in a readout is in the store."""
    w = World(seed=1, turns=12)
    ff = FakeFrontier(w.answers, k=3)
    s = Session(ff, CMLM(seed=1), k=3)
    for t in w.turns:
        asyncio.run(s.ask(t["question"]))
    assert len(ff.calls) == 24                      # two round trips per turn: recall, then answer + teach
    earlier = []
    for i, kw in enumerate(ff.calls):
        turn = i // 2
        msgs = kw["messages"]
        assert msgs[0] == {"role": "user", "content": w.turns[turn]["question"]}
        assert len(msgs) in (1, 3) and all(m["role"] == ("user", "assistant", "user")[j] for j, m in enumerate(msgs))
        body = json.dumps(msgs)
        for text in earlier:
            assert text not in body, (turn, text)
        if len(msgs) == 3:
            assert msgs[1]["content"][0]["name"] == "recall"
            readout = msgs[2]["content"][0]["content"]
            for _, fact in re.findall(r"\[taught turn (\d+)\] (.+)", readout):
                assert fact in s.cmlm.facts
            if turn == 0:
                assert "holds nothing" in readout
        if i % 2 == 1:      # a fact may be recalled later; filler and the question must never ride
            earlier += [x for x in sentences(w.turns[turn]["answer"]) if x != w.turns[turn]["fact"]] + [w.turns[turn]["question"]]
    assert all(x.taught_by == "frontier" and x.pairs == 1 and x.round_trips == 2 for x in s.turns)
    assert all(len(x.recalls) == 1 and x.recalls[0]["k"] == 3 for x in s.turns)
    assert s.cmlm.turns == 12 and len(s.cmlm.facts) == 12


@case
def test_the_request_is_shaped_for_the_frontier():
    s = Session(FakeFrontier([]), CMLM(seed=0))
    kw = s.request([{"role": "user", "content": "q"}])
    assert kw["model"] == MODEL == "claude-opus-5" and kw["fallbacks"] == "default" and kw["betas"] == [FALLBACK_BETA]
    assert kw["output_config"] == {"effort": "low"} and "thinking" not in kw       # Opus 5: adaptive by default
    assert kw["system"][0]["text"] == RULES and kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert [t["name"] for t in kw["tools"]] == ["recall", "teach"]
    for t in (RECALL, TEACH):
        assert t["strict"] is True and t["input_schema"]["additionalProperties"] is False
        assert set(t["input_schema"]["required"]) == set(t["input_schema"]["properties"])
    assert TEACH["input_schema"]["properties"]["pairs"]["items"]["additionalProperties"] is False
    assert s.cmlm.rules == RULES and s.rules_match
    other = Session(FakeFrontier([]), CMLM.new("other rules"))
    assert other.rules_match is False and other.cmlm.rules == "other rules"
    try:
        Session(FakeFrontier([]), effort="extreme"); assert False
    except ValueError:
        pass


@case
def test_sent_stays_bounded_while_the_transcript_grows():
    res = asyncio.run(measure(seed=1, turns=30))
    log = res["turn_log"]
    tr = [x["world_transcript_tokens"] for x in log]
    sent = [x["sent_tokens"] for x in log]
    assert all(b > a for a, b in zip(tr, tr[1:])), tr
    assert max(sent[5:]) - min(sent[5:]) <= 60, sent          # once the readout is full it stays the same size
    assert res["final"]["sent_max"] * 3 < res["final"]["transcript_tokens"]
    assert all(x["round_trips"] == 2 for x in log)
    assert [x["teaching_turn"] for x in log] == list(range(1, 31))


@case
def test_step_zero_is_matching_and_training_moves_the_weights():
    a, b = CMLM(seed=5), CMLM(seed=5, steps=0)
    assert np.array_equal(a.Wq, a.W0) and np.array_equal(a.Wf, a.W0)
    w = World(seed=5, turns=10)
    for t, x in enumerate(w.turns):
        a.teach([(x["question"], x["fact"])], t); b.teach([(x["question"], x["fact"])], t)
    assert a.steps == 200 and b.steps == 0 and a.facts == b.facts and len(a.cues) == 10
    assert not np.array_equal(a.Wq, a.W0) and not np.array_equal(a.Wf, a.W0)
    for x in w.turns:
        q = w.probe(x["topic"])
        assert [r["index"] for r in a.readout(q, 4, trained=False)] == [r["index"] for r in b.readout(q, 4)]
    assert CMLM().readout("empty") == [] and a.readout("anything", 3)[0]["fact"] in a.facts
    empty = a.teach([("", "no cue"), ("no fact", "")], 99)
    assert empty["pairs"] == 0 and empty["steps"] == 0 and empty["loss_last"] is None and a.steps == 200
    assert all(l >= 0 for l in a.loss)


@case
def test_training_beats_matching_where_the_words_differ_measured_over_seeds():
    """20 seeds, 30 turns, through the tool. Thresholds sit under the
    measured rates (40 seeds: last-turn recall trained ~0.66, matching
    ~0.14, window ~0.23)."""
    r = seeds(turns=30)
    f = r["final_recall"]
    assert f["trained"] >= 0.55, f
    assert f["matching"] <= 0.40, f
    assert f["window"] <= 0.35, f
    assert r["trained_beats_matching"] >= 18, r["trained_beats_matching"]
    assert r["worst_final"] >= 0.35, r["worst_final"]
    assert r["final"]["sent_max"] < 520 < 1200 < r["final"]["transcript_tokens"]


@case
def test_where_the_words_overlap_training_buys_less():
    """The overlap world: a probe's three words are drawn from the fact's
    six. Matching already gets part of it; the gain from training must be
    smaller than in the paraphrase world, and the record says so."""
    a, b = seeds(turns=30), seeds(turns=30, overlap=True)
    fa, fb = a["final_recall"], b["final_recall"]
    assert fb["matching"] >= 0.25, fb
    assert (fb["trained"] - fb["matching"]) < (fa["trained"] - fa["matching"]), (fa, fb)


@case
def test_the_record_shows_forgetting_or_its_absence():
    r = seeds(turns=30)
    old, recent = r["by_age"]["old"], r["by_age"]["recent"]
    assert old["probes"] > 0 and recent["probes"] > 0
    assert old["trained"] is not None and old["trained"] >= 0.5, old
    p = r["runs"][0]["probes"][0]
    assert set(p) == {"turn", "topic", "age", "truth", "k", "trained", "matching", "window", "route2", "sent_tokens", "round_trips"}
    assert p["route2"] is None                                   # a flat model has no children to route to
    assert r["final_recall"]["ceiling"] == 1.0 and seeds(turns=60)["final_recall"]["ceiling"] == round(4 / 6, 3)
    assert all(0 <= q["trained"] <= q["truth"] for run in r["runs"] for q in run["probes"])


@case
def test_a_cmlm_is_a_file_that_keeps_training():
    """Save, load: identical readouts. Teach more in the second process's
    model: the earlier facts still come back, the record continues, and a
    file made under other rules is used but flagged."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "me.cmlm")
        res = asyncio.run(measure(seed=3, turns=20, cmlm=CMLM(seed=3)))
        w = World(seed=3, turns=20)
        m = asyncio.run(_model_after(seed=3, turns=20))
        n = m.save(path)
        assert 2_500_000 < n < 4_000_000, n                  # the towers are a fixed size; the store is small
        m2 = CMLM.load(path)
        assert m2.facts == m.facts and m2.cues == m.cues and m2.turns == m.turns == 20 and m2.steps == m.steps
        assert np.array_equal(m2.Wq, m.Wq) and np.array_equal(m2.Wf, m.Wf) and np.array_equal(m2.W0, m.W0)
        for topic in range(len(TOPICS)):
            q = w.probe(topic)
            assert [r["index"] for r in m2.readout(q, 4)] == [r["index"] for r in m.readout(q, 4)]
        before = sum(len(set(w.truth(t, 20)) & {r["fact"] for r in m2.readout(w.probe(t), 4)}) for t in range(10))
        w2 = World(seed=4, turns=20)
        s = Session(FakeFrontier(w2.answers), m2)
        for t in w2.turns:
            asyncio.run(s.ask(t["question"]))
        after = sum(len(set(w.truth(t, 20)) & {r["fact"] for r in m2.readout(w.probe(t), 4)}) for t in range(10))
        assert m2.turns == 40 and len(m2.facts) == 40 and after >= before * 0.5, (before, after)
        m2.save(path)
        assert CMLM.load(path).turns == 40
        odd = CMLM.new("some other rules", seed=1); odd.save(path)
        assert Session(FakeFrontier([]), CMLM.load(path)).rules_match is False
        bad = os.path.join(d, "bad.cmlm")
        with open(bad, "wb") as f:
            np.savez(f, Wq=m.Wq, Wf=m.Wf, W0=m.W0, manifest=np.array(json.dumps({"version": 99})))
        try:
            CMLM.load(bad); assert False
        except ValueError:
            pass


async def _model_after(seed, turns):
    w = World(seed=seed, turns=turns)
    s = Session(FakeFrontier(w.answers), CMLM(seed=seed))
    for t in w.turns:
        await s.ask(t["question"])
    return s.cmlm


@case
def test_a_frontier_that_does_not_teach_is_covered_and_a_refusal_teaches_nothing():
    async def plain(**kw):
        return NS(content=[NS(type="text", text="The deposit is 2,400 rupees. Nothing else.")], stop_reason="end_turn",
                  stop_details=None, model="claude-opus-5", usage=NS(input_tokens=100, output_tokens=20,
                  cache_read_input_tokens=0, cache_creation_input_tokens=0))
    s = Session(NS(beta=NS(messages=NS(create=plain))), CMLM(seed=0))
    t = asyncio.run(s.ask("how much is the deposit"))
    assert t.taught_by == "pattern" and t.pairs == 1 and t.round_trips == 1 and t.recalls == []
    assert s.cmlm.facts == ["The deposit is 2,400 rupees."] and t.cost_usd == round((100 * 5 + 20 * 25) / 1e6, 6)

    async def refuse(**kw):
        return NS(content=[], stop_reason="refusal", stop_details=NS(category="cyber"), model="claude-opus-5",
                  usage=NS(input_tokens=50, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0))
    s = Session(NS(beta=NS(messages=NS(create=refuse))), CMLM(seed=0))
    t = asyncio.run(s.ask("q"))
    assert t.refused and t.pairs == 0 and t.taught_by is None and s.cmlm.facts == [] and s.cmlm.turns == 1

    async def unknown(**kw):
        if len(kw["messages"]) == 1:
            return NS(content=[NS(type="tool_use", id="x", name="nonesuch", input={}),
                               NS(type="tool_use", id="y", name="recall", input={"query": "q", "k": 2})],
                      stop_reason="tool_use", stop_details=None, model="fake-frontier", usage=None)
        assert kw["messages"][2]["content"][0]["is_error"] is True
        return NS(content=[NS(type="text", text="done, 1 thing")], stop_reason="end_turn", stop_details=None, model="fake-frontier", usage=None)
    s = Session(NS(beta=NS(messages=NS(create=unknown))), CMLM(seed=0))
    t = asyncio.run(s.ask("q"))
    assert t.round_trips == 2 and t.cost_usd is None and t.answer == "done, 1 thing"


@case
def test_the_pattern_teacher_is_free_and_degraded():
    t = PatternTeacher()
    q = "What about the light fixture supplier?"
    ans = "Then also while after before since just. The lamp carton courier is 4,199 rupees. Nearly around again maybe perhaps probably."
    assert t.teach(q, ans) == [(q, "The lamp carton courier is 4,199 rupees.")]
    neg = t.teach(q, "The deposit is not 4,199 rupees, whatever anyone says.")
    assert neg and "not" in neg[0][1]                       # it cannot read a negation: written as a fact
    unit = "The courier dispatched the lamp carton on the 14th. Then also while after before since."
    assert t.teach(None, unit) == [(unit, "The courier dispatched the lamp carton on the 14th.")]
    assert t.teach(q, "Short one here. A much longer sentence with no number in it at all.") == \
        [(q, "A much longer sentence with no number in it at all.")]
    assert t.teach(q, "") == []


@case
def test_the_world_is_what_the_doc_says():
    allw = [w for f, c in TOPICS for w in f + c]
    assert len(allw) == len(set(allw)) and not set(allw) & set(FILLER)
    for f, c in TOPICS:
        assert not set(f) & set(c)
    w = World(seed=3, turns=20)
    stop = set(words(PROBE))
    for topic in range(len(TOPICS)):
        pw = set(words(w.probe(topic))) - stop
        assert len(pw) == 3
        for fact in w.truth(topic, 20):
            assert not pw & set(words(fact)), (pw, fact)
        assert all(re.search(r"\d", f) for f in w.truth(topic, 20))
    o = World(seed=3, turns=20, overlap=True)
    for topic in range(len(TOPICS)):
        assert set(words(o.probe(topic))) - stop <= set(TOPICS[topic][0])
    assert features("").sum() == 0 and abs(float(np.linalg.norm(features("two words"))) - 1) < 1e-5
    assert est_tokens("") == 1 and est_tokens("x" * 40) == 10


@case
def test_the_cli_runs_in_fake_mode_and_refuses_plainly_without_a_key():
    with tempfile.TemporaryDirectory() as d:
        src = pathlib.Path(d) / "notes.txt"
        src.write_text("The courier dispatched the lamp carton on the 14th. Then also while.\n\n"
                       "The landlord holds a deposit of 2,400 rupees. Nearly around again.\n")
        me = os.path.join(d, "me.cmlm")
        for argv, needle in ((["--fake", "--turns", "8"], "PROBES"),
                             (["--fake", "--seeds", "2", "--turns", "8"], "OVERLAP world"),
                             (["--fake", "--turns", "6", "--json"], '"final_recall"'),
                             (["--fake", "--file", me, "--turns", "6"], "saved"),
                             (["--fake", "--file", me, "--turns", "6", "--seed", "2"], "loaded"),
                             (["--inspect", me], "the current RULES"),
                             (["--fake", "--tree", "--turns", "8", "--cap", "3"], "PROBES"),
                             (["--fake", "--tree", "--seeds", "2", "--turns", "8", "--cap", "3"], "TREE  "),
                             (["--fake", "--recursive", "--seeds", "2", "--turns", "8", "--cap", "3"], "CASCADE  "),
                             (["--sources", str(src), "when is the deposit due", "--fake", "--file", me], "TAUGHT  2 units")):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                assert cli.main(argv) == 0, argv
            assert needle in out.getvalue(), (argv, out.getvalue()[-400:])
        assert CMLM.load(me).turns == 6 + 10 + 6 + 10 + 2 + 1      # two fake sessions with probes, two units, one question
    real = cli.has_credentials
    cli.has_credentials = lambda: False
    try:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            assert cli.main([]) == 2 and cli.main(["--sources", "x", "q"]) == 2
        assert "no API credentials" in err.getvalue()
    finally:
        cli.has_credentials = real


@case
def test_cmlm_cannot_send_or_execute_and_writes_only_its_own_file():
    src = {p.name: p.read_text() for p in (ROOT / "cmlm").glob("*.py")}
    alltext = "".join(src.values())
    for bad in ("subprocess", "smtplib", "os.system", "shutil", "webbrowser", "exec(", "eval(", "import pickle"):
        assert bad not in alltext, bad
    writes = re.findall(r'open\([^)]*["\'][wa]', alltext)
    assert writes == ['open(path, "w'] and 'with open(path, "wb") as f:' in src["model.py"]   # save(), and nothing else
    for name, text in src.items():
        top = [l for l in text.splitlines() if l.startswith(("import ", "from "))]
        assert not any("anthropic" in l for l in top), name


@case
def test_a_cmlm_for_cmlms_closes_children_and_routes():
    """The tree: children close at the cap, the parent ranks every child,
    the readout is k distinct facts, and the matching baseline is flat --
    the same ranking a flat model at step 0 gives over the same facts."""
    from cmlm.tree import Tree
    w = World(seed=2, turns=12)
    t = Tree(seed=2, cap=5, theta=0.0)
    s = Session(FakeFrontier(w.answers), t)
    for x in w.turns:
        asyncio.run(s.ask(x["question"]))
    st = t.state()
    assert st["children"] == 3 and len(st["closed"]) == 2 and st["facts"] == 12 and st["pairs"] == 12
    assert all(c["why"] == "cap" and c["facts"] == 5 for c in st["closed"]) and [len(c.facts) for c in t.children] == [5, 5, 2]
    assert t.rules == RULES and s.rules_match and t.rules_sha == t.parent.rules_sha and t.parent.facts == ["child 0", "child 1", "child 2"]
    flat = CMLM(seed=2, steps=0)
    for x in w.turns:
        flat.teach([(x["question"], x["fact"])], 0)
    for topic in range(len(TOPICS)):
        q = w.probe(topic)
        assert sorted(t.route(q)) == [0, 1, 2]
        rows = t.readout(q, 4)
        assert len(rows) == 4 and len({r["fact"] for r in rows}) == 4 and all(t.where(r["fact"]) is not None for r in rows)
        assert [r["fact"] for r in t.readout(q, 4, trained=False)] == [r["fact"] for r in flat.readout(q, 4)]
    assert t.where("never taught") is None and Tree(seed=1).readout("empty") == [] and Tree(seed=1).route("q") == [0]
    x = asyncio.run(s.ask(w.turns[0]["question"]))                # the same fact again lands in the open child, once in the readout
    assert t.where(w.turns[0]["fact"]) == 0 and len({r["fact"] for r in t.readout(w.probe(w.turns[0]["topic"]), 12)}) == len(t.readout(w.probe(w.turns[0]["topic"]), 12))


@case
def test_the_tree_holds_recall_where_the_flat_model_forgets():
    """10 seeds, 60 turns, k=6 so the ceiling is 1. Measured over 20 seeds:
    last-turn recall 0.479 flat against 0.578 tree (cap only); old facts
    0.606 against 0.638. And the flat model cannot see its own forgetting:
    self-recall on its cues stays high while old paraphrases fall."""
    flat = asyncio.run(measure_seeds(range(1, 11), turns=60, k=6))
    tree = asyncio.run(measure_seeds(range(1, 11), turns=60, k=6, tree=True, theta=0.0))
    assert tree["final_recall"]["trained"] > flat["final_recall"]["trained"] + 0.03, (flat["final_recall"], tree["final_recall"])
    assert tree["by_age"]["old"]["trained"] >= flat["by_age"]["old"]["trained"] - 0.02, (flat["by_age"], tree["by_age"])
    assert tree["tree"]["children_mean"] == 3.0 and tree["tree"]["why"] == {"forgetting": 0, "cap": 20}
    assert tree["final_recall"]["route2"] >= 0.9 and flat["tree"] is None
    assert abs(tree["final"]["sent_tokens"] - flat["final"]["sent_tokens"]) <= 20         # the frontier sees the same tool
    assert flat["self_recall_last"] >= 0.9 and flat["by_age"]["old"]["trained"] <= 0.7, flat["self_recall_last"]


@case
def test_the_cmlm_is_the_frontier_of_its_own_cmlm():
    """The cascade: a level keeps at most `cap` facts and hands the rest
    down as (vector cue, fact) pairs -- no words between two CMLMs. Recall
    asks every level and marks the depth; the matching baseline is flat;
    a level taught with vector cues cannot be saved yet, and says so."""
    from cmlm.cascade import Recursive
    w = World(seed=2, turns=12)
    r = Recursive(seed=2, cap=5)
    s = Session(FakeFrontier(w.answers), r)
    for x in w.turns:
        asyncio.run(s.ask(x["question"]))
    st = r.state()
    assert st["levels"] == 3 and st["facts"] == 12 and st["pairs"] == 12 and st["per_level"] == [5, 5, 2]
    assert r.rules == RULES == r.context.rules == r.context.context.rules and s.rules_match
    assert all(c is not None for c in r.model.cues) and all(c is None for c in r.context.model.cues)   # words above, vectors below
    assert st["why"]["cap"] + st["why"]["forgetting"] == st["demoted"] == 9      # 7 handed down from the top, 2 of those again from the middle
    newest = [x["fact"] for x in w.turns[-5:]]
    assert sorted(r.model.facts) == sorted(newest)                        # the top holds the newest cap facts
    assert {r.where(x["fact"]) for x in w.turns} <= {0, 1, 2} and r.where("never taught") is None
    flat = CMLM(seed=2, steps=0)
    for x in w.turns:
        flat.teach([(x["question"], x["fact"])], 0)
    for topic in range(len(TOPICS)):
        q = w.probe(topic)
        rows = r.readout(q, 6)
        assert len(rows) == 6 and len({x["fact"] for x in rows}) == 6 and {x["depth"] for x in rows} <= {0, 1, 2}
        assert rows[0]["depth"] == 0                                        # nearest level first
        assert [x["fact"] for x in r.readout(q, 4, trained=False)] == [x["fact"] for x in flat.readout(q, 4)]
    try:
        r.context.model.save("/nonexistent/x.cmlm"); assert False
    except NotImplementedError:
        pass
    m = CMLM(seed=1)
    m.teach([("a cue here", "fact one is 1"), ("another cue", "fact two is 2"), ("a third cue", "fact three is 3")], 0)
    m.forget([1])
    assert m.facts == ["fact one is 1", "fact three is 3"] and m.pair_fact == [0, 1] and m.cues == ["a cue here", "a third cue"]
    assert m.failing() == [] and len(m.cue_vectors(0)) == 1 and Recursive(seed=1).readout("empty") == []


@case
def test_the_cascade_holds_recall_where_the_flat_model_forgets():
    """10 seeds, 60 turns, k=6 (ceiling 1). Every level replays all of its
    own pairs, so nothing forgets within a level; the thresholds sit under
    the 20-seed record in CMLM.md."""
    flat = asyncio.run(measure_seeds(range(1, 11), turns=60, k=6))
    casc = asyncio.run(measure_seeds(range(1, 11), turns=60, k=6, recursive=True))
    assert casc["final_recall"]["trained"] > flat["final_recall"]["trained"] + 0.03, (flat["final_recall"], casc["final_recall"])
    assert casc["by_age"]["old"]["trained"] >= flat["by_age"]["old"]["trained"] - 0.02, (flat["by_age"], casc["by_age"])
    assert casc["cascade"]["levels_mean"] >= 2 and casc["cascade"]["why"]["cap"] > 0 and flat["cascade"] is None
    assert abs(casc["final"]["sent_tokens"] - flat["final"]["sent_tokens"]) <= 20      # the frontier sees the same tool


def _life_available():
    try:
        from cmlm.fused import find_life
        find_life(); return True
    except ImportError as e:
        print(f"        SKIP: {e}")
        return False


@case
def test_living_fused_bolted_on_is_exact_revisable_and_changes_nothing_about_the_map():
    """The Life beside the CMLM: keyed facts are exact, unknown keys ABSTAIN,
    a revision's newest value wins with history kept, the identity is a
    byte-exact sha that survives save and load, and a readout without a key
    is the CMLM's own, untouched."""
    if not _life_available():
        return
    from cmlm.fused import Fused
    import cmlm.frontier as fr
    f = Fused(CMLM(seed=1))
    s = Session(FakeFrontier([]), f)
    assert f.rules == RULES and s.fused and s.rules_match and f.turns == 0
    rec = f.teach([("what is the deposit", "The deposit is 2,400 rupees.", "flat.deposit"),
                   ("when is the visa interview", "The interview is Friday at 9:30.", "visa.interview"),
                   ("an unkeyed fact", "The lamp carton is reference 48213.", "")], 0)
    assert rec["pairs"] == 3 and rec["keyed"] == 2 and rec["life_keys"] == 2 and len(f.facts) == 3
    sha1 = f.life.sha()
    assert f.life.recall("flat.deposit") == "The deposit is 2,400 rupees." and f.life.recall("nonesuch") is None
    rows = f.readout("how much was the down payment", 3, key="flat.deposit")
    assert rows[0]["exact"] and rows[0]["fact"] == "The deposit is 2,400 rupees." and rows[0]["fused"] >= 0.8 and rows[0]["confidence"] == 0.8
    assert all("fused" in r for r in rows) and len(rows) <= 3
    ab = f.readout("anything", 3, key="nonesuch.key")
    assert ab[0].get("abstain") and ab[0]["key"] == "nonesuch.key" and ab[0]["fact"] is None and len(ab) == 4
    plain = CMLM(seed=1); plain.teach([("what is the deposit", "The deposit is 2,400 rupees."), ("when is the visa interview", "The interview is Friday at 9:30."), ("an unkeyed fact", "The lamp carton is reference 48213.")], 0)
    q = "how much was the down payment"
    assert [r["index"] for r in f.readout(q, 3)] == [r["index"] for r in plain.readout(q, 3)]           # no key: the map alone
    assert [r["index"] for r in f.readout(q, 3, trained=False)] == [r["index"] for r in plain.readout(q, 3, trained=False)]
    f.teach([("the deposit changed", "The deposit is 2,600 rupees.", "flat.deposit")], 1)             # a revision, newest wins
    assert f.revisions == 1 and f.life.recall("flat.deposit") == "The deposit is 2,600 rupees."
    assert [v for v, _ in f.life.history("flat.deposit")] == ["The deposit is 2,400 rupees.", "The deposit is 2,600 rupees."]
    top = f.readout(q, 4, key="flat.deposit")
    assert top[0]["fact"] == "The deposit is 2,600 rupees." and top[0]["exact"]
    assert f.life.sha() != sha1 and f.state()["revisions"] == 1 and f.state()["exact_hits"] == 2 and f.state()["abstains"] == 1
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "me.cmlm")
        r = f.save(path)
        assert r["life"] == path + ".life.json" and os.path.exists(r["life"]) and isinstance(r["cmlm"], int)
        g = Fused(CMLM.load(path), life_file=path + ".life.json")
        assert g.life.sha() == f.life.sha() and g.life.recall("flat.deposit") == "The deposit is 2,600 rupees."
        from cmlm.cascade import Recursive
        c = Fused(Recursive(seed=1, cap=2))
        for i in range(5):
            c.teach([(f"cue number {i}", f"Fact number {i} is {i}.", f"k.{i}")], i)
        r = c.save(os.path.join(d, "casc.cmlm"))
        assert str(r["cmlm"]).startswith("not saved") and os.path.exists(r["life"]) and c.life.recall("k.0") == "Fact number 0 is 0."
    for t in (RECALL, TEACH):
        assert set(t["input_schema"]["required"]) == set(t["input_schema"]["properties"])
    assert "key" in RECALL["input_schema"]["properties"] and "key" in TEACH["input_schema"]["properties"]["pairs"]["items"]["properties"]


@case
def test_the_life_is_measured_beside_the_map_and_the_map_is_unchanged():
    """10 seeds, 30 turns: exact by key 1.0, ABSTAIN on unknown keys 1.0,
    newest value on every revised key -- and the paraphrase recall and the
    tokens sent are identical to the run without the Life."""
    if not _life_available():
        return
    plain = seeds(turns=30)
    fused = asyncio.run(measure_seeds(range(1, 21), turns=30, fused=True))
    e = fused["exact"]
    assert e["hit"] == 1.0 and e["false_abstain"] == 0.0 and e["abstain"] == 1.0 and e["known"] > 500 and e["unknown"] > 100, e
    assert e["revised"] > 50 and e["newest"] == 1.0, e
    assert fused["recall"] == plain["recall"] and fused["final_recall"] == plain["final_recall"], (fused["recall"], plain["recall"])
    assert abs(fused["final"]["sent_tokens"] - plain["final"]["sent_tokens"]) <= 5, (fused["final"], plain["final"])   # the key field in a tool call
    assert fused["life"] and not plain["life"]
    assert all(r["model"]["life_sha"] and r["model"]["life_keys"] >= 15 for r in fused["runs"])
    with tempfile.TemporaryDirectory() as d:
        me = os.path.join(d, "me.cmlm")
        for argv, needle in (((["--fake", "--fused", "--turns", "8"]), "LIFE  exact by key"),
                             ((["--fake", "--fused", "--file", me, "--turns", "6"]), "saved"),
                             ((["--fake", "--fused", "--file", me, "--turns", "6", "--seed", "2"]), "life beside it: ")):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                assert cli.main(argv) == 0, argv
            assert needle in out.getvalue(), (argv, out.getvalue()[-300:])
        assert os.path.exists(me + ".life.json")


@case
def test_the_bulk_world_is_the_deep_search_shape_and_leaves_small_worlds_unchanged():
    """bulk appends filler after the turn's own draws, so a bulk=0 world is
    byte-identical to before; with bulk=250 a turn is a 3,300-token
    document holding one fact, and 300 turns pass 980k transcript tokens."""
    a, b = World(seed=1, turns=30), World(seed=1, turns=30, bulk=250)
    assert [t["fact"] for t in a.turns] == [t["fact"] for t in b.turns] and [t["key"] for t in a.turns] == [t["key"] for t in b.turns]
    assert all(bt["answer"].startswith(at["answer"]) for at, bt in zip(a.turns, b.turns))
    assert 3000 < est_tokens(b.turns[0]["answer"]) < 3600 and PatternTeacher().teach("q", b.turns[0]["answer"]) == [("q", b.turns[0]["fact"])]
    big = World(seed=1, turns=300, bulk=250)
    assert big.transcript_tokens(300, RULES) > 950_000
    res = asyncio.run(measure(seed=1, turns=40, k=4, bulk=250, fused=True))
    assert res["final"]["transcript_tokens"] > 130_000 and res["final"]["sent_max"] < 520
    assert res["exact"]["hit"] == 1.0 and res["final_recall"]["window"] == 0.0        # a 320-token window holds no document


@case
def test_a_tool_model_learns_the_calls_and_the_honest_gate_holds_it_back():
    """With a permissive gate it takes over, kills the second round trip and
    most of the tokens, leaves the map untouched, and misfiles keys; with
    the prequential gate at 0.85 it does not take over in 60 turns. Both
    are the measured record, not a claim."""
    if not _life_available():
        return
    from cmlm.toolmodel import ToolModel, KeyHead
    from cmlm.fused import Fused
    # the key head alone: it learns, and its softmax confidence is not calibration
    w = World(seed=1, turns=180); facts = [(t["fact"], t["key"]) for t in w.turns]
    h = KeyHead(seed=1)
    for f, k in facts[:30]:
        h.learn(f, k)
    ok30 = sum(1 for f, k in facts[30:60] if h.predict(f)[0] == k)
    for f, k in facts[30:120]:
        h.learn(f, k)
    ok120 = sum(1 for f, k in facts[120:150] if h.predict(f)[0] == k)
    assert 10 <= ok30 <= 24 and ok120 >= 20 and ok120 > ok30, (ok30, ok120)
    assert h.predict("nothing like a fact at all")[0] and h.self_recall() >= 0.9
    # the loop with a permissive gate (theta 0): it switches after 30 keyed examples
    perm = asyncio.run(measure_seeds(range(1, 6), turns=60, k=6, fused=True, toolmodel=True, warmup=30))
    # measure_seeds builds ToolModel(seed, warmup) with the default gate; emulate the permissive one directly
    w2 = World(seed=2, turns=60)
    tm = ToolModel(seed=2, warmup=30, theta=0.0)
    s = Session(FakeFrontier(w2.answers, k=6, keys=w2.keys), Fused(CMLM(seed=2)), k=6, toolmodel=tm)
    for t in w2.turns[:30]:
        asyncio.run(s.ask(t["question"]))
        for key in list(w2.newest(30))[:2]:
            asyncio.run(s.ask(w2.exact_probe(key)))              # keyed recalls: the ask head's supervision
    assert not tm.ready() or tm.ask.n >= 30
    while not tm.ready():
        for key in list(w2.newest(30)):
            asyncio.run(s.ask(w2.exact_probe(key)))
    before = len(s.turns)
    for t in w2.turns[30:]:
        asyncio.run(s.ask(t["question"]))
    rec = [x for x in s.turns[before:] if x.mode == "recovered"]
    assert len(rec) == 30 and all(x.round_trips == 1 and not x.recalls[0].get("k") is None for x in rec)
    assert all(x.taught_by in ("toolmodel", None) for x in rec) and tm.switched_at is not None
    tools_sent = sum(x.sent_tokens for x in s.turns[:30] if x.mode == "tools") / 30
    rec_sent = sum(x.sent_tokens for x in rec) / len(rec)
    assert rec_sent < 0.6 * tools_sent, (tools_sent, rec_sent)
    assert "tools" not in s.client.calls[-1] or not s.client.calls[-1].get("tools")
    assert tm.state_()["recovered"] == sum(1 for x in s.turns if x.mode == "recovered") >= 30   # probes after the switch are recovered turns too
    # the honest gate: prequential accuracy, not self-recall; it does not switch in 60 turns
    assert perm["toolmodel"]["turns_recovered"] == 0 and all(x is None for x in perm["toolmodel"]["switched_at"])
    assert perm["toolmodel"]["state_prequential"] < 0.85 and perm["final"]["sent_tokens"] > 400
    assert perm["recall"] == asyncio.run(measure_seeds(range(1, 6), turns=60, k=6, fused=True))["recall"]


@case
def test_the_vector_hop_is_the_word_hop_exactly_and_costs_no_tokens():
    """Between two CMLMs the cue travels as the feature vector the level
    already computed. Teaching the level below on that vector reaches the
    same weights and the same readouts as teaching it on the words, since
    features() is deterministic; the hop carries no tokens. Proposition 8."""
    w = World(seed=1, turns=60)
    teach = [(t["question"], t["fact"]) for t in w.turns[:60]]
    probes = [t["question"] for t in w.turns[:60]] + ["deadline for the review", "who signed the northern lease"]
    a, b = CMLM(seed=0), CMLM(seed=0)
    for n, (q, f) in enumerate(teach):
        a.teach([(q, f)], n)                 # words
        b.teach([(features(q), f)], n)       # the vector: what a cascade hands down
    assert np.array_equal(a.Wq, b.Wq) and np.array_equal(a.Wf, b.Wf)
    for pr in probes:
        ra, rb = a.readout(pr, 4), b.readout(pr, 4)
        assert [r["fact"] for r in ra] == [r["fact"] for r in rb]
        assert max(abs(x["score"] - y["score"]) for x, y in zip(ra, rb)) == 0.0
    assert sum(est_tokens(q) for q, _ in teach) > 0        # what a word hop would have carried
    assert not hasattr(features(teach[0][0]), "decode")   # a hashed vector has no words to give back


if __name__ == "__main__":
    ok = fail = 0
    for fn in CASES:
        try:
            fn(); print(f"  PASS  {fn.__name__}"); ok += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {fn.__name__}\n        {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            fail += 1
    print(f"\n  {ok} passed, {fail} failed")
    sys.exit(1 if fail else 0)
