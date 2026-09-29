"""Build the PeerJ Computer Science manuscript, its tables and figure legends from results.json."""
import json, math, os, re
from docxgen import Doc
from refs import REFS, CITES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR", os.path.join(HERE, "submission")); os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.environ.get("RESULTS", os.path.join(HERE, "..", "results.json"))))
t4, t5, t6, t7, t9, t10 = R["t4"], R["t5"], R["t6"], R["t7"], R["t9"], R["t10"]
VH, RS = R["vector_hop"], R["real_session"]
SH = ("flat", "tree", "recursive"); NM = {"flat": "flat", "tree": "tree", "recursive": "cascade"}


def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"
def n0(x): return f"{x:,.0f}"
def ci(s): return f"{s['ci95'][0]:.3f} to {s['ci95'][1]:.3f}"
def ms(s): return f"{s['mean']:.3f} ({s['sd']:.3f})"
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
def num(n): return WORDS.get(n, f"{n:,}")


# ------------------------------------------------------------ derived numbers
T4P, T4O, T4L = t4["30 turns, k=4, paraphrase"], t4["30 turns, k=4, overlap"], t4["60 turns, k=4, paraphrase"]
def t5k(turns, k, s): return t5[f"{turns} turns, k={k}, {s}"]
G5 = [(60, 6), (90, 9), (120, 12)]
fin5 = {(t, s): t5k(t, k, s)["per_seed"]["final"]["mean"] for t, k in G5 for s in SH}
M9 = {s: t9[s]["per_seed"] for s in SH}
M10 = {s: t10[s]["per_seed"] for s in SH}
CK = {s: t10[s]["seed1_checkpoints"] for s in SH}
share = {s: [c["share"] for c in CK[s]] for s in SH}
EX10 = t10["flat"]["exact"]
L6 = {s: t6[f"60 turns, k=6, {s} + Life"] for s in SH}
TOOLS, PERM, HON, KH = t7["tools every turn"], t7["permissive gate"], t7["prequential gate 0.85, 120 turns"], t7["key head alone"]
tm_t, tm_p = TOOLS["toolmodel"], PERM["toolmodel"]
# turns 1-30 are identical in both runs (same seeds, deterministic, the gate not yet open), so the tools-every-turn
# mean over turns 31-60 follows from its mean over all 60 turns and the permissive run's mean over turns 1-30
same_turns_tools = ((tm_t["sent_tools"] * tm_t["turns_tools"] - tm_p["sent_tools"] * tm_p["turns_tools"])
                    / (tm_t["turns_tools"] - tm_p["turns_tools"]))
cut = 1 - tm_p["sent_recovered"] / same_turns_tools
kh_pts = sorted(int(k) for k in KH["mean_accuracy"])
kh_acc = {n: KH["mean_accuracy"][str(n)] for n in kh_pts}
kh_rows = KH["rows"]
def kh_range(n, field): xs = [r[field] for r in kh_rows if r["examples"] == n]; return min(xs), max(xs)
W1M, Rr, dd, Qmin = 1_000_000, 164, 3300, 10
tstar = (W1M - Rr) // (dd + Qmin) + 1
warm_small = 0.1 * (T4P["final"]["transcript_tokens"] - 50) + 1.25 * 50 + 15
be = lambda B: (B - 15 - (1.25 - 0.1) * 50) / 0.1
rs_turns = RS["per_turn"]; rs_tot = RS["totals"]
sizes = [l.split(",") for l in open(os.path.join(HERE, "..", "data", "real_session_sizes.csv")).read().split("\n")[1:] if l]
rs_total_tokens = int(sizes[-1][3]); rs_mean = rs_total_tokens / len(sizes); rs_big = max(int(x[2]) for x in sizes)
rs_cmlm = [r["cmlm"] for r in rs_turns]
t_mean = int((W1M - Rr) // rs_mean) + 1; t_big = int((W1M - Rr) // rs_big) + 1
life_same = all(abs(L6[s]["per_seed"]["final"]["mean"] - M9[s]["final"]["mean"]) < 1e-9 for s in SH)
sent_diff = max(abs(L6[s]["per_seed"]["sent"]["mean"] - M9[s]["sent"]["mean"]) for s in SH)
hon_sw = HON["switched"]
AB, SW = R["ablations"], R["sweep"]
def delta(label):
    v = AB[label]["vs_paper_setting"]
    return v
def delta_text(label):
    v = delta(label)
    return (f"changed recall at the last turn by {v['mean']:+.3f} (95% CI {v['ci95'][0]:+.3f} to {v['ci95'][1]:+.3f}) and "
            f"was better on only {num(v['wins'])} of {v['of']} seeds")
sw_used = [r for r in SW["rows"] if r["lr"] == 0.05 and r["tau"] == 0.1][0]["final_recall_mean"]
sent9 = [M9[s]["sent"]["mean"] for s in SH]

# ------------------------------------------------------------ text
TITLE = ("Context as a trained model: a language model that keeps no transcript and trains an expanding cascade "
         "of small models instead")

ABSTRACT = [
    ("Background", "A language model served through a stateless application programming interface (API) remembers nothing "
     "between calls, so a long session re-sends its whole transcript on every turn until the context window ends it. In "
     "deep-search sessions, where every turn returns a document, that limit arrives within a few hundred turns."),
    ("Methods", "We replace the transcript with a context-mapping language model (CMLM): a small model that starts empty, "
     "is bound to the frontier model's rules and is trained as the person works. The frontier keeps no history. Before "
     "answering it asks the CMLM for the facts it needs through a recall tool, and afterwards it trains the CMLM through a "
     "teach tool. A CMLM that fills hands its oldest facts, as feature vectors rather than words, to a CMLM of its own, so "
     "the structure expands while every level stays small; an exact-key integer memory beside each level holds keyed facts "
     "verbatim. We prove that the request per turn is bounded for any session length, that the exact memory is "
     "revision-correct and that every capped level trains on all of its pairs, and we measure recall in a seeded synthetic "
     "world with a scripted frontier against a hidden truth."),
    ("Results", f"At 300 turns of 3,300-token documents, where the transcript path would send {n0(t10['flat']['final']['transcript_tokens'])} "
     f"tokens per question, the frontier read about {n0(round(t10['flat']['final']['sent_tokens'], -1))} tokens per turn. A single CMLM's recall fell to "
     f"{f2(share['flat'][-1])} of its ceiling while the expanding tree held {f2(share['tree'][-1])} and the cascade {f2(share['recursive'][-1])}; over 20 seeds "
     f"at 60 turns the tree and the cascade beat the single model by {f3(t9['tree']['vs_flat']['mean'])} (95% CI {ci(t9['tree']['vs_flat'])}) and "
     f"{f3(t9['recursive']['vs_flat']['mean'])} ({ci(t9['recursive']['vs_flat'])}). The exact memory answered all {n0(EX10['known'])} keyed probes correctly. "
     f"A model that learns the frontier's tool calls cut tokens per turn by {cut:.0%} under a permissive gate but filed "
     f"{1 - tm_p['key_ok_recovered']:.0%} of recovered facts under the wrong key, and under a predict-then-learn gate it "
     + ("never took over. " if hon_sw == 0 else f"took over on {hon_sw} of 20 seeds. ")
     + f"Applied to a real 34-turn session of {n0(rs_total_tokens)} tokens, the cost model gives {n0(rs_tot['cmlm'])} input tokens "
     f"for the CMLM path against {n0(rs_tot['warm'])} token-equivalents for a cached transcript."),
    ("Conclusions", "Context can be held as a trained, expanding model rather than as re-sent text, so the limit on what it "
     "holds becomes training rather than tokens. All recall results are synthetic and all token counts are estimates; "
     "the live path with a real frontier model remains to be measured."),
]

INTRO = [
    "Every call to a frontier language model through an API starts blank. The caller sends the instructions, the tool "
    "definitions and the question, and for a conversation it also sends every earlier turn. Prompt caching makes the "
    "repeated prefix cheaper but not free, and the context window puts a hard ceiling on how much history a call can "
    "carry (Packer et al., 2023). Compressed prefixes keep a per-call cost, and compression loses what it did not "
    "anticipate (Mu, Li & Goodman, 2023; Deng et al., 2024). In a deep-search session, where each turn brings back a "
    "document of thousands of tokens, the ceiling arrives within a few hundred turns and the session ends.",
    "The usual answers keep the transcript and shrink it: a sliding window; summaries of earlier turns, bounded per level "
    "(Nijkamp et al., 2026); compression of context into latent vectors that a decoder is trained to read (Chevalier et "
    "al., 2023; Liu & Qiu, 2025; Li et al., 2026); retrieval over a store of past text (Khandelwal et al., 2020; Lewis et "
    "al., 2020); or memory tiers that the model manages through function calls (Packer et al., 2023; Yu et al., 2026). "
    "Each still treats context as text to be re-sent, and the latent variants need a decoder trained to read the latents, "
    "which a closed API does not offer. This paper asks a different question: what if the frontier model's context were "
    "not a transcript but a model?",
    "We describe a context-mapping language model (CMLM). It is a small model that starts empty and is bound to the rules "
    "of the frontier model it serves, in the lineage of fast-weight memories (Schmidhuber, 1992), memory networks (Weston, "
    "Chopra & Bordes, 2015) and test-time memorisation (Hardt & Sun, 2023; Behrouz, Zhong & Mirrokni, 2024), but placed "
    "beside the frontier rather than inside it. The frontier keeps no transcript. Before answering it calls recall, asking "
    "the CMLM in its own words for what it needs; after answering it calls teach, handing the CMLM cue-and-fact pairs on "
    "which the CMLM takes gradient steps on local compute, at no token cost. What survives a turn lives in the CMLM's "
    "weights and store, and the CMLM is a file that can be carried between sessions and between frontier models. If the "
    "frontier is a model with a million-token window, its context is instead a small model that fills by training, so "
    "the limit on what it can hold is training rather than tokens.",
    "A single CMLM forgets as it fills, the sequential-learning interference described by McCloskey & Cohen (1989), and "
    "replay (Rebuffi et al., 2017) only delays it. The relation between frontier and CMLM is therefore made recursive: a "
    "CMLM keeps a bounded number of facts and, for the rest, is itself the frontier of a CMLM of its own, handing down "
    "what it cannot hold. Between two CMLMs the cue travels as a feature vector, not as words; only the hop into the "
    "frontier is tokenized. A tree variant closes a full child and opens a new one under a routing parent, in the spirit "
    "of index-routed hierarchical memory (Sun & Zeng, 2025). Beside every level sits an exact-key integer memory, the "
    "Life component of living-fused (Kancheti, 2026), so keyed facts are held verbatim, revised with the newest value "
    "winning and answered with a structural abstention when unknown, which trained memories alone do not deliver on "
    "exact-recall tasks (V P, 2026).",
    "The contributions of this paper are:",
]
CONTRIB = [
    "The CMLM loop: a frontier model with no transcript that pulls its context through a recall tool and trains that "
    "context through a teach tool.",
    "Two expanding structures, a tree and a cascade, in which the CMLMs communicate in vectors, with recall measured "
    "against a falling ceiling as sessions grow from 30 to 300 turns.",
    "The fusion of an exact-key integer memory with the trained map, measured for exactness, abstention and revision at "
    "every depth.",
    "A tool-recovering model that learns the frontier's own tool calls in order to remove the tool definitions and the "
    "recall round trip, together with a predict-then-learn gate that keeps it from taking over before it is ready.",
    "Formal guarantees for the mechanism (bounded input, exact revision, full replay under a cap, the interleave bound, "
    "the caching break-even and an exact, token-free vector hop) and a seeded synthetic world with a hidden truth, in "
    "which every reported number is computed by released code.",
]
RQS = [
    "**RQ1.** Can a small model trained only from a frontier's teach calls return, for a question in other words, the "
    "facts the question needs, and by how much does training beat the same model untrained?",
    "**RQ2.** Does bounding the store per model, by a tree or by a cascade, hold recall as the session grows where a "
    "single model's recall falls?",
    "**RQ3.** Does an exact-key integer memory fused at read time deliver exact recall, abstention and revision without "
    "changing the map's paraphrase recall or its token cost?",
    "**RQ4.** Can a model that learns the frontier's tool calls remove the tool definitions and the recall round trip, "
    "and under what gate is that safe?",
    "**RQ5.** At the scale the design is for, a session whose transcript passes a million tokens, what does the frontier "
    "read per turn and what recall do the expanding shapes keep?",
]
INTRO_END = ("The Guarantees subsection proves what can be proved about the mechanism; RQ1 and RQ2 concern the map's recall, "
             "which no theorem gives and the experiments settle. All recall results come from a synthetic world and a "
             "scripted frontier, token counts are estimates, and the live path with a real frontier model has not been "
             "run; the Limitations subsection lists these limits before the Conclusions draw on the numbers.")

RELATED = [
    ("Context as weights.", "Storing what a sequence has said in parameters rather than in an attention window goes back "
     "to fast-weight programmers (Schmidhuber, 1992) and memory networks (Weston, Chopra & Bordes, 2015), and has returned "
     "as test-time training and as architectures that update a memory module at inference (Hardt & Sun, 2023; Behrouz, "
     "Zhong & Mirrokni, 2024), including training-free associative memories (He et al., 2024). The CMLM is a deliberately "
     "small instance: two linear towers over hashed features (Weinberger et al., 2009), trained with a contrastive "
     "objective (Oord, Li & Vinyals, 2018) in the two-tower retrieval tradition (Huang et al., 2013). Its contribution is "
     "not the substrate but the loop around it, in which the frontier model itself writes the training pairs."),
    ("Retrieval.", "Retrieval-augmented generation (Lewis et al., 2020) and nearest-neighbour language models (Khandelwal et "
     "al., 2020) put a store beside a frozen model and read from it at inference. A CMLM is a retriever whose index is "
     "trained online on the session, and the Life beside it is an exact store. living-fused (Kancheti, 2026) supplies "
     "that store: a recency-dominant integer count table with structural abstention and a confidence-gated blend into a "
     "model's output distribution. Its integration notes name failure modes this work inherits and addresses: blending an "
     "organ that barely knows anything, writing questions into the store, and averaging values instead of overwriting them."),
    ("Latent links between models.", "Prefix-tuning (Li & Liang, 2021) and prompt tuning (Lester, Al-Rfou & Constant, 2021) "
     "show that a few dozen learned vectors can carry what many tokens carry when the receiving model was trained to read "
     "them; gist tokens (Mu, Li & Goodman, 2023) compress an instruction into a handful of such vectors; and the projectors "
     "of Flamingo (Alayrac et al., 2022) and LLaVA (Liu et al., 2023) feed a frozen language model inputs that are not text. "
     "Between two CMLMs this work already passes vectors. Into a closed frontier served by an API it cannot, because the "
     "API accepts tokens, so the last hop is words."),
    ("Forgetting and its measurement.", "Catastrophic forgetting in sequentially trained networks (McCloskey & Cohen, 1989) "
     "and replay as its standard remedy (Rebuffi et al., 2017) frame the CMLM's central problem. The finding reported here, "
     "that a model's recall of its own training cues stays near 1.0 while its recall of paraphrases falls, is a small "
     "instance of the gap between training fit and generalisation. The gate adopted for the tool-recovering model is "
     "prequential evaluation (Dawid, 1984): each example is predicted before it is learned."),
    ("API-side mechanisms.", "Prompt caching, server-side compaction that summarises earlier turns, context editing that "
     "clears old tool results, and deferred loading of tool definitions are the answers currently offered at the API "
     "boundary. All of them keep context as text. The cost analysis in the Discussion accounts for caching explicitly."),
    ("Closest work.", "No paper found in a search of arXiv on 25 September 2026 combines a frontier with no transcript, a "
     "separate small model trained through the frontier's own tool calls, a recursive cascade with vector links, and an "
     "exact integer memory fused at read time; each piece has neighbours. MemGPT (Packer et al., 2023) has the model manage "
     "memory tiers through function calls, but the tiers hold text and the model keeps a working context. An architecture "
     "of levels, ticks and cascaded intelligence (Nijkamp et al., 2026) keeps continuity across context resets with bounded "
     "summaries per level; hierarchical memory with index-based routing (Sun & Zeng, 2025) resembles the tree's parent; and "
     "small language models have been given memory operations for agents (Zhang et al., 2026a). Context cascade compression "
     "(Liu & Qiu, 2025), AutoCompressors (Chevalier et al., 2023) and end-to-end context compression (Li et al., 2026) "
     "compress context into latent tokens for a decoder, the latent link that cannot be made into a closed API. Titans "
     "(Behrouz, Zhong & Mirrokni, 2024), test-time training on nearest neighbours (Hardt & Sun, 2023), test-time training "
     "by context distillation (Wang et al., 2026) and fast-weight product-key memory (Zhao & Jones, 2026) put memory into the "
     "model's own weights, whereas this work leaves the frontier untouched and trains a model beside it. OoO-Spec (Zhang et "
     "al., 2026b) predicts tool calls with a small sidecar model to make calling faster; the tool-recovering model here "
     "predicts them to remove the tool definitions, and reports the gate under which that is unsafe. A correction to "
     "surprisal-aware residual test-time training (V P, 2026) finds that such memories fail exact-recall tasks, which is "
     "the case for an exact organ beside a trained map."),
    ("Research gap.", "The literature divides into four roads, and each leaves the same question open. Memory inside the "
     "model's own weights (Hardt & Sun, 2023; Behrouz, Zhong & Mirrokni, 2024; Wang et al., 2026; Zhao & Jones, 2026) needs "
     "the frontier's parameters, which a served frontier does not expose. Memory as text tiers managed by the model (Packer "
     "et al., 2023; Sun & Zeng, 2025; Yu et al., 2026; Zhang et al., 2026a) still spends tokens re-reading context and "
     "cannot read past the window at answer time. Latent compression of context for a decoder trained to read it "
     "(Chevalier et al., 2023; Liu & Qiu, 2025; Li et al., 2026) requires that decoder, so it is closed to an API-served "
     "frontier. An exact store (Kancheti, 2026) recalls verbatim but cannot answer a paraphrase. No work found (i) trains a "
     "separate context model from the frontier's own tool calls with the frontier untouched, (ii) bounds forgetting by "
     "expanding that model into levels linked by vectors rather than by summarising, (iii) fuses an exact integer memory "
     "with a trained map at read time, (iv) reports recall against an explicit ceiling with an untrained and a token-window "
     "baseline, or (v) measures when a model that has learned the frontier's tool calls is safe to take them over. Table 1 "
     "places the approaches side by side on the properties this gap turns on."),
]

# ------------------------------------------------------------ build
d = Doc(title=TITLE, author="Devieswar Kancheti")
d.title_p(TITLE)
d.p("Devieswar Kancheti^{1}")
d.p("^{1} [Department], [Institution], [City], [State or Province], [Country]")
d.p("")
d.p("Corresponding Author:")
d.p("Devieswar Kancheti^{1}")
d.p("[Institution address, City, Postal code, Country]")
d.p("Email address: [institutional email address]")
d.page_break()

d.heading("Abstract", 1)
for h, t in ABSTRACT:
    d.p(f"**{h}.** {t}")
d.page_break()

d.heading("Introduction", 1)
for t in INTRO: d.p(t)
d.bullets(CONTRIB, numbered=True)
d.p("The experiments answer five research questions.")
d.bullets(RQS)
d.p(INTRO_END)

d.heading("Related Work", 1)
for h, t in RELATED:
    d.p(f"**{h}** {t}")

# ------------------------------------------------------------ methods
d.heading("Materials & Methods", 1)
d.heading("System overview", 2)
d.p("Figure 1 shows the whole system. Words pass only between the person and the frontier; the frontier reaches its "
    "context through two tools; levels of CMLMs hand facts down as vectors; a Life sits beside each level; and a "
    "tool-recovering model, once it is ready, makes the frontier's calls for it. The only tokenized hop inside the system "
    "is between the frontier and level 0, and it is tokenized only because a served API accepts nothing else.")

d.heading("The context-mapping language model", 2)
d.p("A CMLM maps a question to the facts it needs. Text is turned into a hashed feature vector of unigrams and bigrams "
    "with signs, normalised to unit length, in 4,096 slots, so that any word, seen or not, has a slot. Two towers project "
    "the features into a shared 64-dimensional space and score a query against a fact. At step zero both towers are the "
    "same random matrix, so the untrained model is a random projection of the words' cosine similarity; this untrained "
    "model is called matching and serves as a baseline, so that every comparison is the same model before and after "
    "training. Teaching a turn adds its cue-and-fact pairs and takes 20 steps of Adam on a batch of the new pairs plus up "
    "to 48 replayed old pairs, with a softmax over the facts in play at temperature 0.1, pulling each cue towards its fact "
    "and away from the others. Fact strings live in a store, because a hashed vector cannot be decoded; the weights hold "
    "what a store cannot, namely which facts a question never seen before should bring back. A readout is the top k facts "
    "by score. " + (f"A 16-point grid of learning rate and temperature over eight seeds that are not used elsewhere moved "
    f"final recall only between {SW['range'][0]:.2f} and {SW['range'][1]:.2f}; the setting used, a learning rate of 0.05 "
    f"and a temperature of 0.1, scored {sw_used:.3f} against {SW['best']['final_recall_mean']:.3f} for the best cell"
    + (f" (a learning rate of {SW['best']['lr']} and a temperature of {SW['best']['tau']}). The reported runs keep the original "
       "setting and were not re-tuned on this grid." if abs(SW['best']['final_recall_mean'] - sw_used) > 1e-9 else ".")))

d.heading("The loop", 2)
d.p("The frontier is given a system prompt of rules and two tools, and no history. Each turn's messages are the "
    "question, the tool calls with their results, and the answer, and they are dropped when the turn ends. The recall "
    "tool takes a query in the frontier's own words, a count k and an optional exact key; its result is the readout as "
    "text. The teach tool takes pairs of a cue in a person's words, a verbatim fact and an optional key. Both tools are "
    "declared with strict, closed schemas, so the frontier can only ever produce strings and a count. A message whose only "
    "tool call is teach ends the turn without a further call. If the frontier does not teach, a pattern teacher writes the "
    "pairs (the question as cue and the sentences that carry a number as facts) and the record says so. The transcript "
    "path is counted on every turn but never sent, so the two paths sit side by side in the record.")

d.heading("Expansion: the tree and the cascade", 2)
d.p("A single CMLM forgets as it fills, because training on new pairs moves the weights that held the old ones. Two "
    "structures bound the store per model.")
d.p("The **tree** closes a child once it holds 30 facts; its weights never move again and a new child opens. A parent "
    "CMLM, whose facts are the child labels, learns from the same cues which child a question belongs to and ranks the "
    "children, and the readout interleaves their answers by rank. A self-measured trigger, closing a child when its recall "
    "of its own cues falls below 0.9, was also measured over 20 seeds at 60 turns: against the cap alone it "
    + delta_text("tree, self-recall trigger 0.9") + ", because a model fits its own cues while losing their paraphrases. The trigger is "
    "switched off in every other reported run.")
d.p("The **cascade** makes the frontier-to-CMLM relation the same at every level. A CMLM keeps at most 30 facts and, "
    "for the rest, is the frontier of a CMLM of its own: after every teach it hands its oldest facts down, and the level "
    "below does the same. Between levels the cue travels as the feature vector already computed, and only the fact "
    "travels as a string, so nothing between CMLMs is tokenized. Recall asks every level and interleaves the answers by "
    "rank, nearest level first. Every level replays all of its own pairs on every turn, so no level forgets within itself. "
    "Two alternatives were measured over the same 20 seeds: merging levels by a within-level normalised score "
    + delta_text("cascade, normalised merge") + ", and handing a fact down as soon as its own cue stops returning it "
    + delta_text("cascade, hand down on failure") + ". Both remain as options in the code.")

d.heading("The exact-key memory beside the map", 2)
d.p("Beside any CMLM sits a Life: an exact-key, recency-dominant integer count table taken unchanged from living-fused "
    "(Kancheti, 2026). The teach tool's key writes the fact into it verbatim, and the recall tool's key reads it back "
    "verbatim or returns a structural ABSTAIN. When a key is asked for, the Life's counts are blended into the map's top "
    "k as a distribution with weight $t/(t+0.25)$, the Life's own gate, so a single assertion by the frontier already "
    "speaks. A key that a model guessed, rather than one the frontier asked for, earns one slot (the Life's current value "
    "first) instead of the full blend. A revision is a second write under the same key: the newest value wins and the "
    "history is kept. Only teach writes the Life; recall never does. The store is integer and the floating-point blend "
    "happens only at read time, so the Life has a byte-exact identity that the map's weights cannot have.")

d.heading("The tool-recovering model", 2)
d.p("The tool definitions and the recall round trip exist only because the frontier decides what to ask and what to "
    "teach. A tool-recovering model with the CMLM's architecture learns those decisions from the frontier's own calls: one "
    "head maps a question to the key it asks for and another maps a stated sentence to the key it belongs under, each key "
    "by dotted part with one CMLM per position. While the frontier calls the tools, every keyed call is a training pair, "
    "predicted before it is learned so that a running prequential accuracy is kept. When that accuracy clears 0.85 on both "
    "heads after at least 30 keyed examples, the loop switches: the readout is fetched before the frontier is called, the "
    "frontier answers under short rules with no tools in one round trip, and the teach is recovered from its answer. Two "
    "rules were forced by measurement: a label taught only once is a guess and is not predicted, and a sentence that "
    "restates the context sent or a fact already held is never taught back.")

d.heading("How the models communicate", 2)
d.p("Three links carry the whole system. **CMLM to frontier.** The frontier reaches its CMLM through recall and teach. "
    "On a frontier served through an API the boundary is tokens, because the API accepts nothing else, so the readout is "
    "rendered as k facts in text, each marked with the turn in which it was taught, and the teach pairs arrive as "
    "cue-and-fact strings with an optional key. That rendering is the only place in the system where tokens are counted. "
    "On an open-weights frontier the same link could be a trained projector into the model's embedding space; that "
    "variant is proposed, not built. **CMLM to CMLM.** A CMLM never sends words to another CMLM. When a level hands a fact "
    "down it sends the cue's 4,096-dimensional feature vector, which it has already computed, together with the fact "
    "string that the store must keep for the frontier, and the level below trains on the vector directly. A query "
    "descends the cascade as the same vector, computed once at the top, and every level scores it with its own towers; the "
    "readout climbs back as ranked facts. The weights are not shared: the vector is the common language because every "
    "CMLM hashes words the same way. **CMLM to Life.** Beside every CMLM the Life speaks exact strings: a key written by "
    "the frontier goes into the integer table with its fact, and at read time the Life's counts for that key are blended "
    "into the readout. The Life never sees the CMLM's vectors, and the CMLM sees the Life's counts only through that blend.")

d.heading("Formal statement", 2)
d.p("Table 2 lists the notation. Let a session be a sequence of turns $t = 1, 2, \\dots$, each with a question $q_t$ "
    "and an answer $a_t$ from which the teacher extracts pairs $(c, f)$ of a cue and a fact, and let $F_t$ be the fact "
    "store after turn $t$.")
d.p("**Features.** Every string $x$ is mapped to a fixed-width vector by hashing its unigrams and bigrams $G(x)$ into "
    "$D = 4096$ slots with a sign:")
d.eq(r"\phi(x) = \frac{v(x)}{\lVert v(x)\rVert_2}, \qquad v(x) = \sum_{g \in G(x)} \sigma(g)\, e_{h(g)}, \qquad h: G \to \{1,\dots,D\},\ \sigma(g) \in \{-1,+1\}")
d.p("**Score.** Two towers $W_q, W_f \\in \\mathbb{R}^{D \\times m}$ with $m = 64$ project the features into a shared "
    "space, and at step zero $W_q = W_f = W_0$:")
d.eq(r"s(q, f) = \big(\phi(q)^{\top} W_q\big)\cdot\big(\phi(f)^{\top} W_f\big)")
d.p("**Training.** Teaching turn $t$ forms a batch $B_t$ of its new pairs $N_t$ and a replay sample $R_t$ of at most "
    "48 earlier pairs, and takes 20 Adam steps at learning rate 0.05 on a softmax over the candidate facts $C_t$ (the "
    "facts of $B_t$ plus up to 32 sampled from the store) at temperature $\\tau = 0.1$:")
d.eq(r"\mathcal{L}_t = -\frac{1}{|B_t|} \sum_{(c,f)\in B_t} \log \frac{\exp\big(s(c,f)/\tau\big)}{\sum_{f' \in C_t} \exp\big(s(c,f')/\tau\big)}, \qquad B_t = N_t \cup R_t")
d.p("**Readout and ceiling.** A query returns the $k$ highest-scoring facts. A probe on topic $i$ needs $n_i$ facts, so "
    "over all probes recall is bounded by a ceiling, and every late-session result is reported both raw and as a share of "
    "that ceiling:")
d.eq(r"\mathrm{readout}_k(q) = \operatorname*{arg\,top}_{f \in F_t,\, k}\; s(q, f), \qquad \mathrm{recall@}k \;\le\; \frac{\sum_i \min(k, n_i)}{\sum_i n_i}")
d.p("**Cascade.** Level $\\ell$ keeps a store $F^{(\\ell)}$ with $|F^{(\\ell)}| \\le C = 30$. After each teach the oldest "
    "$|F^{(\\ell)}| - C$ facts are handed to level $\\ell + 1$ as pairs $(\\phi(c), f)$, and level $\\ell + 1$ teaches on "
    "them and applies the same rule. A query descends as the single vector $\\phi(q)$, each level returns its own top $k$, "
    "and the readout is the rank interleave, nearest level first, truncated to $k$:")
d.eq(r"\mathrm{out}_k(q) = \mathrm{trunc}_k\Big( \mathrm{interleave}_{r=0,1,\dots}\big( \mathrm{readout}^{(0)}_k(q)[r],\ \mathrm{readout}^{(1)}_k(q)[r],\ \dots \big) \Big)")
d.p("**Tree.** A child closes at $C$ facts and its towers are frozen. A parent CMLM with label facts $\\ell_j$ is taught "
    "$(c, \\ell_j)$ for every pair taught into child $j$, ranks the children by $s(q, \\ell_j)$, and the readout "
    "interleaves the children's top $k$ in that order.")
d.p("**Life.** For key $\\kappa$ and value $v$ the integer table $N$ holds counts, and a write is recency-dominant, so "
    "the newest value wins while the history stays. An absent key returns ABSTAIN. At read time the counts are blended "
    "into the readout, $p$ being the softmax of the readout's top-$k$ scores, under the Life's own gate with "
    "$C_0 = 0.25$:")
d.eq(r"t_\kappa = \sum_{v} N[\kappa][v], \qquad N[\kappa][v] \leftarrow N[\kappa][v] + t_\kappa + 1, \qquad \mathrm{recall}(\kappa) = \operatorname*{arg\,max}_{v} N[\kappa][v]")
d.eq(r"w_\kappa = \frac{t_\kappa}{t_\kappa + C_0}, \qquad \tilde{p}(v) = (1 - w_\kappa)\, p(v) + w_\kappa \frac{N[\kappa][v]}{t_\kappa}")
d.p("**Tool-recovering model.** Each head $h$ predicts a key part before learning example $i$, giving a prequential "
    "accuracy over its $n_h$ supervised examples, and the loop switches only when both heads clear $\\theta = 0.85$ after "
    "at least 30 examples. A predicted part is emitted only if its softmax share is at least 0.6 and its label was taught "
    "at least twice.")
d.eq(r"A_h = \frac{1}{n_h} \sum_{i=1}^{n_h} \mathbb{1}\big[\hat{y}_i = y_i\big], \qquad \mathrm{switch} \iff \min(n_{\mathrm{ask}}, n_{\mathrm{state}}) \ge 30 \ \wedge\ A_{\mathrm{ask}} \ge \theta \ \wedge\ A_{\mathrm{state}} \ge \theta")
d.p("**Cost per turn.** With $R$ the rules, $Q_t$ the question, $A_t$ the answer, $K$ the readout, $J$ one tool call, "
    "$D_t = Q_{t-1} + A_{t-1}$ the delta that the last turn appended and $P_t = R + \\sum_{i<t}(Q_i + A_i)$ the "
    "transcript, the transcript path sends $P_t + Q_t$ without a cache, or $0.1\\,(P_t - D_t) + 1.25\\,D_t + Q_t$ "
    "token-equivalents under a cache that reads at a tenth of the input price and writes at 1.25 times it, while the CMLM "
    "path with two uncached round trips sends")
d.eq(r"I_{\mathrm{CMLM}}(t) = 2R + 2Q_t + K + J, \qquad I_{\mathrm{T}}(t) = P_t + Q_t")
d.p("**Algorithms.** The three procedures that the paper measures are given as pseudocode.")
d.code("""Algorithm 1  One turn of the loop, with no transcript
Input: question q, CMLM M, Life L, frontier F with tools recall and teach
 1. messages <- [q]                                  # no history
 2. loop:
 3.    r <- F(rules, tools, messages)
 4.    if r asks recall(query, k, key):
 5.        rows <- M.readout(query, k)
           if key: rows <- L.answer(key, rows)        # exact, ABSTAIN or blend
 6.        messages <- messages + [r, result(rows)]; continue
 7.    if r asks teach(pairs):
 8.        for (c, f, key) in pairs:
               M.teach(c, f); if key: L.learn(key, f)
 9.    break                                         # teach-only ends the turn
10. return the answer text of r; drop messages""")
d.code("""Algorithm 2  Cascade teach with hand-down
Input: level l with store F and cap C; level l+1, created when first needed
 1. F.teach(pairs)                          # 20 Adam steps: new pairs + replay
 2. while |F| > C:
 3.     f <- oldest fact in F; V <- the cue vectors taught with f
 4.     level(l+1).teach({(v, f) : v in V})     # vectors down, no words
 5.     F.forget(f)
 6. recall(query): v <- phi(query)
       return interleave_by_rank(level_0.top_k(v), level_1.top_k(v), ...)[:k]""")
d.code("""Algorithm 3  Tool-recovering model with a prequential gate
Input: heads H_ask, H_state (empty CMLMs over key parts), warm-up 30, threshold theta
 1. on a supervised turn with frontier calls:
 2.     for each keyed recall (q, key):
            score H_ask.predict(q) == key; H_ask.learn(q, key)
 3.     for each keyed teach (f, key):
            score H_state.predict(f) == key; H_state.learn(f, key)
 4. ready <- n_ask >= 30 and n_state >= 30 and A_ask >= theta and A_state >= theta
 5. if ready, on every later turn:
 6.     key <- the literal key in q, else H_ask.predict(q)
               if its share >= 0.6 and its label was seen >= 2 times
 7.     rows <- M.readout(q, k) with key          # a guessed key earns one slot
 8.     answer <- F(short rules, no tools, [context(rows), q])   # one round trip
 9.     for each sentence f in answer with a value that restates neither
        the context nor a held fact:
10.         M.teach(q, f); key' <- H_state.predict(f)
            if confident: L.learn(key', f)""")

d.heading("Guarantees", 2)
d.p("This subsection proves what holds for the mechanism; what the map recalls is measured in the Results and is not a "
    "theorem. Let $R$ be the tokens of the rules, $Q_t \\le Q_{\\max}$ the question of turn $t$, $A_t$ the answer, $d$ a "
    "lower bound on the answers of a deep-search session (a document per turn), $K \\le k\\,\\ell_{\\max}$ the readout for "
    "facts of at most $\\ell_{\\max}$ tokens, $J$ a bound on the tokens of one tool call, and $W$ the context window.")
d.p("**Theorem 1 (bounded input).** Under Algorithm 1 the input sent in turn $t$ satisfies, for every $t$,")
d.eq(r"I_{\mathrm{CMLM}}(t) \;\le\; 2R + 2Q_{\max} + K + J \;=:\; B, \qquad I_{\mathrm{T}}(t) \;=\; R + Q_t + \sum_{i<t}(Q_i + A_i) \;\ge\; R + (t-1)\,d,")
d.p("and $B$ does not depend on $t$.")
d.p("*Proof.* Algorithm 1 builds every request from the question alone (line 1). The only additions before the answer "
    "are one recall call and its result (line 6), so the second round trip carries the rules, the question, the call and "
    "at most $K$ tokens of readout, and no earlier turn is ever appended. Summing the two round trips gives $B$. The "
    "transcript path carries every earlier question and answer by definition, and each answer has at least $d$ tokens. ∎")
d.p("**Corollary 1 (the window).** The transcript path exceeds any window $W$ at the latest by turn")
d.eq(r"t^{*} = \Big\lfloor \frac{W - R}{d + Q_{\min}} \Big\rfloor + 1,")
d.p(f"whereas the CMLM path fits every window $W \\ge B$ for every $t$. With $W = 10^6$, $R = 164$, $d = 3300$ and "
    f"$Q_{{\\min}} = 10$, $t^{{*}} = {tstar}$; the 300-turn run below reached {n0(t10['flat']['final']['transcript_tokens'])} transcript tokens "
    f"at $t = 300$ while the CMLM path sent about {n0(round(t10['flat']['final']['sent_tokens'], -1))} tokens per turn throughout. This is the sense in "
    "which the design is proved for deep search: past the window the transcript path does not exist, and the CMLM path's "
    "request does not grow. ∎")
d.p("**Theorem 2 (the Life is exact and revision-correct).** With the write rule $N[\\kappa][v] \\leftarrow N[\\kappa][v] "
    "+ t_\\kappa + 1$, where $t_\\kappa$ is the total count under $\\kappa$ before the write, the value written last is the "
    "unique maximiser under $\\kappa$; an untaught key returns ABSTAIN; and no count ever decreases.")
d.p("*Proof.* Let $v^{*}$ be the value just written and $t_\\kappa$ the total before the write. Afterwards "
    "$N[\\kappa][v^{*}] \\ge t_\\kappa + 1 > t_\\kappa \\ge N[\\kappa][v]$ for every other $v$, because $t_\\kappa$ is the sum "
    "of all counts under $\\kappa$ and each is non-negative. So $\\mathrm{recall}(\\kappa) = v^{*}$ uniquely, whatever was "
    "written before and however often. A key that was never written has no entry, and recall returns ABSTAIN by "
    "construction. Counts are only ever incremented, so every earlier value remains in the table with its count. ∎")
d.p(f"*Corollary 2.* On any keyed probe of a taught key the Life's answer is the newest fact with probability 1, "
    "independent of the session's length, of the depth at which the fact's cue sits and of the state of the map. The "
    f"measured {n0(EX10['known'])} correct answers out of {n0(EX10['known'])}, and {n0(EX10['revised'])} out of {n0(EX10['revised'])} on revised keys, are "
    "this corollary observed.")
d.p("**Theorem 3 (full replay under the cap).** If a level holds at most $C$ pairs and the replay size $\\rho$ satisfies "
    "$\\rho \\ge C$, then every training step at that level optimises the objective over all of the level's pairs.")
d.p("*Proof.* The batch is the new pairs together with a sample of $\\min(\\rho, n_{\\mathrm{old}})$ old pairs. With "
    "$n_{\\mathrm{old}} \\le C \\le \\rho$ the sample is the whole set, so $B_t$ equals the level's entire pair set at every "
    "step. ∎")
d.p("*Proposition 3 (the flat model's replay gap).* A flat model holding $n > \\rho$ pairs includes a given old pair in a "
    "teach with probability $\\rho/(n - m)$, where $m$ is the number of new pairs, so the expected number of consecutive "
    "teaches whose objective excludes that pair is $(n - m)/\\rho - 1$, which grows linearly in $n$. Each such teach moves "
    "the towers without that pair's constraint. This is the mechanism by which the flat model's recall of old facts falls "
    "with the session while the capped levels' recall does not. ∎")
d.p("**Theorem 4 (the interleave guarantee).** With $L$ levels asked and a readout of size $k$, every needed fact ranked "
    "at a position $r < \\lfloor k/L \\rfloor$ within its own level appears in the readout, whatever its depth.")
d.p("*Proof.* Rank interleaving emits rank 0 of every level, then rank 1 of every level, and so on, and stops at $k$ "
    "facts. Ranks 0 to $\\lfloor k/L \\rfloor - 1$ of all $L$ levels amount to $L \\lfloor k/L \\rfloor \\le k$ facts, so all "
    "of them are emitted before the cut. ∎")
d.p("**Proposition 5 (the ceiling).** For probes on topics holding $n_i$ facts with $k$ asked for, "
    "$\\mathrm{recall@}k \\le \\sum_i \\min(k, n_i) / \\sum_i n_i$. *Proof.* A readout of $k$ facts contains at most "
    "$\\min(k, n_i)$ of the $n_i$ facts needed. ∎")
d.p("**Proposition 6 (break-even under caching).** With cache reads at price ratio $r$ and writes at $w$ relative to "
    "uncached input, the warm transcript path costs $r\\,(P_t - D_t) + w\\,D_t + Q_t$ per turn and the CMLM path costs $B$ "
    "uncached. The CMLM path is cheaper exactly when")
d.eq(r"P_t \;>\; \frac{B - Q_t - (w - r)\,D_t}{r}.")
d.p(f"*Proof.* Compare the two costs and solve for $P_t$. With $r = 0.1$, $w = 1.25$, $Q_t \\approx 15$ and $D_t \\approx 50$, "
    f"the threshold is about {n0(round(be(430), -2))} tokens of transcript for the small world ($B \\approx 430$) and about "
    f"{n0(round(be(560), -2))} for the document world ($B \\approx 560$). Below it the transcript path is cheaper inside one warm "
    "session; above it, and in every cold session, the CMLM path is cheaper. The threshold falls as $D_t$ grows: a turn "
    "that appends thousands of tokens must be written to the cache once at $w$, which alone exceeds $B$, so for sessions "
    "whose turns return documents the CMLM path is cheaper from the second turn on, with or without a cache. ∎")
d.p("**Proposition 7 (the gate is honest).** The prequential accuracy $A_h$ is a mean of indicators "
    "$\\mathbb{1}[\\hat{y}_i = y_i]$ in which each $\\hat{y}_i$ is a function of examples 1 to $i - 1$ only, so every term "
    "scores an unseen example; recall of the taught cues scores the training set itself and can be 1 while $A_h$ is far "
    "below it. *Proof.* By construction of Algorithm 3, lines 2 and 3: predict, then learn. ∎")
d.p("**Proposition 8 (the vector hop is exact and free).** Let a level hand a pair down as $(\\phi(c), f)$ rather than "
    "$(c, f)$. Then the level below reaches the same weights and the same readouts as it would from the words, and the "
    "hop carries no tokens and no decoding step.")
d.p(f"*Proof.* Teaching on a string computes $\\phi(c)$ deterministically before the first gradient step, so the batch, "
    "the loss and every update are identical whether $\\phi(c)$ or $c$ arrives, and the scores and readouts follow. A word "
    "hop would have to render the cue as text and encode it again, which between two models behind an API is a generated "
    "cue and a round trip; the vector hop is a copy of a vector that the level has already computed. Checked on 60 pairs "
    f"of world seed 1: weights {'bit-identical' if VH['weights_identical'] else 'NOT identical'}, {VH['readouts_identical']} of "
    f"{VH['probes']} readouts identical, maximum score difference {VH['max_score_diff']:.1f}, and {VH['word_hop_tokens']} tokens that the "
    "cues would have cost as words against none. ∎")
d.p("Proposition 8 does not say that the vector carries more than the words: $\\phi$ is a hash of the words, so it carries "
    "exactly what they carry and cannot be decoded back into them. The saving between CMLMs is the generation step and its "
    "tokens, not information. Any loss from putting a model's state into words arises only on the hop into the frontier, "
    "which this work still makes in words. Theorems 1 to 4 hold for any two-tower substrate, but whether the readout's top "
    "ranks contain the facts that a paraphrased question needs is a property of training and generalisation, which the "
    "experiments measure against a hidden truth.")

d.heading("Experimental design", 2)
d.p("**The synthetic world.** No third-party dataset is used. The world is generated from a seed and is byte-identical "
    "across runs (Table 3). It has ten topics, each with six fact words and six cue words, disjoint within a topic and "
    "across topics. A teaching turn asks about one topic in three of its cue words and is answered with one fact, stated "
    "in three of its fact words with a value (an amount, a schedule, a reference, a count or a deadline), between two "
    "filler sentences. In the paraphrase world a probe shares no word with the fact it needs; in the overlap world the cue "
    "words are the fact words. Each fact carries a key of the form topic.kind, so a key stated twice with a new value is a "
    "natural revision. With bulk, every answer is padded with 250 filler sentences to a document of about 3,300 tokens "
    "that holds the same single fact, which is the deep-search shape. The only preprocessing is the hashing of words into "
    "features described above.")
d.p("**The scripted frontier.** A scripted frontier stands in for the model: it calls recall with the question as the "
    "query, answers a teaching turn with the world's scripted text, quotes the readout back on a probe, and teaches the "
    "fact under its key. Probe scores therefore measure the CMLM's recall as the frontier would see it; the frontier's own "
    "judgement is not measured. This choice isolates the context model, which is the object of study, from the "
    "variability of a live model, and it is the main threat to external validity.")
d.p("**Probes, metrics and their justification.** Every five turns, every topic taught so far is probed in other cue "
    "words, and the truth is that topic's facts so far; with the Life, every key taught so far is probed exactly and three "
    "keys that were never taught are probed beside them. Recall@k is the share of the facts a probe needs that appear in "
    "the frontier's answer. It is the right metric because the frontier can use only what the readout returns, and it is "
    "read against the ceiling of Proposition 5, because with more facts per topic than k no readout can reach 1. Old-fact "
    "recall restricts recall to facts taught more than 15 turns before the probe and operationalises forgetting. For the "
    "Life, exact hit is the share of keyed probes on taught keys whose answer holds the newest fact, false abstain is the "
    "share of those that answered ABSTAIN, abstain is the share of probes on never-taught keys that answered ABSTAIN, and "
    "newest is exact hit restricted to keys stated more than once. Sent tokens count everything sent in a turn over all "
    "of its round trips.")
d.p("**Baselines.** Matching is the same model at step zero. The window baseline is the last 320 tokens of transcript. "
    "The full transcript recalls everything at its full token cost by construction and is reported as a token count.")
d.p("**Token accounting.** Token counts are characters divided by four, the estimate used throughout the code; the API's "
    "usage fields would be the truth and were not available. The transcript path is counted on every turn and never "
    "sent.")
d.p("**Runs and statistics.** Seeds 1 to 40 are used for the 30- and 60-turn comparisons with the untrained baseline, "
    "seeds 1 to 20 for the 60- to 120-turn comparisons of shapes and for the tool model, seeds 1 to 5 for the 300-turn "
    "document runs, and seeds 1 to 3 for the key head tested alone. Means over seeds are reported with standard deviations "
    "and t-based 95% confidence intervals, and the shapes are compared with the flat model by per-seed paired differences "
    "on the same seed, which removes the variation between worlds.")
d.p("**Computing infrastructure.** All runs used one Python process on an Apple M4 Pro (12 cores, 24 GB of memory) "
    "under macOS 26.5.2, with Python 3.14.7 and NumPy 2.4.4 and no GPU. Training and recall are real computations. Wall "
    "times are process wall-clock times measured while other runs shared the machine, so they are indicative only; every "
    "other number is deterministic for a given seed. The script that regenerates every table is supplied as "
    "Supplemental Code S1.")
d.p("**A human-driven frontier.** With no API credential available, Claude Fable 5.1 (Anthropic) in an interactive "
    "session wrote the frontier's recall queries and keyed teach pairs for 12 turns of world seed 1, then answered ten "
    "probes seeing only the readouts, scored against a truth that the answers never saw. The cue writing was not blind; "
    "the answering was. This is a sanity check, not a result.")
d.p("**A real transcript, costed.** The conversation in which this work was developed is real data of the deep-search "
    "shape. Only its per-turn sizes are used (Supplemental Data S1): the number of characters in each of the person's 34 "
    "questions and the number appended by each turn. No message text is used or released.")

# ------------------------------------------------------------ results
d.heading("Results", 1)
d.heading("Training beats matching where the words differ (RQ1)", 2)
p30 = T4P["final_recall"]; o30 = T4O["final_recall"]; l60 = T4L["final_recall"]
d.p(f"Over 40 seeds and 30 turns the trained CMLM recalled {f3(p30['trained'])} of the facts that probes needed at the last "
    f"turn, against {f3(p30['matching'])} for the same model at step zero and {f3(p30['window'])} for a 320-token window, and it beat "
    f"the untrained model on {T4P['trained_beats_matching']} of 40 seeds (Table 4). Where the probe's words overlap the fact's, "
    f"matching already reaches {f3(o30['matching'])} and training adds {f3(o30['trained'] - o30['matching'])} rather than "
    f"{f3(p30['trained'] - p30['matching'])}. The frontier was sent {n0(T4P['final']['sent_tokens'])} tokens per turn over two round trips "
    f"while the transcript path reached {n0(T4P['final']['transcript_tokens'])}, and training took {T4P['wall_ms_per_turn']:.0f} ms per turn.")

d.heading("Expansion holds recall where the flat model forgets (RQ2)", 2)
fl = [fin5[(t, 'flat')] for t, _ in G5]; tr = [fin5[(t, 'tree')] for t, _ in G5]; rc = [fin5[(t, 'recursive')] for t, _ in G5]
o120 = {s: t5k(120, 12, s)["by_age"] for s in SH}
d.p(f"As sessions doubled from 60 to 120 turns, the flat model's recall at the last turn fell from {f3(fl[0])} to "
    f"{f3(fl[-1])}, while the cascade stayed between {f3(min(rc))} and {f3(max(rc))} and the tree between {f3(min(tr))} and "
    f"{f3(max(tr))} (Table 5, Fig. 2). Old facts show the mechanism: at 120 turns the flat model recalled "
    f"{f3(o120['flat']['old']['trained'])} of facts taught more than 15 turns before the probe against "
    f"{f3(o120['flat']['recent']['trained'])} of recent ones, while the tree held old facts at {f3(o120['tree']['old']['trained'])} and "
    f"the cascade at {f3(o120['recursive']['old']['trained'])}. The tree's closed children were trained only on what they hold and then "
    "frozen; the cascade's top level keeps training, so its weights still carry facts it has handed down. Sent tokens per "
    f"turn at 120 turns were {n0(t5k(120, 12, 'flat')['final']['sent_tokens'])} to {n0(t5k(120, 12, 'recursive')['final']['sent_tokens'])} against a transcript "
    f"path of {n0(t5k(120, 12, 'flat')['final']['transcript_tokens'])}.")

d.heading("A million tokens deep (RQ5)", 2)
def rng(s): xs = share[s]; return f"{min(xs):.2f} and {max(xs):.2f}"
d.p(f"At 300 turns of 3,300-token documents the transcript path would send {n0(t10['flat']['final']['transcript_tokens'])} tokens per "
    f"question, beyond a million-token window by Corollary 1; the frontier was sent about {n0(round(t10['flat']['final']['sent_tokens'], -1))} "
    f"tokens per turn throughout (Table 6, Fig. 4). Read as a share of a ceiling that falls through the session, the flat "
    f"model slid from {f2(share['flat'][0])} at turn 50 to {f2(share['flat'][-1])} at turn 300, and by the end its recall of the "
    f"cues it had itself been taught had fallen to {f3(t10['flat']['self_recall_last'])}, against {f3(t10['tree']['self_recall_last'])} "
    f"for the tree and {f3(t10['recursive']['self_recall_last'])} for the cascade. Over the same checkpoints the tree stayed "
    f"between {rng('tree')} of its ceiling and the cascade between {rng('recursive')} (Fig. 3), because no level holds more than 30 facts "
    f"and every level replays all of them. A 320-token window holds none of these documents and recalled nothing. The Life "
    f"was exact on all {n0(EX10['known'])} keyed probes at every depth.")

d.heading("Statistical reliability", 2)
v9t, v9r = t9["tree"]["vs_flat"], t9["recursive"]["vs_flat"]
v10t, v10r = t10["tree"]["vs_flat"], t10["recursive"]["vs_flat"]
d.p(f"The central comparison, expansion against a single model, holds with the seeds separated (Table 7). Over 20 seeds at "
    f"60 turns with $k = 6$, the tree's recall at the last turn exceeded the flat model's by {f3(v9t['mean'])} (95% CI "
    f"{ci(v9t)}), winning on {v9t['wins']} of 20 seeds, and the cascade's by {f3(v9r['mean'])} ({ci(v9r)}), winning on "
    f"{v9r['wins']} of 20. Tokens sent per turn were {min(sent9):.0f} to {max(sent9):.0f} for the three shapes, so the gain is "
    f"not bought with a larger readout. At 300 turns, with five seeds, the tree beat the flat model by {f3(v10t['mean'])} "
    f"({ci(v10t)}), winning on {num(v10t['wins'])} of five seeds, and the cascade by {f3(v10r['mean'])} ({ci(v10r)}), winning on "
    f"{num(v10r['wins'])} of five (Table 8). Old-fact recall separates the shapes most sharply. Five seeds give wide intervals, and the "
    f"paired rows are the stronger evidence.")

d.heading("The exact memory changes nothing about the map (RQ3)", 2)
ex6 = {s: L6[s]["exact"] for s in SH}
d.p(f"With the Life attached, over 20 seeds and 60 turns with $k = 6$, every keyed probe was answered exactly, every "
    f"unknown key drew ABSTAIN, and every revised key returned its newest value, in the flat, tree and cascade forms alike "
    f"(Table 9). Paraphrase recall was {'identical to' if life_same else 'different from'} the runs without the Life, and "
    f"tokens sent differed by at most {sent_diff:.1f} per turn, the key field that a tool call gains. In the tree the "
    f"earlier value of a revised key often sits in a closed child whose weights never move again; the Life returned the "
    f"newest value on every one of the {n0(ex6['tree']['revised'])} revised-key probes.")

d.heading("The tool-recovering model: what it removes and what it costs (RQ4)", 2)
p_exact, t_exact = PERM["exact"], TOOLS["exact"]
lo30, hi30 = kh_range(kh_pts[0], "correct"); lo120, hi120 = kh_range(kh_pts[-1], "correct")
clo, chi = min(r["committed"] for r in kh_rows), max(r["committed"] for r in kh_rows)
d.p(f"Under a permissive gate that switched after 30 keyed examples, the tool model cut tokens per turn on turns 31 "
    f"to 60 from {n0(same_turns_tools)}, with the tools, to {n0(tm_p['sent_recovered'])}, and round trips from {tm_t['trips_tools']:.0f} to "
    f"{tm_p['trips_recovered']:.0f}, with paraphrase recall at {f3(PERM['per_seed']['final']['mean'])} against "
    f"{f3(TOOLS['per_seed']['final']['mean'])} with the tools (Table 10). It also filed facts under the wrong key: recovered "
    f"teach keys were correct {f3(tm_p['key_ok_recovered'])} of the time while the model committed to a key "
    f"{f3(tm_p['keyed_recovered'])} of the time, and the Life's exact recall fell from {f3(t_exact['hit'])} to "
    f"{f3(p_exact['hit'])}, with false abstentions at {f3(p_exact['false_abstain'])}. Tested alone on unseen facts, the "
    f"key head was right on {lo30} to {hi30} of 30 after {kh_pts[0]} examples and {lo120} to {hi120} of 30 after "
    f"{kh_pts[-1]}, while its softmax confidence committed on {clo} to {chi} of 30 at every accuracy (Fig. 5). Under the "
    f"prequential gate at 0.85, over 120 turns and 20 seeds, the model took over on {hon_sw} of 20 seeds; keys predicted "
    f"from facts ended at a prequential accuracy of {f3(HON['toolmodel']['state_prequential'])} and keys predicted from "
    f"questions at {f3(HON['toolmodel']['ask_prequential'])}."
    + (f" The prequential figure averages over every example since the first, so it lags the head's current accuracy, "
       f"which on unseen facts had reached {kh_acc[kh_pts[-1]]:.3f} after {kh_pts[-1]} examples." if kh_acc[kh_pts[-1]] >= 0.85 > HON['toolmodel']['state_prequential'] else ""))

d.heading("A human-driven frontier, 12 turns", 2)
d.p("With Claude writing the recall queries and keyed teach pairs and then answering ten probes from the readouts "
    "alone, 11 of the 12 facts that the probes needed came back in the readouts and all 11 were answered, with no wrong "
    "quotation and no wrong abstention; all 12 keys were exact on recall, and 418 tokens were sent per turn. The one miss "
    "was a topic with two facts of which the readout held one. Cues written by a model, richer than the scripted "
    "frontier's bare questions, gave a higher readout recall than the synthetic runs, which suggests that the frontier's "
    "cue quality is a lever that the synthetic world understates. This is one seed and a sanity check only.")

d.heading("A real transcript, costed", 2)
d.p(f"The real session ran to {n0(rs_total_tokens)} estimated tokens over 34 turns of the person, a mean of {n0(rs_mean)} tokens "
    f"appended per turn and one turn of {n0(rs_big)}. Applying the per-turn cost model to its actual sizes, with the rules at "
    f"164 tokens, a readout of four facts and one tool call, the transcript path would have sent {n0(rs_tot['cold'])} input "
    f"tokens without a cache, or {n0(rs_tot['warm'])} token-equivalents with a warm cache, and the CMLM path {n0(rs_tot['cmlm'])}, "
    f"between {min(rs_cmlm)} and {max(rs_cmlm)} per turn (Table 11, Fig. 6). The CMLM path is cheaper than the cached transcript on every "
    f"turn except turn {', '.join(str(x) for x in RS['warm_cheaper_than_cmlm_on_turns'])}, because every later turn begins with the "
    f"first turn's {n0(int(sizes[0][2]))} tokens of history to read and the previous turn's output to write. By Corollary 1 the "
    f"session would cross a million-token window at about turn {t_mean} at its mean pace, and at turn {t_big} at the pace of its "
    "largest turn. This applies the cost model to real sizes; it is not a live run, and it says nothing about what the map "
    "would have recalled from this transcript.")

# ------------------------------------------------------------ discussion
d.heading("Discussion", 1)
d.p(f"**Where the saving is.** Prompt caching decides it. Inside one warm session the transcript path reads its history at "
    f"a tenth of the input price, so at {n0(T4P['final']['transcript_tokens'])} tokens of transcript the CMLM path, with two uncached "
    f"round trips, costs more input: {n0(T4P['final']['sent_tokens'])} tokens against about {n0(warm_small)} token-equivalents. It wins "
    "past a few thousand tokens of transcript, across sessions, after any gap longer than the cache lifetime, past the "
    "window, where the transcript path does not exist, and, by Proposition 6, from the second turn of any session whose "
    "turns return documents. Output tokens are unchanged. The design is therefore for deep search, not for short chat "
    "under the window.")
d.p("**What theory does not give.** Three things are settled only by measurement. Forgetting: a single CMLM loses old "
    "facts as it fills, and bounding the store per model is what holds recall. Retrieval precision: a hashed feature space "
    "collides, and a fact can come back for the wrong question, which is why verbatim facts live in a store and the "
    "weights hold only the map. The teacher's choice: the memory can return only what was written into it, and what is "
    "worth writing has no theoretical answer.")
d.p(f"**A model cannot see its own forgetting.** A model's recall of the cues it was taught stayed high while its recall "
    f"of paraphrases of those cues fell, and the same blindness made the key head commit on {clo} to {chi} of 30 "
    "predictions at every accuracy. The self-measured triggers that were tried, closing a child on its own recall and "
    "handing a fact down when its own cue stopped returning it, did no better than a fixed cap, and the key head's own "
    "confidence was no guide to its accuracy. A held-out cue per fact, written "
    "by the frontier, would give the model a way to re-ask in other words.")
d.p("**Tree versus cascade.** The tree is higher because a closed child's weights are clean; the cascade needs no router, "
    "orders itself by age and speaks to its own context in vectors. The cascade's training cost per turn grows with depth, "
    f"since every hand-down trains the level below: at 300 turns a session took {t10['recursive']['wall_s_per_session']:.1f} s "
    f"against {t10['tree']['wall_s_per_session']:.1f} s for the tree and {t10['flat']['wall_s_per_session']:.1f} s for the flat model. "
    "Which is right depends on what a heavier substrate does with a frozen weight set.")
d.p("**The latent link.** Between two CMLMs nothing is tokenized. Into a frontier served by an API the last hop must be "
    "words. With an open-weights frontier the link could be real, and its pieces exist: a trained projector into the "
    "model's embedding space for the readout (Alayrac et al., 2022; Liu et al., 2023) and gist tokens for the fixed "
    "instructions (Mu, Li & Goodman, 2023). What remains to be measured is whether a projected vector carries a fact as "
    "faithfully as its words.")
d.p(f"**The honest gate.** The tool-recovering model shows both edges of the design. The overhead it targets is real, "
    f"{cut:.0%} of tokens per turn and a round trip, and so is the cost of taking it over early, measured as misfiled keys. "
    "Under a gate that predicts before it learns, the model in this world was not ready within 120 turns. The result is "
    "not that the overhead cannot be removed but that it cannot be removed safely by a hashed two-tower model with about "
    "a hundred keyed examples, and the record states the accuracy at which it would be safe to try.")
d.heading("Limitations", 2)
d.p("**Internal validity.** The frontier is scripted: it asks with the question's words, quotes the readout back and "
    "teaches the fact under its true key, so the frontier's own judgement, cue quality, re-asking and abstention are not "
    "measured. The 12-turn human-driven run had its cues written by the same session that had seen the facts; only its "
    "answering was blind.")
d.p("**External validity.** The world is synthetic: one fact per answer, one value per fact, disjoint vocabularies and "
    "filler without facts. Real answers state several facts in varied words, and a pattern teacher that keeps sentences "
    "with numbers would miss most of them. One substrate, a hashed two-tower linear model, was tested, and a heavier model "
    "may move every number in either direction. The live path with a real frontier through an API has not been run.")
d.p("**Construct validity.** Recall@k is read against a ceiling that depends on k and on facts per topic, so both raw and "
    "ceiling-normalised values are given. Forgetting is operationalised as recall of facts taught more than 15 turns "
    "earlier. Token counts are character-based estimates; the API's usage fields are the truth, and the cost analysis "
    "inherits the estimate's error, including whether the rules-and-tools prefix reaches a provider's minimum cacheable "
    "length.")
d.p("**Conclusion validity.** Five seeds at 300 turns, twenty at 60 to 120 turns, forty at 30 and 60 turns and three for "
    "the key head's curve. Means are given with standard deviations, 95% confidence intervals and per-seed paired "
    "comparisons for the central claims.")

d.heading("Conclusions", 1)
d.p("A frontier model can keep no transcript and still answer, if its context is a small model that it trains and "
    "queries rather than text that it re-sends. The limit on what such a context holds is training, not tokens, and the "
    "answer to that limit is expansion rather than a larger window. In a synthetic world that model held what the frontier "
    "taught it, forgot as it filled, and stopped forgetting once it expanded into a tree or a cascade of bounded models "
    "that talk to each other in vectors. An exact integer memory beside every level made keyed facts verbatim, revisable "
    "and refusable at any depth without touching the map. At the scale the design is for, three hundred documents and a "
    "million tokens of transcript that no request could carry, the frontier read about 560 tokens a turn and the "
    "expanding shapes kept most of the recall that their readout size allowed. The overhead of the tools themselves can "
    "be learned away, but not safely by this substrate on about a hundred examples. The next step is the one this paper "
    "could not take: the same loop with a real frontier behind the tools, its usage fields as the token counts, and its "
    "own cues and keys as the teacher.")

d.heading("Acknowledgements", 1)
d.p("**Use of generative AI.** The prototype code, the experiments, the figures and drafts of this manuscript were produced "
    "with Claude (Anthropic), using the models Claude Fable 5.1 and Claude Opus 5.5 through Claude Code version 2.1.92, "
    "working under the author's direction. In the human-driven frontier experiment, Claude Fable 5.1 acted as the frontier "
    "model. The tools were used to write and run code, to draft and revise text, and to check references against arXiv "
    "and Crossref, because the author developed the work in conversation with the model. The author conceived the method, "
    "directed every experiment, reviewed all content, confirmed its originality and accuracy, checked the terms of use of "
    "the tools, and takes full responsibility for the integrity of the whole manuscript, including the references.")

d.heading("References", 1)
def sentence_case(title):
    out, cap_next = [], True
    for w in re.split(r"(\s+)", title):
        if not w.strip():
            out.append(w); continue
        parts = w.split("-"); np_ = []
        c0 = re.sub(r"[^A-Za-z]", "", parts[0])
        if len(parts) > 1 and len(c0) > 1 and (c0.isupper() or any(ch.isupper() for ch in c0[1:])):
            out.append(w); cap_next = False; continue
        for pi, p in enumerate(parts):
            core = re.sub(r"[^A-Za-z]", "", p)
            keep = (len(core) > 1 and (core.isupper() or any(ch.isupper() for ch in core[1:])))
            if (cap_next and pi == 0) or keep:
                np_.append(p)
            else:
                np_.append(p.lower())
        out.append("-".join(np_))
        cap_next = w.endswith("?")
    return "".join(out)
for key, ref in sorted(REFS, key=lambda kr: kr[1].lower()):
    m = re.match(r"^(.*?\. \d{4}[ab]?\. )(.*?)(\. (?:In: |arXiv preprint|Neural|Psychology|Journal|Available).*)$", ref)
    if m and "arxiv.org" in ref and "DOI" not in ref:
        ref = m.group(1) + sentence_case(m.group(2)) + m.group(3)
    d.p(ref, ppr='<w:ind w:left="360" w:hanging="360"/>')
d.save(os.path.join(OUT, "Manuscript.docx"))

# ------------------------------------------------------------ citation audit
text = " ".join(d.body)
used = set()
for form, key in CITES.items():
    parts = form.rsplit(", ", 1)
    name, year = parts[0].replace("&", "&amp;"), parts[1]
    pattern = re.escape(name) + r"(?:, | \()" + re.escape(year)
    if re.search(pattern, text):
        used.add(key)
unused = [k for k, _ in REFS if k not in used]
print("references not cited in the text:", unused or "none")

# ------------------------------------------------------------ abstract length
abs_chars = sum(len(h) + 2 + len(t) for h, t in ABSTRACT) + 3
print("abstract characters:", abs_chars, "(limit 3,000)")
print("title characters:", len(TITLE), "(limit 250)")
json.dump({"abstract_chars": abs_chars, "unused_refs": unused}, open(os.path.join(OUT, "_audit.json"), "w"))
