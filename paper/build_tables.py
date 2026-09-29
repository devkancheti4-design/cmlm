"""Tables 1-10 as separate .docx files, plus the titles and legends for figures and tables."""
import json, os
from docxgen import Doc
import build_manuscript as B      # reuses the loaded results and derived numbers

OUT = B.OUT
t4, t5, t6, t7, t9, t10 = B.t4, B.t5, B.t6, B.t7, B.t9, B.t10
f3, n0, ms, ci = B.f3, B.n0, B.ms, B.ci
SH, NM = B.SH, {"flat": "Flat", "tree": "Tree", "recursive": "Cascade"}

TABLES = []   # (number, title, legend, rows, widths, landscape, notes)

TABLES.append((1, "Prior approaches to context beyond the window, on the properties this paper measures.",
    "Re-sent per turn: what the request carries on every turn. Frontier weights needed: whether the approach needs access to, or training of, the serving model's parameters.",
    [["Approach", "Where memory lives", "Re-sent per turn", "Past the window", "Exact recall", "Revision", "Paraphrase", "Frontier weights needed"],
     ["Transcript (baseline)", "the request", "all history", "no", "yes, within the window", "restate", "yes", "no"],
     ["Sliding window", "the request", "last W tokens", "loses the rest", "within W only", "restate", "yes", "no"],
     ["Summarisation (Nijkamp et al., 2026)", "summaries", "summaries", "yes", "lossy", "lossy", "yes", "no"],
     ["Latent compression (Chevalier et al., 2023; Liu & Qiu, 2025; Li et al., 2026)", "latent vectors", "latents", "yes", "approximate", "retrain", "yes", "yes, a trained decoder"],
     ["Model-managed text tiers (Packer et al., 2023; Sun & Zeng, 2025; Yu et al., 2026)", "text stores and working context", "working context and retrieved text", "yes", "when retrieved", "overwrite", "via retrieval", "no"],
     ["Memory in weights (Hardt & Sun, 2023; Behrouz, Zhong & Mirrokni, 2024)", "the model's parameters", "nothing", "yes", "unreliable (V P, 2026)", "none", "yes", "yes"],
     ["Exact store (Kancheti, 2026)", "integer table beside the model", "readout", "yes", "yes", "yes", "no", "no"],
     ["This work", "a trained model per level and an integer table", "rules and readout, 430 to 560 tokens", "yes", "yes, the Life", "yes, the Life", "yes, the map", "no"]],
    [16, 13, 13, 9, 11, 9, 9, 11], True, []))

TABLES.append((2, "Notation.", "",
    [["Symbol", "Meaning"],
     ["$q_t$, $a_t$", "the question and the answer of turn $t$"],
     ["$(c, f)$", "a cue and a fact: one teach pair"],
     ["$F_t$, $F^{(\\ell)}$", "the fact store after turn $t$; the store of level $\\ell$"],
     ["$\\phi(x)$, $D$", "the hashed feature vector of a string; its width, 4,096"],
     ["$W_q$, $W_f$, $W_0$, $m$", "the query and fact towers, their shared initial matrix, and the shared width, 64"],
     ["$s(q, f)$, $\\tau$", "the score of a fact for a query; the softmax temperature, 0.1"],
     ["$B_t$, $N_t$, $R_t$, $C_t$", "the training batch, its new pairs, its replay sample and the candidate facts"],
     ["$k$, $n_i$", "facts asked for; facts that topic $i$ holds"],
     ["$C$", "facts a level or child holds before it hands down or closes, 30"],
     ["$\\kappa$, $N$, $t_\\kappa$, $w_\\kappa$, $C_0$", "a key, the Life's count table, the key's total count, its gate weight, and the gate constant 0.25"],
     ["$A_h$, $\\theta$", "a head's prequential accuracy; the switch threshold, 0.85"],
     ["$R$, $Q_t$, $A_t$, $K$, $J$, $P_t$, $D_t$", "tokens of the rules, the question, the answer, the readout, one tool call, the transcript so far and the last turn's delta"],
     ["$W$, $d$, $B$", "the context window; a lower bound on answer length; the bound on the CMLM path's input per turn"]],
    [30, 70], False, []))

TABLES.append((3, "The synthetic world and the settings used in every run unless a table says otherwise.", "",
    [["Setting", "Value"],
     ["Topics; cue and fact words per topic; value kinds", "10; 6 and 6, all disjoint; 5"],
     ["Facts per turn; filler sentences per answer; bulk filler", "1; 2; 250 sentences, about 3,300 tokens"],
     ["Possible keys (topic.kind)", "50"],
     ["Probes", "every 5 turns, every topic taught so far, plus 3 never-taught keys with the Life"],
     ["Feature width D; shared width m; temperature τ", "4,096; 64; 0.1"],
     ["Optimiser; learning rate; steps per teach; replay; extra negatives", "Adam (0.9, 0.999); 0.05; 20; 48; 32"],
     ["Readout k", "4 at 30 turns; equal to facts per topic at 60 to 120 turns; 12 at 300 turns"],
     ["Cap C per level or child; tree self-recall trigger", "30; off"],
     ["Life gate constant C0", "0.25"],
     ["Tool model: warm-up; gate θ; part acceptance; minimum label count", "30 keyed examples; 0.85 prequential; 0.6 softmax share; 2"],
     ["Seeds", "40 (30 and 60 turns, Table 4); 20 (60 to 120 turns, tool model, ablations); 5 (300 turns); 3 (key head alone); 8, seeds 41 to 48 (hyperparameter grid)"],
     ["Hardware and software", "Apple M4 Pro, 12 cores, 24 GB; macOS 26.5.2; Python 3.14.7; NumPy 2.4.4; no GPU"]],
    [45, 55], False, []))

P, O, L = B.T4P, B.T4O, B.T4L
def fr(x, k): return f3(x["final_recall"][k])
TABLES.append((4, "Short sessions: the trained CMLM against the same model at step zero and a token window, 40 seeds.",
    "Recall is pooled over the probes of the last turn. Old facts were taught more than 15 turns before the probe; the old and recent values pool all probes of the session. Token counts are characters divided by four.",
    [["Measure", "30 turns, k = 4, paraphrase", "30 turns, k = 4, overlap", "60 turns, k = 4, paraphrase"],
     ["Trained CMLM, recall at the last turn (fraction)", fr(P, "trained"), fr(O, "trained"), fr(L, "trained")],
     ["Matching, step zero (fraction)", fr(P, "matching"), fr(O, "matching"), fr(L, "matching")],
     ["Last 320 tokens of transcript (fraction)", fr(P, "window"), fr(O, "window"), fr(L, "window")],
     ["Ceiling at the last turn (fraction)", fr(P, "ceiling"), fr(O, "ceiling"), fr(L, "ceiling")],
     ["Seeds on which training beat matching (count of 40)", str(P["trained_beats_matching"]), str(O["trained_beats_matching"]), str(L["trained_beats_matching"])],
     ["Old facts, trained (fraction)", f3(P["by_age"]["old"]["trained"]), f3(O["by_age"]["old"]["trained"]), f3(L["by_age"]["old"]["trained"])],
     ["Recent facts, trained (fraction)", f3(P["by_age"]["recent"]["trained"]), f3(O["by_age"]["recent"]["trained"]), f3(L["by_age"]["recent"]["trained"])],
     ["Transcript path at the last turn (tokens)", n0(P["final"]["transcript_tokens"]), n0(O["final"]["transcript_tokens"]), n0(L["final"]["transcript_tokens"])],
     ["Sent to the frontier per turn (tokens)", n0(P["final"]["sent_tokens"]), n0(O["final"]["sent_tokens"]), n0(L["final"]["sent_tokens"])],
     ["Training time (ms per turn)", f"{P['wall_ms_per_turn']:.0f}", f"{O['wall_ms_per_turn']:.0f}", f"{L['wall_ms_per_turn']:.0f}"]],
    [40, 20, 20, 20], False, []))

rows5 = [["Session", "Shape", "Recall at the last turn, mean (SD) (fraction)", "95% CI (fraction)", "Old facts (fraction)", "Recent facts (fraction)", "Sent per turn (tokens)", "Training (ms per turn)"]]
for turns, k in B.G5:
    for s in SH:
        x = B.t5k(turns, k, s); st = x["per_seed"]["final"]
        rows5.append([f"{turns} turns, k = {k}", NM[s], ms(st), ci(st), f3(x["by_age"]["old"]["trained"]), f3(x["by_age"]["recent"]["trained"]),
                      n0(x["final"]["sent_tokens"]), f"{x['wall_ms_per_turn']:.0f}"])
TABLES.append((5, "Recall as sessions grow: the flat model against the tree and the cascade, 20 seeds.",
    "k equals the facts per topic at the last turn, so the ceiling is 1. Old facts were taught more than 15 turns before the probe; old and recent values pool all probes of the session.",
    rows5, [14, 10, 18, 14, 10, 10, 11, 13], True, []))

def col10(s):
    x = t10[s]; ck = {c["turn"]: c["share"] for c in x["seed1_checkpoints"]}
    ex = x["exact"]; extra = (f"{x['tree']['children_mean']:.0f} children" if s == "tree" else
                              f"{x['cascade']['levels_mean']:.0f} levels" if s == "recursive" else "1 model")
    return [ms(x["per_seed"]["final"]), f"{ck[50]:.2f}, {ck[150]:.2f}, {ck[300]:.2f}", ms(x["per_seed"]["old"]), f3(x["self_recall_last"]),
            f"{ex['hit']:.3f} of {n0(ex['known'])}", f"{ex['abstain']:.3f} of {n0(ex['unknown'])}", f"{ex['newest']:.3f} of {n0(ex['revised'])}",
            n0(x["final"]["sent_tokens"]), n0(x["final"]["transcript_tokens"]), extra, f"{x['wall_s_per_session']:.1f}"]
c10 = {s: col10(s) for s in SH}
labels10 = ["Recall@12 at the last turn, mean (SD) (fraction)", "Share of the ceiling at turns 50, 150 and 300, seed 1 (fraction)",
            "Old facts, mean (SD) (fraction)", "Recall of its own taught cues at the last turn (fraction)", "Life: exact on keyed probes (fraction of probes)",
            "Life: ABSTAIN on unknown keys (fraction of probes)", "Life: newest value on revised keys (fraction of probes)", "Sent per turn (tokens)",
            "Transcript path at turn 300 (tokens)", "Models at the end (count)", "Training per session (s)"]
TABLES.append((6, "A million tokens deep: 300 turns of 3,300-token documents, five seeds, each shape with the Life.",
    "The ceiling of recall@12 at the last turn is 0.400. Token counts are characters divided by four; the transcript path is counted, never sent.",
    [["Measure", "Flat + Life", "Tree + Life", "Cascade + Life"]] + [[labels10[i], c10["flat"][i], c10["tree"][i], c10["recursive"][i]] for i in range(len(labels10))],
    [40, 20, 20, 20], False, []))

def stat_rows(T):
    rows = [["Shape", "Recall, mean (SD) (fraction)", "95% CI (fraction)", "Old facts, mean (SD) (fraction)", "95% CI (fraction)",
             "Gain over flat, mean [95% CI] (fraction)", "Seeds won (count)", "Sent per turn (tokens)"]]
    for s in SH:
        ps = T[s]["per_seed"]; v = T[s].get("vs_flat")
        rows.append([NM[s], ms(ps["final"]), ci(ps["final"]), ms(ps["old"]), ci(ps["old"]),
                     (f"+{v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]" if v else ""),
                     (f"{v['wins']} of {v['of']}" if v else ""), f"{ps['sent']['mean']:.0f}"])
    return rows
TABLES.append((7, "Recall at the last turn and old-fact recall at 60 turns, k = 6, 20 seeds: mean, standard deviation, 95% confidence interval, and the paired difference against the flat model.",
    "Intervals are t-based on per-seed values; the gains are per-seed differences against the flat model on the same seed.", stat_rows(t9), [10, 14, 13, 14, 13, 16, 9, 11], True, []))
TABLES.append((8, "The same statistics for the 300-turn document run, k = 12, five seeds.",
    "The ceiling of recall@12 at the last turn is 0.400. Intervals are t-based on per-seed values; the gains are per-seed differences against the flat model on the same seed.",
    stat_rows(t10), [10, 14, 13, 14, 13, 16, 9, 11], True, []))

def col9(s):
    x = B.L6[s]; ex = x["exact"]
    return [f"{ex['hit']:.3f} of {n0(ex['known'])}", f"{ex['false_abstain']:.3f}", f"{ex['abstain']:.3f} of {n0(ex['unknown'])}",
            f"{ex['newest']:.3f} of {n0(ex['revised'])}", f"{x['per_seed']['final']['mean']:.3f} and {B.M9[s]['final']['mean']:.3f}",
            f"{x['per_seed']['sent']['mean']:.0f} and {B.M9[s]['sent']['mean']:.0f}"]
c9 = {s: col9(s) for s in SH}
labels9 = ["Exact by key (fraction of keyed probes)", "False ABSTAIN on a known key (fraction)", "ABSTAIN on unknown keys (fraction of probes)",
           "Newest value on revised keys (fraction of probes)", "Paraphrase recall with and without the Life (fraction)", "Sent per turn with and without the Life (tokens)"]
TABLES.append((9, "The Life beside each shape: exactness, abstention and revision, with the map's paraphrase recall, 20 seeds, 60 turns, k = 6.", "",
    [["Measure", "Flat + Life", "Tree + Life", "Cascade + Life"]] + [[labels9[i], c9["flat"][i], c9["tree"][i], c9["recursive"][i]] for i in range(len(labels9))],
    [40, 20, 20, 20], False, []))

tt, tp = B.tm_t, B.tm_p
TABLES.append((10, "The tool-recovering model: what a permissive gate removes and what it misfiles, 20 seeds, 60 turns, k = 6; and the prequential gate over 120 turns.",
    "Tools every turn: the same runs with the tool model observing but never switching. Permissive gate: the switch threshold set to 0, so the model takes over after 30 keyed examples. Sent tokens compare the same turns, 31 to 60: turns 1 to 30 are identical in both runs, so the tools-every-turn mean over turns 31 to 60 follows from its mean over all 60 turns.",
    [["Measure", "Tools every turn", "Recovered, permissive gate"],
     ["Sent to the frontier per turn, turns 31 to 60 (tokens)", n0(B.same_turns_tools), n0(tp["sent_recovered"])],
     ["Round trips per turn (count)", f"{tt['trips_tools']:.0f}", f"{tp['trips_recovered']:.0f}"],
     ["Paraphrase recall at the last turn, mean of seeds (fraction)", f3(B.TOOLS["per_seed"]["final"]["mean"]), f3(B.PERM["per_seed"]["final"]["mean"])],
     ["Recovered teach keys correct (fraction)", "", f3(tp["key_ok_recovered"])],
     ["Recovered teaches committed to a key (fraction)", "", f3(tp["keyed_recovered"])],
     ["Life exact by key (fraction)", f3(B.TOOLS["exact"]["hit"]), f3(B.PERM["exact"]["hit"])],
     ["False ABSTAIN on known keys (fraction)", f3(B.TOOLS["exact"]["false_abstain"]), f3(B.PERM["exact"]["false_abstain"])],
     ["Newest value on revised keys (fraction)", f3(B.TOOLS["exact"]["newest"]), f3(B.PERM["exact"]["newest"])],
     ["Prequential gate 0.85, 120 turns: seeds on which it took over (count of 20)", "", str(B.hon_sw)],
     ["Prequential gate: final accuracy, keys from facts and from questions (fraction)", "", f"{f3(B.HON['toolmodel']['state_prequential'])} and {f3(B.HON['toolmodel']['ask_prequential'])}"]],
    [50, 25, 25], False, []))

rs = B.RS
TABLES.append((11, "A real 34-turn session, costed with the per-turn model of the paper.",
    "Sizes are those of the conversation in which this work was developed; no message text is used. Token counts are characters divided by four. Warm cache: history read at 0.1 and the last turn written at 1.25 times the input price.",
    [["Measure", "Value"],
     ["Turns of the person (count)", "34"],
     ["Transcript at the end (tokens)", n0(B.rs_total_tokens)],
     ["Mean appended per turn; largest single turn (tokens)", f"{n0(B.rs_mean)}; {n0(B.rs_big)}"],
     ["Input over the session, transcript without a cache (tokens)", n0(rs["totals"]["cold"])],
     ["Input over the session, transcript with a warm cache (token-equivalents)", n0(rs["totals"]["warm"])],
     ["Input over the session, CMLM path (tokens)", n0(rs["totals"]["cmlm"])],
     ["CMLM path per turn, lowest to highest (tokens)", f"{min(B.rs_cmlm)} to {max(B.rs_cmlm)}"],
     ["Turns on which the warm transcript was cheaper than the CMLM path", ", ".join(str(x) for x in rs["warm_cheaper_than_cmlm_on_turns"]) or "none"],
     ["Turn at which a one-million-token window would be crossed, at the mean pace and at the largest turn's pace", f"{B.t_mean} and {B.t_big}"]],
    [62, 38], False, []))

for num, title, legend, rows, widths, land, notes in TABLES:
    d = Doc(title=f"Table {num}", author="Devieswar Kancheti", line_numbers=False, landscape=land, line=240)
    d.p(f"**Table {num}.** {title}")
    d.table(rows, widths=widths, size=18 if land else 20)
    if legend:
        d.p(legend, "TableText", ppr='<w:spacing w:before="80"/>')
    d.save(os.path.join(OUT, f"Table {num}.docx"))
print("tables:", len(TABLES))

FIGS = [
    (1, "Architecture of the system.", "Words pass only between the person and the frontier model. The frontier keeps no transcript and reaches its context through two tools, recall and teach. CMLM levels hand their oldest facts down as feature vectors, at no token cost, and ranked facts climb back. An exact-key integer memory (the Life) sits beside every level. A tool-recovering model can make the frontier's calls once a prequential gate allows it."),
    (2, "Recall at the last turn as sessions grow, by shape.", "Bars are means over 20 seeds and whiskers are 95% confidence intervals over seeds. k equals the facts per topic at the last turn, so the ceiling is 1. Hatching distinguishes the shapes without colour."),
    (3, "Recall as a share of its ceiling through a 300-turn session.", "Seed 1 of the run with a 3,300-token document per turn and k = 12; markers are the probe checkpoints every 50 turns. The ceiling falls as facts per topic pass k."),
    (4, "Input tokens per question through the 300-turn session.", "Log scale. The transcript path is counted on every turn and never sent; the CMLM path is what the frontier was actually sent. Cascade, seed 1; token counts are characters divided by four. The dashed line marks a one-million-token context window."),
    (5, "Accuracy and confidence of the key head on unseen facts.", "The key head was taught n keyed examples and tested on the next 30 facts of the same world, for three seeds. Lines are means and dots single seeds. Committed counts predictions whose softmax share was at least 0.6; the dashed line is the prequential gate at 0.85."),
    (6, "Cumulative input tokens of a real 34-turn session under three paths.", "The paper's cost model applied to the per-turn sizes of the conversation in which this work was developed. Log scale; token counts are characters divided by four. No message text is used."),
]
with open(os.path.join(OUT, "Figure and table titles and legends.txt"), "w") as fh:
    for n, t, l in FIGS:
        fh.write(f"Figure {n}\nTitle: {t}\nLegend: {l}\n\n")
    for num, title, legend, *_ in TABLES:
        fh.write(f"Table {num}\nTitle: {title}\n" + (f"Legend: {legend}\n" if legend else "") + "\n")
print("legends written")
