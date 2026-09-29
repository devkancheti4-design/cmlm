"""A small LaTeX-subset to Office Math (OMML) converter, enough for the paper's equations."""
from xml.sax.saxutils import escape

SYM = {
    r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥", r"\in": "∈", r"\notin": "∉", r"\cup": "∪", r"\cap": "∩",
    r"\cdot": "·", r"\times": "×", r"\top": "⊤", r"\to": "→", r"\rightarrow": "→", r"\leftarrow": "←",
    r"\iff": "⟺", r"\wedge": "∧", r"\vee": "∨", r"\neq": "≠", r"\approx": "≈", r"\infty": "∞", r"\dots": "…",
    r"\ldots": "…", r"\cdots": "⋯", r"\pm": "±", r"\mid": "∣", r"\lVert": "‖", r"\rVert": "‖", r"\|": "‖",
    r"\lfloor": "⌊", r"\rfloor": "⌋", r"\lceil": "⌈", r"\rceil": "⌉", r"\{": "{", r"\}": "}", r"\ell": "ℓ",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\epsilon": "ϵ", r"\theta": "θ", r"\kappa": "κ",
    r"\lambda": "λ", r"\mu": "μ", r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ", r"\phi": "φ",
    r"\varphi": "φ", r"\omega": "ω", r"\Delta": "Δ", r"\Sigma": "Σ", r"\Phi": "Φ", r"\partial": "∂",
    r"\forall": "∀", r"\exists": "∃", r"\emptyset": "∅", r"\sim": "∼", r"\propto": "∝", r"\ast": "∗",
    r"\coloneqq": "≔", r"\langle": "⟨", r"\rangle": "⟩", r"\qed": "∎", r"\%": "%", r"\#": "#", r"\_": "_",
}
SPACE = {r"\,": " ", r"\;": " ", r"\:": " ", r"\ ": " ", r"\quad": " ", r"\qquad": "  ", r"\!": ""}
FUNCS = {r"\log", r"\exp", r"\min", r"\max", r"\sin", r"\cos", r"\arg", r"\det", r"\lim", r"\sup", r"\inf"}
NARY = {r"\sum": "∑", r"\prod": "∏", r"\int": "∫", r"\bigcup": "⋃"}
SIZED = {r"\big", r"\Big", r"\bigg", r"\Bigg", r"\left", r"\right", r"\bigl", r"\bigr", r"\Bigl", r"\Bigr"}
OPEN = {"(": ")", "[": "]", r"\{": r"\}", r"\lfloor": r"\rfloor", r"\lceil": r"\rceil", r"\lVert": r"\rVert", "|": "|", r"\langle": r"\rangle"}
STOP = {"=", "<", ">", "≤", "≥", ",", "  ", " ", "⟺", "≔", ";"}
RPR = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>'
MODE = {"inline": False}


def tokenize(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            j = i + 1
            if j < len(s) and s[j].isalpha():
                while j < len(s) and s[j].isalpha():
                    j += 1
                if j < len(s) and s[j] == "*" and s[i:j] == r"\operatorname":
                    j += 1
            else:
                j += 1
            out.append(s[i:j]); i = j
        elif c.isdigit():
            j = i
            while j < len(s) and (s[j].isdigit() or (s[j] == "." and j + 1 < len(s) and s[j + 1].isdigit())):
                j += 1
            out.append(s[i:j]); i = j
        elif c.isspace():
            i += 1
        else:
            out.append(c); i += 1
    return out


def run(text, plain=False):
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ""
    return f'<m:r>{sty}{RPR}<m:t xml:space="preserve">{escape(text)}</m:t></m:r>'


class P:
    def __init__(self, s):
        self.t = tokenize(s); self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self):
        tok = self.t[self.i]; self.i += 1; return tok

    def raw_group(self):
        """The literal text of a {...} group, for \\mathrm and \\text."""
        assert self.take() == "{"
        depth, parts = 1, []
        while True:
            tok = self.take()
            if tok == "{":
                depth += 1
            elif tok == "}":
                depth -= 1
                if depth == 0:
                    break
            parts.append(tok)
        out = []
        for p in parts:
            if p in SPACE: out.append(SPACE[p] or "")
            elif p in SYM: out.append(SYM[p])
            elif p.startswith("\\"): out.append(p[1:])
            else: out.append(p)
        return "".join(out)

    def group(self):
        if self.peek() == "{":
            self.take(); xs = self.seq(stop="}"); self.take(); return xs
        return [self.item()]

    def seq(self, stop=None):
        items = []
        while self.peek() is not None and self.peek() != stop:
            if stop is None and self.peek() == "}":
                break
            items.append(self.item())
        return self.bind_nary(items)

    def bind_nary(self, items):
        out, i = [], 0
        while i < len(items):
            it = items[i]
            if isinstance(it, dict) and it.get("nary") and it.get("e") is None:
                j = i + 1; e = []
                while j < len(items) and not (isinstance(items[j], dict) and items[j].get("text") in STOP):
                    e.append(items[j]); j += 1
                it = dict(it, e=e); out.append(it); i = j
            else:
                out.append(it); i += 1
        return out

    def scripts(self, base):
        sub = sup = None
        while self.peek() in ("_", "^"):
            tok = self.take()
            if tok == "_": sub = self.group()
            else: sup = self.group()
        if base.get("nary"):
            base = dict(base, sub=sub, sup=sup); return base
        if base.get("limlow") is not None and sub is not None:
            return dict(base, lim=sub)
        if sub is None and sup is None:
            return base
        return {"script": base, "sub": sub, "sup": sup}

    def item(self):
        tok = self.take()
        if tok == "{":
            xs = self.seq(stop="}"); self.take(); base = {"group": xs}
        elif tok == r"\frac":
            base = {"frac": (self.group(), self.group())}
        elif tok in (r"\mathrm", r"\text", r"\operatorname", r"\textrm", r"\mathit"):
            base = {"text": self.raw_group(), "plain": tok != r"\mathit"}
        elif tok == r"\operatorname*":
            base = {"limlow": self.raw_group()}
        elif tok == r"\mathbb":
            g = self.raw_group(); base = {"text": {"1": "𝟙", "R": "ℝ", "N": "ℕ", "E": "𝔼"}.get(g, g), "plain": True}
        elif tok == r"\mathcal":
            g = self.raw_group(); base = {"text": {"L": "ℒ", "D": "𝒟", "F": "ℱ"}.get(g, g), "plain": True}
        elif tok in (r"\hat", r"\tilde", r"\bar", r"\vec"):
            base = {"acc": {r"\hat": "̂", r"\tilde": "̃", r"\bar": "̅", r"\vec": "⃗"}[tok], "e": self.group()}
        elif tok in NARY:
            base = {"nary": NARY[tok], "e": None}
        elif tok in FUNCS:
            base = {"text": tok[1:], "plain": True}
        elif tok in SIZED:
            opener = self.take()
            if tok in (r"\right", r"\bigr", r"\Bigr"):
                base = {"text": SYM.get(opener, opener)}          # a lone closer
            else:
                closer = OPEN.get(opener, opener)
                inner = []
                while self.peek() is not None:
                    if self.peek() in SIZED and self.i + 1 < len(self.t) and self.t[self.i + 1] == closer:
                        self.take(); self.take(); break
                    inner.append(self.item())
                base = {"delim": (SYM.get(opener, opener), SYM.get(closer, closer)), "e": self.bind_nary(inner)}
        elif tok in SPACE:
            base = {"text": SPACE[tok], "plain": True}
        elif tok in SYM:
            base = {"text": SYM[tok]}
        elif tok.startswith("\\"):
            base = {"text": tok[1:], "plain": True}
        else:
            base = {"text": tok, "plain": tok.replace(".", "").isdigit() or not tok.isalpha()}
        return self.scripts(base)


def emit(items):
    return "".join(emit1(x) for x in items)


def emit1(x):
    if "group" in x:
        return emit(x["group"])
    if "frac" in x:
        n, d = x["frac"]
        return f"<m:f><m:num>{emit(n)}</m:num><m:den>{emit(d)}</m:den></m:f>"
    if "limlow" in x:
        lim = emit(x.get("lim") or [])
        base = run(x["limlow"], plain=True)
        if not lim:
            return base
        return f"<m:limLow><m:e>{base}</m:e><m:lim>{lim}</m:lim></m:limLow>"
    if "acc" in x:
        return f'<m:acc><m:accPr><m:chr m:val="{x["acc"]}"/></m:accPr><m:e>{emit(x["e"])}</m:e></m:acc>'
    if "nary" in x:
        sub, sup = x.get("sub"), x.get("sup")
        loc = "subSup" if MODE["inline"] else "undOvr"
        pr = f'<m:naryPr><m:chr m:val="{x["nary"]}"/><m:limLoc m:val="{loc}"/>' + \
             ('<m:subHide m:val="1"/>' if sub is None else "") + ('<m:supHide m:val="1"/>' if sup is None else "") + "</m:naryPr>"
        return f"<m:nary>{pr}<m:sub>{emit(sub or [])}</m:sub><m:sup>{emit(sup or [])}</m:sup><m:e>{emit(x['e'] or [])}</m:e></m:nary>"
    if "delim" in x:
        b, e = x["delim"]
        return f'<m:d><m:dPr><m:begChr m:val="{escape(b)}"/><m:endChr m:val="{escape(e)}"/></m:dPr><m:e>{emit(x["e"])}</m:e></m:d>'
    if "script" in x:
        base = emit1(x["script"])
        if x["sub"] is not None and x["sup"] is not None:
            return f"<m:sSubSup><m:e>{base}</m:e><m:sub>{emit(x['sub'])}</m:sub><m:sup>{emit(x['sup'])}</m:sup></m:sSubSup>"
        if x["sub"] is not None:
            return f"<m:sSub><m:e>{base}</m:e><m:sub>{emit(x['sub'])}</m:sub></m:sSub>"
        return f"<m:sSup><m:e>{base}</m:e><m:sup>{emit(x['sup'])}</m:sup></m:sSup>"
    if "text" in x:
        return run(x["text"], plain=x.get("plain", False))
    raise ValueError(x)


def omath(latex, inline=False):
    MODE["inline"] = inline
    p = P(latex)
    items = p.seq()
    if p.peek() is not None:
        raise ValueError(f"unparsed from token {p.i}: {p.t[p.i:]}")
    return f"<m:oMath>{emit(items)}</m:oMath>"


def omath_para(latex):
    return f'<m:oMathPara><m:oMathParaPr><m:jc m:val="left"/></m:oMathParaPr>{omath(latex)}</m:oMathPara>'


if __name__ == "__main__":
    tests = [r"\phi(x) = \frac{v(x)}{\lVert v(x)\rVert_2}, \qquad v(x) = \sum_{g \in G(x)} \sigma(g)\, e_{h(g)}",
             r"\mathcal{L}_t = -\frac{1}{|B_t|} \sum_{(c,f)\in B_t} \log \frac{\exp\big(s(c,f)/\tau\big)}{\sum_{f' \in C_t} \exp\big(s(c,f')/\tau\big)}",
             r"\mathrm{readout}_k(q) = \operatorname*{arg\,top}_{f \in F_t,\, k}\; s(q, f)",
             r"t^{*} = \Big\lfloor \frac{W - R}{d + Q_{\min}} \Big\rfloor + 1",
             r"A_h = \frac{1}{n_h} \sum_{i=1}^{n_h} \mathbb{1}\big[\hat{y}_i = y_i\big]",
             r"\tilde{p}(v) = (1 - w_\kappa)\, p(v) + w_\kappa \frac{N[\kappa][v]}{t_\kappa}"]
    for t in tests:
        x = omath(t); print(len(x), x[:160])
