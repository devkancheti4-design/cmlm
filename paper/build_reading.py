"""A reading copy of the manuscript: every table and figure placed after the paragraph that first cites it.
The PeerJ upload manuscript keeps them as separate files, as the journal requires."""
import os, re
import build_tables as T            # builds the upload manuscript and the table files, and holds TABLES and FIGS
from docxgen import Doc, runs

B = T.B
FIG = os.path.join(B.HERE, "figures")
body = list(B.d.body)
plain = [re.sub(r"<[^>]+>", "", x) for x in body]


def first_cite(pattern):
    for i, t in enumerate(plain):
        if re.search(pattern, t):
            return i
    raise ValueError(pattern)


inserts = {}      # body index -> list of xml blocks to place after it
for num, title, legend, rows, widths, land, notes in T.TABLES:
    i = first_cite(rf"Table {num}(?!\d)")
    t = Doc(landscape=False)
    t.p(f"**Table {num}.** {title}", ppr='<w:keepNext/><w:spacing w:before="240" w:after="80" w:line="240" w:lineRule="auto"/>')
    t.table(rows, widths=widths, size=15 if len(rows[0]) >= 7 else 18)
    if legend:
        t.p(legend, "TableText", ppr='<w:spacing w:before="60" w:after="200"/>')
    inserts.setdefault(i, []).append(("table", num, t.body))
fig_pos = {}
for num, title, legend in T.FIGS:
    i = first_cite(rf"(?:Fig\.|Figure) {num}(?!\d)")
    inserts.setdefault(i, []).append(("figure", num, (title, legend)))

d = Doc(title=B.TITLE, author="Devieswar Kancheti")
for i, x in enumerate(body):
    d.body.append(x)
    for kind, num, payload in sorted(inserts.get(i, []), key=lambda k: (k[0] != "table", k[1])):
        if kind == "table":
            d.body.extend(payload)
        else:
            title, legend = payload
            d.image(os.path.join(FIG, f"Figure {num}.png"), width_in=6.2)
            d.body.append('<w:p><w:pPr><w:pStyle w:val="TableText"/><w:spacing w:before="40" w:after="240"/></w:pPr>'
                          + runs(f"**Figure {num}.** {title} {legend}") + "</w:p>")
out = os.path.join(B.OUT, "Manuscript with tables and figures.docx")
d.save(out)
print("reading copy:", out, "| tables placed:", len(T.TABLES), "| figures placed:", len(T.FIGS))
