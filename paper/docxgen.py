"""A minimal .docx writer for journal manuscripts: Normal and Heading styles, Times 12,
US Letter with 2.5 cm margins, line numbers, page numbers, native Word equations, tables."""
import re, zipfile, datetime
from xml.sax.saxutils import escape
from omml import omath, omath_para

W_NS = ('xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" mc:Ignorable="w14"')

FONT = "Times New Roman"
MONO = "Courier New"


def styles_xml(line=360):
    def st(sid, name, ppr="", rpr="", based="Normal", nxt="Normal", typ="paragraph", extra=""):
        b = f'<w:basedOn w:val="{based}"/>' if based else ""
        return (f'<w:style w:type="{typ}" w:styleId="{sid}"><w:name w:val="{name}"/>{b}<w:next w:val="{nxt}"/>'
                f'{extra}<w:qFormat/><w:pPr>{ppr}</w:pPr><w:rPr>{rpr}</w:rPr></w:style>')
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles {W_NS}>'
            f'<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:eastAsia="{FONT}" w:cs="{FONT}"/>'
            '<w:sz w:val="24"/><w:szCs w:val="24"/><w:lang w:val="en-US"/></w:rPr></w:rPrDefault>'
            f'<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="{line}" w:lineRule="auto"/><w:jc w:val="left"/></w:pPr></w:pPrDefault></w:docDefaults>'
            '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/>'
            '<w:pPr><w:jc w:val="left"/></w:pPr><w:rPr/></w:style>'
            + st("Title", "Title", '<w:spacing w:before="0" w:after="240" w:line="276" w:lineRule="auto"/><w:outlineLvl w:val="0"/>',
                 '<w:b/><w:sz w:val="32"/><w:szCs w:val="32"/>')
            + st("Heading1", "heading 1", '<w:keepNext/><w:spacing w:before="360" w:after="120"/><w:outlineLvl w:val="0"/>',
                 '<w:b/><w:sz w:val="28"/><w:szCs w:val="28"/>')
            + st("Heading2", "heading 2", '<w:keepNext/><w:spacing w:before="240" w:after="120"/><w:outlineLvl w:val="1"/>',
                 '<w:b/><w:sz w:val="24"/><w:szCs w:val="24"/>')
            + st("Heading3", "heading 3", '<w:keepNext/><w:spacing w:before="200" w:after="80"/><w:outlineLvl w:val="2"/>',
                 '<w:b/><w:i/><w:sz w:val="24"/><w:szCs w:val="24"/>')
            + st("Code", "Code", '<w:keepLines/><w:spacing w:before="60" w:after="60" w:line="240" w:lineRule="auto"/><w:ind w:left="284"/>',
                 f'<w:rFonts w:ascii="{MONO}" w:hAnsi="{MONO}" w:cs="{MONO}"/><w:sz w:val="17"/><w:szCs w:val="17"/>')
            + st("TableText", "Table Text", '<w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/>',
                 '<w:sz w:val="20"/><w:szCs w:val="20"/>')
            + st("Equation", "Equation", '<w:spacing w:before="120" w:after="120"/><w:ind w:left="567"/>', "")
            + '</w:styles>')


def settings_xml():
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:settings {W_NS}>'
            '<w:defaultTabStop w:val="720"/><w:characterSpacingControl w:val="doNotCompress"/>'
            '<m:mathPr><m:mathFont m:val="Cambria Math"/><m:brkBin m:val="before"/><m:brkBinSub m:val="--"/>'
            '<m:smallFrac m:val="0"/><m:dispDef/><m:lMargin m:val="0"/><m:rMargin m:val="0"/><m:defJc m:val="left"/>'
            '<m:wrapIndent m:val="1440"/><m:intLim m:val="subSup"/><m:naryLim m:val="undOvr"/></m:mathPr>'
            '<w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat>'
            '</w:settings>')


def footer_xml():
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:ftr {W_NS}><w:p><w:pPr><w:jc w:val="right"/></w:pPr>'
            '<w:r><w:rPr><w:sz w:val="20"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'
            '<w:r><w:rPr><w:sz w:val="20"/></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
            '<w:r><w:rPr><w:sz w:val="20"/></w:rPr><w:fldChar w:fldCharType="separate"/></w:r>'
            '<w:r><w:rPr><w:sz w:val="20"/></w:rPr><w:t>1</w:t></w:r>'
            '<w:r><w:rPr><w:sz w:val="20"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')


INLINE = re.compile(r"(\$[^$]+\$|\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\^\{[^}]*\}|~\{[^}]*\})")


def runs(text, rpr=""):
    """Inline markup: **bold**, *italic*, `code`, $latex$, ^{superscript}, ~{subscript}."""
    t = text.strip()
    if t.startswith("$") and t.endswith("$") and t.count("$") == 2:
        return omath_para(t[1:-1])
    out = []
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("$") and part.endswith("$") and len(part) > 2:
            out.append(omath(part[1:-1], inline=True))
            continue
        r = rpr
        if part.startswith("**"):
            r += "<w:b/>"; part = part[2:-2]
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r += "<w:i/>"; part = part[1:-1]
        elif part.startswith("`"):
            r += f'<w:rFonts w:ascii="{MONO}" w:hAnsi="{MONO}" w:cs="{MONO}"/><w:sz w:val="21"/>'; part = part[1:-1]
        elif part.startswith("^{"):
            r += '<w:vertAlign w:val="superscript"/>'; part = part[2:-1]
        elif part.startswith("~{"):
            r += '<w:vertAlign w:val="subscript"/>'; part = part[2:-1]
        out.append(f'<w:r><w:rPr>{r}</w:rPr><w:t xml:space="preserve">{escape(part)}</w:t></w:r>')
    return "".join(out)


class Doc:
    def __init__(self, title="", author="", line_numbers=True, landscape=False, line=360):
        self.body, self.title, self.author = [], title, author
        self.line_numbers, self.landscape, self.line = line_numbers, landscape, line
        self.images = []

    def image(self, path, width_in=6.2):
        import struct
        with open(path, "rb") as fh:
            head = fh.read(24)
        w, h = struct.unpack(">II", head[16:24])
        n = len(self.images) + 1; rid = f"rIdImg{n}"
        self.images.append((rid, path, f"image{n}.png"))
        cx = int(width_in * 914400); cy = int(cx * h / w)
        self.body.append(
            '<w:p><w:pPr><w:keepNext/><w:jc w:val="center"/><w:spacing w:before="120" w:after="60" w:line="240" w:lineRule="auto"/></w:pPr><w:r><w:drawing>'
            f'<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{n}" name="Figure {n}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic>'
            f'<pic:nvPicPr><pic:cNvPr id="{n}" name="image{n}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')

    def p(self, text="", style="Normal", ppr="", rpr=""):
        s = f'<w:pStyle w:val="{style}"/>' if style != "Normal" else ""
        self.body.append(f"<w:p><w:pPr>{s}{ppr}</w:pPr>{runs(text, rpr)}</w:p>")

    def heading(self, text, level=1):
        self.p(text, f"Heading{level}")

    def title_p(self, text):
        self.p(text, "Title")

    def eq(self, latex):
        self.body.append(f'<w:p><w:pPr><w:pStyle w:val="Equation"/></w:pPr>{omath_para(latex)}</w:p>')

    def code(self, text):
        for line in text.rstrip("\n").split("\n"):
            self.body.append('<w:p><w:pPr><w:pStyle w:val="Code"/><w:spacing w:before="0" w:after="0"/></w:pPr>'
                             f'<w:r><w:t xml:space="preserve">{escape(line) if line else " "}</w:t></w:r></w:p>')
        self.body.append('<w:p><w:pPr><w:spacing w:before="0" w:after="60"/></w:pPr></w:p>')

    def bullets(self, items, numbered=False):
        for n, it in enumerate(items, 1):
            mark = f"{n}." if numbered else "•"
            self.body.append(f'<w:p><w:pPr><w:ind w:left="567" w:hanging="340"/><w:spacing w:after="60"/></w:pPr>'
                             f'<w:r><w:t xml:space="preserve">{mark}\t</w:t></w:r>{runs(it)}</w:p>')

    def page_break(self):
        self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def table(self, rows, widths=None, header=1, size=20, notes=None):
        ncol = len(rows[0])
        page_w = (15840 if self.landscape else 12240) - 2 * 1417
        widths = widths or [page_w // ncol] * ncol
        scale = page_w / sum(widths); widths = [int(w * scale) for w in widths]
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        border = ('<w:tblBorders><w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
                  '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/></w:tblBorders>')
        x = [f'<w:tbl><w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/>{border}<w:tblLayout w:type="fixed"/>'
             '<w:tblCellMar><w:left w:w="70" w:type="dxa"/><w:right w:w="70" w:type="dxa"/></w:tblCellMar></w:tblPr>'
             f'<w:tblGrid>{grid}</w:tblGrid>']
        for ri, row in enumerate(rows):
            hdr = ri < header
            trpr = '<w:trPr><w:cantSplit/>' + ('<w:tblHeader/>' if hdr else '') + '</w:trPr>'
            cells = []
            for ci, cell in enumerate(row):
                bd = ('<w:tcBorders><w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/></w:tcBorders>'
                      if ri == header - 1 else "")
                rp = f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>' + ("<w:b/>" if hdr else "")
                paras = "".join(f'<w:p><w:pPr><w:pStyle w:val="TableText"/></w:pPr>{runs(line, rp)}</w:p>'
                                for line in str(cell).split("\n"))
                cells.append(f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>{bd}</w:tcPr>{paras}</w:tc>')
            x.append(f"<w:tr>{trpr}{''.join(cells)}</w:tr>")
        x.append("</w:tbl>")
        self.body.append("".join(x))
        for n in (notes or []):
            self.p(n, "TableText", ppr='<w:spacing w:before="60"/>')

    def save(self, path):
        pw, ph = (15840, 12240) if self.landscape else (12240, 15840)
        orient = ' w:orient="landscape"' if self.landscape else ""
        ln = '<w:lnNumType w:countBy="1" w:distance="283" w:restart="continuous"/>' if self.line_numbers else ""
        sect = (f'<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter"/><w:pgSz w:w="{pw}" w:h="{ph}"{orient}/>'
                '<w:pgMar w:top="1417" w:right="1417" w:bottom="1417" w:left="1417" w:header="708" w:footer="567" w:gutter="0"/>'
                f'{ln}<w:cols w:space="708"/></w:sectPr>')
        doc = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {W_NS}><w:body>'
               + "".join(self.body) + sect + "</w:body></w:document>")
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        files = {
            "[Content_Types].xml": ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Default Extension="png" ContentType="image/png"/>'
                '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
                '<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>'
                '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
                '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
                '</Types>'),
            "_rels/.rels": ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
                '</Relationships>'),
            "word/_rels/document.xml.rels": ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                '<Relationship Id="rIdSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>'
                '<Relationship Id="rIdFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
                + "".join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{name}"/>'
                          for rid, _, name in self.images)
                + '</Relationships>'),
            "word/document.xml": doc,
            "word/styles.xml": styles_xml(self.line),
            "word/settings.xml": settings_xml(),
            "word/footer1.xml": footer_xml(),
            "docProps/core.xml": ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
                'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
                f'<dc:title>{escape(self.title)}</dc:title><dc:creator>{escape(self.author)}</dc:creator>'
                f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
                f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>'),
            "docProps/app.xml": ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
                '<Application>docxgen</Application></Properties>'),
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in files.items():
                z.writestr(name, data)
            for _, path, name in self.images:
                z.write(path, f"word/media/{name}")
        return path
