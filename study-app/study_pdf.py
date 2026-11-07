"""Designs the study-guide PDF you open in Freenotes (or any PDF notes app).

The AI decides WHAT goes in the guide; this module decides how it LOOKS, so every
export is consistent. Four layouts: full guide, Cornell notes, one-page cheat sheet
and practice workbook. Formulas are drawn as real maths, charts come from
visuals.py, diagrams are drawn here, and there is lined space for handwriting.
"""
from __future__ import annotations

import io
import os
import re
import tempfile
import textwrap
from datetime import date
from itertools import zip_longest
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Flowable, Frame, Image, KeepTogether,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)

from visuals import render_chart

# ---------- look ----------
INK = colors.HexColor("#1f2937")
MUTED = colors.HexColor("#6b7280")
ACCENT = colors.HexColor("#3b5bdb")
RULE = colors.HexColor("#d7dbe3")
BOX = {"blue": ("#eef2ff", "#3b5bdb"), "green": ("#ecfdf3", "#2f9e44"), "amber": ("#fff8e6", "#e8a400"),
       "red": ("#fff1f1", "#e03131"), "gray": ("#f4f5f7", "#868e96"), "violet": ("#f5f0ff", "#7048e8")}
DIFF = {"easy": ("Easy", "#2f9e44"), "medium": ("Medium", "#e8a400"), "hard": ("Hard", "#e03131")}

_FONTS = False


def _register_fonts():
    global _FONTS
    if _FONTS:
        return
    d = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    for name, f in [("Body", "DejaVuSans.ttf"), ("Body-Bold", "DejaVuSans-Bold.ttf"),
                    ("Body-Italic", "DejaVuSans-Oblique.ttf"), ("Body-BoldItalic", "DejaVuSans-BoldOblique.ttf")]:
        pdfmetrics.registerFont(TTFont(name, os.path.join(d, f)))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")
    _FONTS = True


def _styles(base: float) -> dict:
    s = {}
    s["title"] = ParagraphStyle("title", fontName="Body-Bold", fontSize=base + 11, leading=base + 15, textColor=INK, spaceAfter=2)
    s["meta"] = ParagraphStyle("meta", fontName="Body", fontSize=base - 1, leading=base + 3, textColor=MUTED, spaceAfter=10)
    s["h1"] = ParagraphStyle("h1", fontName="Body-Bold", fontSize=base + 5, leading=base + 9, textColor=ACCENT, spaceBefore=12, spaceAfter=4)
    s["h2"] = ParagraphStyle("h2", fontName="Body-Bold", fontSize=base + 2, leading=base + 6, textColor=INK, spaceBefore=8, spaceAfter=3)
    s["body"] = ParagraphStyle("body", fontName="Body", fontSize=base, leading=base * 1.45, textColor=INK, spaceAfter=base * 0.5)
    s["small"] = ParagraphStyle("small", parent=s["body"], fontSize=base - 2, leading=(base - 2) * 1.35, textColor=MUTED, spaceAfter=2)
    s["bullet"] = ParagraphStyle("bullet", parent=s["body"], leftIndent=14, bulletIndent=3, spaceAfter=base * 0.25)
    s["boxtitle"] = ParagraphStyle("boxtitle", fontName="Body-Bold", fontSize=base - 0.5, leading=base + 3, spaceAfter=2)
    s["cue"] = ParagraphStyle("cue", parent=s["body"], fontName="Body-Bold", fontSize=base - 1, textColor=ACCENT)
    s["caption"] = ParagraphStyle("caption", parent=s["small"], alignment=TA_CENTER, spaceAfter=8)
    return s


def _clean(text) -> str:
    """Drop emoji and other symbols the PDF font can't draw."""
    text = str(text or "")
    text = re.sub(r"[\U00010000-\U0010FFFF\u2600-\u27BF\uFE0F\u200D]", "", text)
    return text.replace("\u2212", "-").strip()


# ---------- maths ----------
class Maths:
    """Renders LaTeX with matplotlib's built-in maths engine (no LaTeX install needed)."""

    def __init__(self, tmpdir: str, base: float):
        self.tmp, self.base, self.n = tmpdir, base, 0

    @staticmethod
    def _tidy(tex: str) -> str:
        tex = tex.strip().strip("$")
        tex = re.sub(r"\\(text|operatorname|textrm|mbox)\{", r"\\mathrm{", tex)
        tex = re.sub(r"\\[dt]frac", r"\\frac", tex)
        tex = re.sub(r"\\begin\{[a-z*]+\}|\\end\{[a-z*]+\}", "", tex)
        tex = tex.replace("\\\\", r"\quad ").replace("&", "")
        tex = re.sub(r"\\(big|Big|bigg|Bigg)[lr]?", "", tex)
        tex = tex.replace(r"\displaystyle", "").replace(r"\,", r"\ ").replace(r"\;", r"\ ").replace(r"\!", "")
        tex = tex.replace(r"\le ", r"\leq ").replace(r"\ge ", r"\geq ").replace(r"\ne ", r"\neq ")
        tex = re.sub(r"\\(left|right)\.", "", tex)
        return tex

    def png(self, tex: str, size: float) -> tuple[str, float, float] | None:
        tex = self._tidy(tex)
        if not tex:
            return None
        fig = plt.figure(figsize=(0.01, 0.01))
        try:
            fig.text(0, 0, f"${tex}$", fontsize=size, color="#1f2937")
            buf = io.BytesIO()
            fig.savefig(buf, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.015, format="png")
        except Exception:
            return None
        finally:
            plt.close(fig)
        from PIL import Image as PILImage
        buf.seek(0)
        w, h = PILImage.open(buf).size
        self.n += 1
        path = os.path.join(self.tmp, f"m{self.n}.png")
        with open(path, "wb") as f:
            f.write(buf.getvalue())
        return path, w * 72 / 300, h * 72 / 300

    def inline(self, tex: str, size: float) -> str:
        r = self.png(tex, size)
        if not r:
            return f'<font name="Body-Italic">{escape(tex)}</font>'
        path, w, h = r
        return f'<img src="{path}" width="{w:.1f}" height="{h:.1f}" valign="{-(h - size) / 2 - 1:.1f}"/>'

    def block(self, tex: str, max_w: float) -> Flowable:
        r = self.png(tex, self.base * 1.8)
        if not r:
            return Paragraph(escape(tex), ParagraphStyle("raw", fontName="Courier", fontSize=self.base - 1, alignment=TA_CENTER))
        path, w, h = r
        if w > max_w:
            w, h = max_w, h * max_w / w
        img = Image(path, width=w, height=h)
        img.hAlign = "CENTER"
        img.spaceBefore, img.spaceAfter = 3, 6
        return img


# ---------- markdown -> flowables ----------
class Writer:
    def __init__(self, styles: dict, maths: Maths, width: float):
        self.s, self.m, self.width = styles, maths, width

    def inline(self, text: str, style: ParagraphStyle | None = None) -> str:
        size = (style or self.s["body"]).fontSize
        parts = re.split(r"(?<!\\)\$([^$]+?)\$", _clean(text))
        out = []
        for i, p in enumerate(parts):
            if i % 2:
                out.append(self.m.inline(p, size))
            else:
                p = escape(p)
                p = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", p)
                p = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", p)
                p = re.sub(r"`(.+?)`", r"<b>\1</b>", p)
                out.append(p)
        return "".join(out)

    def para(self, text: str, style: str = "body", **kw) -> Paragraph:
        st = self.s[style]
        return Paragraph(self.inline(text, st), st, **kw)

    def blocks(self, md: str, width: float | None = None) -> list:
        width = width or self.width
        out, para = [], []

        def flush():
            if para:
                out.append(self.para(" ".join(para)))
                para.clear()

        lines = _clean(md).replace("\r", "").split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("$$"):
                flush()
                tex = line[2:]
                if tex.endswith("$$") and len(tex) >= 2:
                    tex = tex[:-2]
                else:
                    i += 1
                    while i < len(lines) and "$$" not in lines[i]:
                        tex += " " + lines[i]
                        i += 1
                    if i < len(lines):
                        tex += " " + lines[i].split("$$")[0]
                out.append(self.m.block(tex, width))
            elif re.match(r"^[-*•] ", line):
                flush()
                out.append(Paragraph(self.inline(line[2:]), self.s["bullet"], bulletText="•"))
            elif re.match(r"^\d+[.)] ", line):
                flush()
                num = re.match(r"^(\d+)", line).group(1)
                out.append(Paragraph(self.inline(re.sub(r"^\d+[.)] ", "", line)), self.s["bullet"], bulletText=f"{num}."))
            elif line.startswith("#"):
                flush()
                out.append(self.para(line.lstrip("#").strip(), "h2"))
            elif not line:
                flush()
            else:
                para.append(line)
            i += 1
        flush()
        return out

    def box(self, title: str, content: list, tone: str = "blue", width: float | None = None):
        bg, edge = BOX[tone]
        width = width or self.width
        head = Paragraph(escape(_clean(title)), ParagraphStyle("bt", parent=self.s["boxtitle"], textColor=colors.HexColor(edge)))
        t = Table([[[head] + content]], colWidths=[width])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
            ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(edge)),
            ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        t.spaceAfter = 8
        return t

    def bullets(self, items: list[str]) -> list:
        return [Paragraph(self.inline(x), self.s["bullet"], bulletText="•") for x in items if str(x).strip()]


# ---------- pictures ----------
def _picture(png: bytes | None, max_w: float, max_h: float = 3.3 * inch) -> Image | None:
    if not png:
        return None
    from PIL import Image as PILImage
    w, h = PILImage.open(io.BytesIO(png)).size
    scale = min(max_w / w, max_h / h)
    img = Image(io.BytesIO(png), width=w * scale, height=h * scale)
    img.hAlign = "CENTER"
    return img


def draw_diagram(edges: list[dict], title: str = "") -> bytes | None:
    """Flowchart / concept map from arrows, laid out left to right."""
    edges = [e for e in edges if isinstance(e, dict) and e.get("from") and e.get("to")][:16]
    if not edges:
        return None
    nodes = []
    for e in edges:
        for k in ("from", "to"):
            label = _clean(e[k])
            if label not in nodes:
                nodes.append(label)
    parents = {n: [] for n in nodes}
    for e in edges:
        parents[_clean(e["to"])].append(_clean(e["from"]))
    level = {}

    def depth(n, seen=()):
        if n in level:
            return level[n]
        ps = [p for p in parents[n] if p not in seen]
        level[n] = 0 if not ps else 1 + max(depth(p, seen + (n,)) for p in ps)
        return level[n]

    for n in nodes:
        depth(n)
    cols = {}
    for n in nodes:
        cols.setdefault(level[n], []).append(n)
    ncols, nrows = max(cols) + 1, max(len(v) for v in cols.values())
    gap = 3.0 if any(e.get("label") for e in edges) else 2.4
    fig_w, fig_h = max(4.5, gap * 0.92 * ncols), max(1.3, 1.0 * nrows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    pos = {}
    for c, members in cols.items():
        for r, n in enumerate(members):
            pos[n] = (c * gap + 1.1, (nrows - 1) / 2 - r + (len(members) - 1) / 2 - (nrows - 1) / 2 + (nrows - 1) / 2)
    bw, bh = 1.8, 0.7
    for e in edges:
        a, b = pos[_clean(e["from"])], pos[_clean(e["to"])]
        ax.annotate("", xy=(b[0] - bw / 2, b[1]), xytext=(a[0] + bw / 2, a[1]),
                    arrowprops=dict(arrowstyle="-|>", color="#7c8db5", lw=1.3, shrinkA=0, shrinkB=0,
                                    connectionstyle="arc3,rad=0.0"))
        if e.get("label"):
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, _clean(e["label"]), fontsize=10,
                    color="#3b5bdb", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    for n, (x, y) in pos.items():
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#eef2ff", ec="#3b5bdb", lw=1.2))
        ax.text(x, y, "\n".join(textwrap.wrap(n, 16)[:3]), ha="center", va="center", fontsize=11.5, color="#1f2937")
    ax.set_xlim(0, (ncols - 1) * gap + 2.2)
    ax.set_ylim(-0.6, nrows - 0.4)
    ax.axis("off")
    if title:
        ax.set_title(_clean(title), fontsize=9, color="#1f2937")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return buf.getvalue()


def _visual(vis: dict | None, w: Writer, max_w: float, max_h: float = 2.6 * inch) -> list:
    vis = vis or {}
    png = None
    try:
        if vis.get("kind") == "chart":
            png = render_chart(vis.get("chart_type", ""), vis.get("params") or {})
        elif vis.get("kind") == "diagram":
            png = draw_diagram(vis.get("edges") or [])
    except Exception:
        png = None
    img = _picture(png, max_w, max_h)
    if not img:
        return []
    img.spaceBefore = 4
    out = [img]
    if vis.get("caption"):
        out.append(Paragraph(escape(_clean(vis["caption"])), w.s["caption"]))
    return out


def _kt(items: list) -> list:
    """Keep a picture with its caption at the top level of the page."""
    return [KeepTogether(items)] if items else []


# ---------- handwriting space ----------
class Lines(Flowable):
    """Ruled lines to write on."""

    def __init__(self, n: int, width: float, gap: float = 24):
        super().__init__()
        self.n, self.w, self.gap = n, width, gap

    def wrap(self, *_):
        return self.w, self.n * self.gap + 4

    def draw(self):
        self.canv.setStrokeColor(RULE)
        self.canv.setLineWidth(0.6)
        for i in range(1, self.n + 1):
            y = self.n * self.gap + 4 - i * self.gap
            self.canv.line(0, y, self.w, y)


# ---------- page frame ----------
def _footer(title: str, notes_margin: float):
    def draw(canv, doc):
        canv.saveState()
        pw, ph = doc.pagesize
        canv.setFont("Body", 7.5)
        canv.setFillColor(MUTED)
        canv.drawString(doc.leftMargin, 0.45 * inch, _clean(title)[:90])
        canv.drawRightString(pw - 0.6 * inch, 0.45 * inch, f"{doc.page}")
        if notes_margin:
            x = pw - notes_margin + 0.15 * inch
            canv.setStrokeColor(RULE)
            canv.setDash(2, 3)
            canv.line(x, 0.8 * inch, x, ph - 0.7 * inch)
            canv.setFont("Body", 7)
            canv.drawString(x + 6, ph - 0.62 * inch, "your notes")
        canv.restoreState()
    return draw


def _doc(buf, title: str, layout: str):
    pw, ph = letter
    if layout == "One-page cheat sheet":
        lm = rm = 0.45 * inch
        gap = 0.25 * inch
        fw = (pw - lm - rm - gap) / 2
        frames = [Frame(lm, 0.7 * inch, fw, ph - 1.2 * inch, id="c1", leftPadding=0, rightPadding=0),
                  Frame(lm + fw + gap, 0.7 * inch, fw, ph - 1.2 * inch, id="c2", leftPadding=0, rightPadding=0)]
        doc = BaseDocTemplate(buf, pagesize=letter, leftMargin=lm, rightMargin=rm, title=title)
        doc.addPageTemplates([PageTemplate(frames=frames, onPage=_footer(title, 0))])
        return doc, fw
    notes = 1.55 * inch if layout in ("Full study guide", "Practice workbook") else 0
    lm, rm = 0.8 * inch, (notes or 0.75 * inch)
    doc = BaseDocTemplate(buf, pagesize=letter, leftMargin=lm, rightMargin=rm, title=title)
    fw = pw - lm - rm
    doc.addPageTemplates([PageTemplate(frames=[Frame(lm, 0.75 * inch, fw, ph - 1.35 * inch, leftPadding=0, rightPadding=0)],
                                       onPage=_footer(title, notes))])
    return doc, fw


# ---------- the guide ----------
def build_pdf(guide: dict, meta: dict, style: dict) -> bytes:
    _register_fonts()
    layout = style.get("layout", "Full study guide")
    extras = set(style.get("extras", []))
    base = 8.6 if layout == "One-page cheat sheet" else (10.2 if style.get("length") == "Condensed" else 10.6)
    title = _clean(guide.get("title")) or "Study guide"
    buf = io.BytesIO()
    doc, width = _doc(buf, title, layout)
    tmp = tempfile.mkdtemp(prefix="studyflow-")
    w = Writer(_styles(base), Maths(tmp, base), width)
    s = w.s
    story = []

    meta_line = " · ".join(_clean(meta.get(k)) for k in ("course", "chapter", "lesson") if meta.get(k))
    story += [Paragraph(escape(title), s["title"]),
              Paragraph(escape(" · ".join(x for x in (meta_line, date.today().strftime("%B %d, %Y")) if x)), s["meta"])]
    if meta.get("goal"):
        story.append(w.para(f"**Goal:** {meta['goal']}", "small"))

    def src(item):
        if "Source tags (where each part came from)" in extras and item.get("sources"):
            return [Paragraph("From: " + escape(", ".join(map(str, item["sources"]))), s["small"])]
        return []

    sections = guide.get("sections", [])
    if layout != "One-page cheat sheet":
        story.append(w.box("Overview", w.blocks(guide.get("overview", "")), "blue"))
        if "Concept map" in extras and guide.get("concept_map"):
            img = _picture(draw_diagram(guide["concept_map"]), width, 3.4 * inch)
            if img:
                story += [Paragraph("Concept map", s["h2"]), img, Spacer(1, 6)]

    # ---- layouts ----
    if layout == "Cornell notes":
        cue_w = width * 0.28
        for sec in sections:
            story += [CondPageBreak(1.6 * inch), Paragraph(w.inline(sec["heading"], s["h1"]), s["h1"])] + src(sec)
            right = w.blocks(sec.get("body", ""), width - cue_w - 16)
            if sec.get("key_points"):
                right.append(w.box("Key points", w.bullets(sec["key_points"]), "green", width - cue_w - 16))
            right += _visual(sec.get("visual"), w, width - cue_w - 16, 2.6 * inch)
            cues = [Paragraph(w.inline(c, s["cue"]), s["cue"]) for c in sec.get("cues", [])]
            rows = [[c or "", r or ""] for c, r in zip_longest(cues, right)]
            t = Table(rows, colWidths=[cue_w, width - cue_w], splitByRow=1)
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("LINEAFTER", (0, 0), (0, -1), 0.8, RULE),
                                   ("LEFTPADDING", (1, 0), (1, -1), 14), ("LEFTPADDING", (0, 0), (0, -1), 0),
                                   ("RIGHTPADDING", (0, 0), (0, -1), 10),
                                   ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
            story.append(t)
            summ = [w.para(sec.get("summary", ""))] if sec.get("summary") else []
            story.append(w.box("Summary (add your own below)", summ + [Lines(2, width - 20, 20)], "gray"))
    elif layout == "One-page cheat sheet":
        story.append(w.para(guide.get("overview", ""), "small"))
        for sec in sections:
            story.append(Paragraph(w.inline(sec["heading"], s["h2"]), s["h2"]))
            story += w.bullets(sec.get("key_points") or [sec.get("summary", "")])
            story += _kt(_visual(sec.get("visual"), w, width, 1.6 * inch))
    elif layout == "Practice workbook":
        story.append(Paragraph("Quick recap", s["h1"]))
        for sec in sections:
            items = w.bullets(sec.get("key_points") or [sec.get("summary", "")])
            story.append(w.box(sec["heading"], items + _visual(sec.get("visual"), w, width - 24, 2.2 * inch), "blue"))
    else:  # Full study guide
        for sec in sections:
            story += [CondPageBreak(1.6 * inch), Paragraph(w.inline(sec["heading"], s["h1"]), s["h1"])] + src(sec)
            story += w.blocks(sec.get("body", ""))
            if "Explain-it-simply boxes" in extras and sec.get("simple"):
                story.append(w.box("In plain words", [w.para(sec["simple"])], "violet"))
            story += _kt(_visual(sec.get("visual"), w, width))
            if sec.get("key_points"):
                story.append(w.box("Key points", w.bullets(sec["key_points"]), "green"))
            if sec.get("connects_to"):
                story.append(w.para(f"*Connects to:* {sec['connects_to']}", "small"))

    # ---- shared end matter ----
    if guide.get("formula_sheet"):
        story.append(Paragraph("Formula sheet", s["h1"] if layout != "One-page cheat sheet" else s["h2"]))
        rows = []
        for f in guide["formula_sheet"]:
            fw = width * (0.45 if layout != "One-page cheat sheet" else 0.5)
            rows.append([Paragraph(f"<b>{escape(_clean(f.get('name')))}</b>", s["body"]),
                         w.m.block(f.get("latex", ""), fw),
                         w.para(f.get("when_to_use", ""), "small")])
        if layout == "One-page cheat sheet":
            for name, tex, use in rows:
                story += [name, tex, use]
        else:
            t = Table(rows, colWidths=[width * 0.22, width * 0.45, width * 0.33], splitByRow=1)
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                   ("LINEBELOW", (0, 0), (-1, -2), 0.5, RULE),
                                   ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
            story.append(t)
    if "Common-mistake warnings" in extras and guide.get("common_mistakes"):
        story.append(w.box("Common mistakes", w.bullets(guide["common_mistakes"]), "red"))
    if "Memory tricks" in extras and guide.get("memory_tricks"):
        story.append(w.box("Memory tricks", w.bullets(guide["memory_tricks"]), "amber"))

    qs = guide.get("practice_questions", [])
    if qs and layout != "One-page cheat sheet":
        if layout != "Practice workbook":
            story.append(PageBreak())
        intro = [Paragraph("Practice questions", s["h1"]),
                 w.para("Work each one out in the space, then check the answer key at the end.", "small")]
        n_lines = 9 if layout == "Practice workbook" else 6
        for i, q in enumerate(qs, 1):
            label, col = DIFF.get(q.get("difficulty"), ("", "#868e96"))
            head = f'<b>Q{i}.</b> <font color="{col}" size="{base - 2}">{label}</font>'
            block = (intro if i == 1 else []) + [Paragraph(head, s["h2"]), w.para(q.get("question", ""))] + src(q) + [Lines(n_lines, width)]
            story += [KeepTogether(block), Spacer(1, 8)]
        story += [PageBreak(), Paragraph("Answer key", s["h1"])]
        for i, q in enumerate(qs, 1):
            story += [Paragraph(f"<b>Q{i}</b>", s["h2"])] + w.blocks(q.get("answer", ""))
    elif qs:  # cheat sheet: a few quick checks
        story.append(Paragraph("Quick check", s["h2"]))
        for q in qs[:3]:
            story += [w.para(f"**Q:** {q.get('question', '')}", "small"), w.para(f"**A:** {q.get('answer', '')}", "small")]

    if layout not in ("One-page cheat sheet",):
        story += [CondPageBreak(2.5 * inch), Paragraph("My notes", s["h1"]), Lines(14, width)]

    doc.build(story)
    return buf.getvalue()