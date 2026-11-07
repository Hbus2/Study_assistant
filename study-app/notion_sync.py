"""Sends a study session to Notion as a page you can open on your iPad.

One page per session (Subject - Chapter - Lesson), laid out as:
  callout with session info, then sections for Definitions, Formulas,
  Key concepts, Procedures, Examples, Tips. Formulas become real Notion
  equations, charts are uploaded as images, and diagrams are Mermaid
  blocks (Notion renders them).
Re-syncing the same session replaces that page's content, so it never duplicates.
"""
from __future__ import annotations

import re

from notion_client import Client

SECTIONS = [
    ("definition", "📘 Definitions"),
    ("formula", "🧮 Formulas"),
    ("concept", "💡 Key concepts"),
    ("procedure", "🪜 How-to / procedures"),
    ("example", "✏️ Worked examples"),
    ("tip", "🎯 Exam tips"),
]
STARS = {3: "⭐⭐⭐ ", 2: "⭐⭐ ", 1: ""}


def page_id_from(url_or_id: str) -> str:
    m = re.findall(r"[0-9a-fA-F]{32}", url_or_id.replace("-", ""))
    if not m:
        raise ValueError("Couldn't find a Notion page ID in that link.")
    return m[-1]


def _rt(text: str, bold: bool = False) -> list[dict]:
    """Rich text, split into Notion's 2000-character pieces, with $inline$ maths as equations."""
    out = []
    for i, part in enumerate(re.split(r"(?<!\$)\$([^$\n]+)\$(?!\$)", text or "")):
        if not part:
            continue
        for j in range(0, len(part), 1900):
            chunk = part[j:j + 1900]
            if i % 2:
                out.append({"type": "equation", "equation": {"expression": chunk}})
            else:
                out.append({"type": "text", "text": {"content": chunk},
                            "annotations": {"bold": bold}})
    return out or [{"type": "text", "text": {"content": ""}}]


def _markdown_blocks(md: str) -> list[dict]:
    """Small Markdown -> Notion converter: bullets, numbered lists, $$ display maths, paragraphs."""
    blocks, para = [], []

    def flush():
        if para:
            blocks.append({"type": "paragraph", "paragraph": {"rich_text": _rt(" ".join(para))}})
            para.clear()

    lines, i = (md or "").splitlines(), 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("$$"):
            flush()
            expr = line[2:]
            if expr.endswith("$$") and len(expr) > 2:
                expr = expr[:-2]
            else:
                i += 1
                while i < len(lines) and "$$" not in lines[i]:
                    expr += "\n" + lines[i]
                    i += 1
                if i < len(lines):
                    expr += "\n" + lines[i].split("$$")[0]
            blocks.append({"type": "equation", "equation": {"expression": expr.strip()}})
        elif re.match(r"^[-*•] ", line):
            flush()
            blocks.append({"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": _rt(line[2:].replace("**", ""))}})
        elif re.match(r"^\d+[.)] ", line):
            flush()
            blocks.append({"type": "numbered_list_item", "numbered_list_item": {"rich_text": _rt(re.sub(r"^\d+[.)] ", "", line).replace("**", ""))}})
        elif not line:
            flush()
        else:
            para.append(line.replace("**", "").lstrip("#").strip())
        i += 1
    flush()
    return blocks


class NotionSync:
    def __init__(self, token: str):
        self.client = Client(auth=token)

    def _upload_png(self, png: bytes, name: str) -> str:
        up = self.client.file_uploads.create(mode="single_part", filename=name, content_type="image/png")
        self.client.file_uploads.send(up["id"], file=(name, png, "image/png"))
        return up["id"]

    def _item_blocks(self, item: dict, chart_png: bytes | None) -> list[dict]:
        title = STARS.get(item.get("importance", 1), "") + item["title"]
        blocks = [{"type": "heading_3", "heading_3": {"rich_text": _rt(title)}}]
        if item["type"] == "definition":
            blocks.append({"type": "callout", "callout": {
                "rich_text": _rt(item["content"]), "icon": {"type": "emoji", "emoji": "📘"},
                "color": "blue_background"}})
        else:
            if item.get("latex"):
                blocks.append({"type": "equation", "equation": {"expression": item["latex"]}})
            blocks.extend(_markdown_blocks(item["content"]))
        vis = item.get("visual") or {}
        caption = _rt(vis.get("caption", "")) if vis.get("caption") else []
        if chart_png:
            fid = self._upload_png(chart_png, re.sub(r"\W+", "_", item["title"])[:40] + ".png")
            blocks.append({"type": "image", "image": {"type": "file_upload", "file_upload": {"id": fid}, "caption": caption}})
        elif vis.get("kind") == "diagram" and vis.get("mermaid"):
            blocks.append({"type": "code", "code": {"language": "mermaid", "rich_text": _rt(vis["mermaid"]), "caption": caption}})
        return blocks

    def push(self, parent: str, meta: dict, notes: list[dict], charts: dict[str, bytes],
             existing_page_id: str | None = None) -> tuple[str, str]:
        """Create (or refresh) the session page. Returns (page_id, url)."""
        title = " – ".join(v for v in (meta.get("subject"), meta.get("chapter"), meta.get("lesson")) if v) or "Study session"
        info = " · ".join(f"{k.title()}: {v}" for k, v in meta.items() if v)
        blocks = [{"type": "callout", "callout": {"rich_text": _rt(info or "Study session"),
                                                  "icon": {"type": "emoji", "emoji": "🎓"}, "color": "gray_background"}},
                  {"type": "table_of_contents", "table_of_contents": {}}]
        for key, heading in SECTIONS:
            items = sorted((n for n in notes if n["type"] == key), key=lambda n: -n.get("importance", 1))
            if not items:
                continue
            blocks.append({"type": "heading_2", "heading_2": {"rich_text": _rt(heading)}})
            for n in items:
                blocks.extend(self._item_blocks(n, charts.get(n["id"])))
            blocks.append({"type": "divider", "divider": {}})

        if existing_page_id:
            page_id = existing_page_id
            self.client.pages.update(page_id=page_id, properties={"title": {"title": _rt(title)}})
            while True:
                kids = self.client.blocks.children.list(block_id=page_id).get("results", [])
                if not kids:
                    break
                for child in kids:
                    self.client.blocks.delete(block_id=child["id"])
            page = self.client.pages.retrieve(page_id=page_id)
        else:
            page = self.client.pages.create(
                parent={"page_id": page_id_from(parent)},
                icon={"type": "emoji", "emoji": "📚"},
                properties={"title": {"title": _rt(title)}},
            )
            page_id = page["id"]
        for i in range(0, len(blocks), 90):  # Notion caps 100 blocks per request
            self.client.blocks.children.append(block_id=page_id, children=blocks[i:i + 90])
        return page_id, page.get("url", "")
