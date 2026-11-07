"""StudyFlow — a live study session that builds your notes while you learn.

Run:  streamlit run app.py
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv not installed: keys can still be typed in the sidebar
    def load_dotenv(*_, **__):
        return False

from ai import (DEFAULT_STYLE, EXPORT_MODELS, EXTRAS, LAYOUTS, NOTES_MODELS, OPENAI_EXPORT_MODELS,
                OPENAI_NOTES_MODELS, OPENAI_TUTOR_MODELS, TUTOR_MODELS, DemoAI, OpenAIStudyAI, StudyAI)
from visuals import render_chart

load_dotenv()
SESSIONS = Path(__file__).parent / "sessions"
SESSIONS.mkdir(exist_ok=True)
TYPE_ICON = {"definition": "📘", "formula": "🧮", "concept": "💡",
             "procedure": "🪜", "example": "✏️", "tip": "🎯"}

st.set_page_config(page_title="StudyFlow", page_icon="📚", layout="wide")
st.markdown("""<style>
.note-card {border:1px solid rgba(128,128,128,.25); border-radius:10px; padding:.6rem .8rem; margin-bottom:.5rem}
.pending  {border-left:4px solid #f59f00}
.small    {font-size:.8rem; opacity:.7}
</style>""", unsafe_allow_html=True)


# ---------- state ----------
def new_session() -> dict:
    return {"id": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "meta": {"subject": "", "course": "", "chapter": "", "lesson": "", "goal": ""},
            "messages": [], "notes": [], "pending": [], "notion_page_id": None}


ss = st.session_state
ss.setdefault("session", new_session())
ss.setdefault("staged", [])          # screenshots waiting to be sent
ss.setdefault("last_paste", None)
ss.setdefault("charts", {})          # note id -> PNG bytes (cache)
S = ss.session


SETTINGS = Path(__file__).parent / ".studyflow_settings.json"


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except (OSError, ValueError):
        return {}


def save_settings(data: dict):
    SETTINGS.write_text(json.dumps({**load_settings(), **data}))


ICLOUD = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs"


def _safe_name(x: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "-", x or "").strip(" .-")


def export_path(meta: dict, layout: str) -> Path:
    """iCloud Drive/StudyFlow/<Subject>/ on a Mac (syncs to the iPad); ./exports otherwise."""
    root = ICLOUD / "StudyFlow" if ICLOUD.is_dir() else Path(__file__).parent / "exports"
    folder = root / (_safe_name(meta.get("subject")) or "General")
    folder.mkdir(parents=True, exist_ok=True)
    parts = [meta.get("subject"), meta.get("chapter"), meta.get("lesson")]
    stem = " - ".join(_safe_name(p) for p in parts if p) or datetime.now().strftime("Study guide %Y-%m-%d")
    return folder / f"{stem} ({layout}).pdf"


def save_session():
    (SESSIONS / f"{S['id']}.json").write_text(json.dumps(S, indent=1))


def chart_for(note: dict) -> bytes | None:
    vis = note.get("visual") or {}
    if vis.get("kind") != "chart":
        return None
    if note["id"] not in ss.charts:
        ss.charts[note["id"]] = render_chart(vis.get("chart_type", ""), vis.get("params"))
    return ss.charts[note["id"]]


def known_titles() -> list[str]:
    return [n["title"] for n in S["notes"] + S["pending"]]


def add_items(items: list[dict], mode: str):
    seen = {t.lower() for t in known_titles()}
    for it in items:
        if it["title"].lower() in seen:
            continue
        seen.add(it["title"].lower())
        it["id"] = uuid.uuid4().hex[:8]
        it["added"] = datetime.now().isoformat(timespec="minutes")
        if mode == "Auto-save everything" or (mode == "Auto-save core, ask about the rest" and it.get("importance", 1) == 3):
            S["notes"].append(it)
        else:
            S["pending"].append(it)


def accept(nid):
    it = next(p for p in S["pending"] if p["id"] == nid)
    S["pending"].remove(it)
    S["notes"].append(it)
    save_session()


def reject(nid):
    S["pending"] = [p for p in S["pending"] if p["id"] != nid]
    save_session()


def delete_note(nid):
    S["notes"] = [n for n in S["notes"] if n["id"] != nid]
    save_session()


def pin_answer(idx):
    m = S["messages"][idx]
    q = S["messages"][idx - 1].get("text", "") if idx else ""
    S["notes"].append({"id": uuid.uuid4().hex[:8], "type": "example", "importance": 2,
                       "title": (q[:60] or "Saved explanation").strip(), "content": m["text"],
                       "added": datetime.now().isoformat(timespec="minutes")})
    save_session()


def img_to_payload(data: bytes, name: str = "") -> dict:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else "png"
    mt = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/png")
    return {"media_type": mt, "data": base64.b64encode(data).decode()}


def to_markdown() -> str:
    m = S["meta"]
    title = " – ".join(v for v in (m["subject"], m["chapter"], m["lesson"]) if v) or "Study notes"
    out = [f"# {title}", "", " · ".join(f"**{k.title()}:** {v}" for k, v in m.items() if v), ""]
    for t, icon in TYPE_ICON.items():
        group = sorted((n for n in S["notes"] if n["type"] == t), key=lambda n: -n.get("importance", 1))
        if group:
            out += [f"## {icon} {t.title()}s", ""]
            for n in group:
                out += [f"### {'⭐' * n.get('importance', 1)} {n['title']}", ""]
                if n.get("latex"):
                    out += [f"$$ {n['latex']} $$", ""]
                out += [n["content"], ""]
                vis = n.get("visual") or {}
                if vis.get("kind") == "diagram" and vis.get("mermaid"):
                    out += ["```mermaid", vis["mermaid"], "```", ""]
    return "\n".join(out)


# ---------- sidebar ----------
with st.sidebar:
    st.title("📚 StudyFlow")
    def _env(name):
        v = os.getenv(name, "").strip()
        return "" if (not v or v.endswith("...")) else v

    default_provider = 1 if (_env("OPENAI_API_KEY") and not _env("ANTHROPIC_API_KEY")) else 0
    provider = st.radio("AI provider", ["Claude (Anthropic)", "ChatGPT (OpenAI)"],
                        index=default_provider, horizontal=True)
    use_openai = provider.startswith("ChatGPT")
    env_name = "OPENAI_API_KEY" if use_openai else "ANTHROPIC_API_KEY"
    api_key = _env(env_name)
    if api_key:
        st.caption(f"✅ Using the key from your .env file ({env_name})")
    else:
        api_key = st.text_input("OpenAI API key (sk-...)" if use_openai else "Anthropic API key (sk-ant-...)",
                                type="password", help=f"Or put it in .env as {env_name}=...")
    demo = st.toggle("Demo mode (no API key needed)", value=not api_key,
                     help="Sample answers and notes so you can try the app. Nothing is sent to the AI.")

    st.subheader("This session")
    meta = S["meta"]
    meta["subject"] = st.text_input("Subject", meta["subject"], placeholder="Statistics")
    meta["course"] = st.text_input("Course / textbook", meta["course"], placeholder="STAT 201 – OpenIntro Stats")
    c1, c2 = st.columns(2)
    meta["chapter"] = c1.text_input("Chapter", meta["chapter"], placeholder="Ch 5")
    meta["lesson"] = c2.text_input("Lesson", meta["lesson"], placeholder="5.2 CLT")
    meta["goal"] = st.text_input("Goal today (optional)", meta["goal"], placeholder="Understand confidence intervals")

    st.subheader("How the AI behaves")
    capture = st.radio("Saving notes", ["Ask me before saving", "Auto-save core, ask about the rest",
                                        "Auto-save everything", "Off (manual pins only)"], index=1)
    strict = st.toggle("Keep it on this subject", value=True)
    style = st.select_slider("Answer style", ["very brief", "concise", "thorough, step by step"], value="concise")
    with st.expander("Models"):
        tutor_model = st.selectbox("Tutor", OPENAI_TUTOR_MODELS if use_openai else TUTOR_MODELS,
                                   accept_new_options=True, help="Pick one, or type another model name.")
        notes_model = st.selectbox("Note-taker", OPENAI_NOTES_MODELS if use_openai else NOTES_MODELS,
                                   accept_new_options=True)
        export_model = st.selectbox("Study-guide builder", OPENAI_EXPORT_MODELS if use_openai else EXPORT_MODELS,
                                    accept_new_options=True, help="Used once per export, so a stronger model is worth it.")

    def make_ai():
        return DemoAI() if demo else (OpenAIStudyAI(api_key) if use_openai else StudyAI(api_key))

    st.subheader("✨ Export study guide")
    style = {**DEFAULT_STYLE, **load_settings().get("style", {})}
    with st.expander("🎨 My study style", expanded=False):
        new = {}
        new["layout"] = st.selectbox("Layout", list(LAYOUTS), index=list(LAYOUTS).index(style["layout"]),
                                     help="\n\n".join(f"**{k}:** {v}" for k, v in LAYOUTS.items()))
        new["order"] = st.radio("Inside each topic", ["Rule first, then example", "Example first, then the rule"],
                                index=0 if style["order"].startswith("Rule") else 1)
        new["visuals"] = st.select_slider("Visuals", ["Minimal", "Balanced", "Heavy"], value=style["visuals"])
        new["length"] = st.select_slider("Length", ["Condensed", "Standard", "Detailed"], value=style["length"])
        c1, c2 = st.columns(2)
        new["practice_count"] = c1.number_input("Practice questions", 0, 20, int(style["practice_count"]))
        new["difficulty"] = c2.selectbox("Difficulty", ["Easier", "Mixed", "Exam level", "Challenging"],
                                         index=["Easier", "Mixed", "Exam level", "Challenging"].index(style["difficulty"])
                                         if style["difficulty"] in ["Easier", "Mixed", "Exam level", "Challenging"] else 1)
        new["extras"] = st.multiselect("Include", EXTRAS, default=[e for e in style["extras"] if e in EXTRAS])
        new["custom"] = st.text_area("Anything else? (in your own words)", style.get("custom", ""),
                                     placeholder="e.g. I learn best from step-by-step worked problems. Keep explanations short.")
        if new != {k: style.get(k) for k in new}:
            save_settings({"style": new})
            style = {**style, **new}
        st.caption("Saved automatically. Every export uses this style.")
    double_check = st.toggle("Double-check against my session", value=True,
                             help="A second AI pass flags anything in the guide that wasn't actually covered.")

    has_content = bool(S["notes"] or S["messages"])
    if st.button("✨ Build study guide (PDF)", type="primary", use_container_width=True,
                 disabled=not (has_content and (api_key or demo)),
                 help="The AI reasons over everything from this session and organises it in your study style."):
        try:
            ai_ = make_ai()
            with st.spinner("Reading your session and writing the guide… (about a minute)"):
                guide = ai_.build_study_guide(meta, S["notes"], S["messages"], export_model, style, deep=True)
            issues = []
            if double_check:
                with st.spinner("Double-checking it against your session…"):
                    try:
                        issues = ai_.verify_guide(meta, S["notes"], S["messages"], guide, export_model)
                    except Exception as e:
                        st.warning(f"Skipped the double-check: {e}")
            with st.spinner("Designing the PDF…"):
                from study_pdf import build_pdf
                pdf = build_pdf(guide, meta, style)
            path = export_path(meta, style["layout"])
            path.write_bytes(pdf)
            S["guide"], S["guide_issues"] = guide, issues
            save_session()
            ss.last_pdf = (path.name, pdf, str(path))
        except Exception as e:
            st.error(f"Couldn't build the guide: {e}")
    if ss.get("last_pdf"):
        name, pdf, where = ss.last_pdf
        in_icloud = ICLOUD in Path(where).parents
        st.success(("Saved to iCloud Drive → StudyFlow. On your iPad: Files app → iCloud Drive → StudyFlow, "
                    "then share it to Freenotes.") if in_icloud else f"Saved to {where}")
        st.download_button("⬇️ Download the PDF", pdf, file_name=name, mime="application/pdf", use_container_width=True)
        if S.get("guide_issues"):
            with st.expander(f"⚠️ {len(S['guide_issues'])} thing(s) to double-check"):
                st.caption("The checker thinks these weren't covered in your session:")
                for it in S["guide_issues"]:
                    st.markdown(f"- **{it.get('where', '')}:** {it.get('statement', '')}  \n  _{it.get('why', '')}_")

    with st.expander("Other exports (Markdown, Notion)"):
        st.download_button("⬇️ Notes as Markdown (.md)", to_markdown(), file_name=f"{S['id']}-notes.md",
                           use_container_width=True)
        notion_token = _env("NOTION_TOKEN") or st.text_input("Notion integration token", type="password")
        notion_parent = _env("NOTION_PARENT_PAGE") or st.text_input("Notion page link to file notes under")
        if st.button("🔄 Sync to Notion", use_container_width=True, disabled=not (notion_token and notion_parent and S["notes"])):
            from notion_sync import NotionSync
            try:
                with st.spinner("Building your Notion page…"):
                    charts = {n["id"]: chart_for(n) for n in S["notes"] if chart_for(n)}
                    pid, url = NotionSync(notion_token).push(notion_parent, meta, S["notes"], charts, S["notion_page_id"])
                    S["notion_page_id"] = pid
                    save_session()
                st.success("Synced. Open it in the Notion app on your iPad.")
                if url:
                    st.link_button("Open in Notion", url)
            except Exception as e:
                st.error(f"Notion sync failed: {e}")

    st.subheader("Sessions")
    past = sorted(SESSIONS.glob("*.json"), reverse=True)
    pick = st.selectbox("Open a past session", ["—"] + [p.stem for p in past])
    b1, b2 = st.columns(2)
    if b1.button("Open", disabled=pick == "—", use_container_width=True):
        ss.session = json.loads((SESSIONS / f"{pick}.json").read_text())
        ss.charts, ss.last_pdf = {}, None
        st.rerun()
    if b2.button("New", use_container_width=True):
        save_session() if S["messages"] else None
        ss.session, ss.charts, ss.staged, ss.last_pdf = new_session(), {}, [], None
        st.rerun()

# ---------- main ----------
chat_col, notes_col = st.columns([3, 2], gap="large")

with chat_col:
    head = " · ".join(v for v in (meta["subject"], meta["chapter"], meta["lesson"]) if v)
    st.header(head or "Start a study session")
    if not head:
        st.caption("Fill in the subject, chapter and lesson on the left so the tutor and your notes stay organised.")

    for i, m in enumerate(S["messages"]):
        with st.chat_message(m["role"]):
            for img in m.get("images", []):
                st.image(base64.b64decode(img["data"]), width=320)
            if m.get("text"):
                st.markdown(m["text"])
            if m["role"] == "assistant":
                st.button("📌 Save this explanation", key=f"pin{i}", on_click=pin_answer, args=(i,))

    # screenshot staging: paste from clipboard (or attach in the chat box)
    try:
        from streamlit_paste_button import paste_image_button
        pasted = paste_image_button("📋 Paste screenshot", key="paste", errors="ignore")
        if pasted.image_data is not None:
            buf = io.BytesIO()
            pasted.image_data.convert("RGB").save(buf, format="PNG")
            h = hashlib.md5(buf.getvalue()).hexdigest()
            if h != ss.last_paste:
                ss.last_paste = h
                ss.staged.append(img_to_payload(buf.getvalue()))
    except ImportError:
        pass
    if ss.staged:
        cols = st.columns(min(len(ss.staged), 4) + 1)
        for c, img in zip(cols, ss.staged):
            c.image(base64.b64decode(img["data"]), width=140)
        if cols[-1].button("✕ Clear"):
            ss.staged = []
            st.rerun()

    prompt = st.chat_input("Ask a question, or attach/paste a screenshot…",
                           accept_file="multiple", file_type=["png", "jpg", "jpeg", "webp", "gif"])
    if prompt:
        text = prompt.text if hasattr(prompt, "text") else str(prompt)
        images = ss.staged + [img_to_payload(f.getvalue(), f.name) for f in getattr(prompt, "files", [])]
        ss.staged = []
        if not api_key and not demo:
            st.error("Add your API key in the sidebar, or turn on Demo mode.")
            st.stop()
        S["messages"].append({"role": "user", "text": text, "images": images})
        with st.chat_message("user"):
            for img in images:
                st.image(base64.b64decode(img["data"]), width=320)
            st.markdown(text)
        ai = make_ai()
        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(ai.stream_answer(S["messages"], meta, tutor_model, strict, style))
            except Exception as e:
                S["messages"].pop()
                st.error(f"The tutor couldn't answer: {e}")
                st.stop()
        S["messages"].append({"role": "assistant", "text": answer})
        if not capture.startswith("Off"):
            with st.spinner("Picking out what's worth keeping…"):
                try:
                    add_items(ai.extract_notes(text, bool(images), answer, meta, notes_model, known_titles()), capture)
                except Exception as e:
                    st.warning(f"Couldn't extract notes this time: {e}")
        save_session()
        st.rerun()

with notes_col:
    st.header("📝 Your notes")
    st.caption(f"{len(S['notes'])} saved · {len(S['pending'])} waiting for you")

    if S["pending"]:
        st.subheader("Save these for studying?")
        for p in list(S["pending"]):
            with st.container(border=True):
                st.markdown(f"{TYPE_ICON[p['type']]} **{p['title']}** <span class='small'>{'⭐' * p.get('importance', 1)}</span>", unsafe_allow_html=True)
                st.markdown(p["content"])
                a, b, _ = st.columns([1, 1, 2])
                a.button("Save", key=f"ok{p['id']}", on_click=accept, args=(p["id"],), type="primary")
                b.button("Skip", key=f"no{p['id']}", on_click=reject, args=(p["id"],))
        if len(S["pending"]) > 1:
            if st.button("Save all"):
                S["notes"] += S["pending"]
                S["pending"] = []
                save_session()
                st.rerun()
        st.divider()

    if not S["notes"]:
        st.info("Key definitions, formulas and concepts will build up here as you study.")
    view = st.segmented_control("Show", ["All"] + [t.title() for t in TYPE_ICON], default="All") or "All"
    for n in sorted(S["notes"], key=lambda n: (list(TYPE_ICON).index(n["type"]), -n.get("importance", 1))):
        if view != "All" and n["type"] != view.lower():
            continue
        with st.expander(f"{TYPE_ICON[n['type']]} {n['title']}  {'⭐' * n.get('importance', 1)}", expanded=n.get("importance", 1) == 3):
            if n.get("latex"):
                st.latex(n["latex"])
            st.markdown(n["content"])
            vis = n.get("visual") or {}
            png = chart_for(n)
            if png:
                st.image(png, caption=vis.get("caption"))
            elif vis.get("kind") == "diagram" and vis.get("mermaid"):
                st.code(vis["mermaid"], language="mermaid")
                st.caption("Diagram — renders as a picture in Notion")
            st.button("Delete", key=f"del{n['id']}", on_click=delete_note, args=(n["id"],))