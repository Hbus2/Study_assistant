"""AI layer: the tutor (answers questions, reads screenshots) and the note-taker
(pulls key items out of each exchange in a fixed, structured format).

Control lives here: the system prompts keep Claude on your subject, and the
note-taker is forced to answer through a JSON schema, so it can only return
the kinds of notes the app knows how to store.
"""
from __future__ import annotations

import json
from typing import Iterator

import anthropic

TUTOR_MODELS = ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"]
NOTES_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
EXPORT_MODELS = ["claude-sonnet-5-5", "claude-opus-5-5"]

ITEM_TYPES = ["definition", "formula", "concept", "procedure", "example", "tip"]
CHART_TYPES = ["normal", "t", "chi2", "binomial", "poisson", "uniform",
               "scatter", "bar", "line", "boxplot", "histogram"]


def _context(meta: dict) -> str:
    parts = [f"{k.title()}: {v}" for k, v in meta.items() if v]
    return "\n".join(parts) if parts else "No session details given yet."


def tutor_system(meta: dict, strict: bool, style: str) -> str:
    focus = (
        "Stay on this subject. If the student asks something unrelated, answer in one "
        "line at most and steer back to the lesson."
        if strict else
        "Prefer this subject, but you may answer related side questions."
    )
    return f"""You are a patient study tutor working through a live study session.

SESSION
{_context(meta)}

HOW TO HELP
- {focus}
- When the student sends a screenshot (a textbook page, a homework problem, lecture
  slides), first say briefly what it shows, then help with it.
- For "how do I do this" questions, teach the method step by step and explain why
  each step works. Show the worked answer, then a one-line takeaway.
- Use the terms and notation this course would use. Write maths in LaTeX with $...$
  inline and $$...$$ for display equations.
- Answer style: {style}.
- Do not invent page numbers, quotes or textbook content you cannot see."""


NOTES_TOOL = {
    "name": "record_study_notes",
    "description": "Record the items from this exchange worth keeping for later study.",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": "Key study items. Empty if the exchange had nothing worth keeping.",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ITEM_TYPES},
                        "title": {"type": "string", "description": "Short name, e.g. 'Standard error of the mean'."},
                        "content": {"type": "string", "description": "The note itself, concise, in Markdown with LaTeX ($...$) for maths."},
                        "latex": {"type": "string", "description": "For formulas only: the bare LaTeX expression, no $ signs."},
                        "importance": {"type": "integer", "enum": [1, 2, 3], "description": "3 = core, likely on an exam."},
                        "visual": {
                            "type": "object",
                            "description": "Optional. Only when a picture genuinely helps memorise this item.",
                            "properties": {
                                "kind": {"type": "string", "enum": ["chart", "diagram"]},
                                "chart_type": {"type": "string", "enum": CHART_TYPES},
                                "params": {
                                    "type": "object",
                                    "description": (
                                        "Chart parameters. normal: mu, sigma, shade_from, shade_to. "
                                        "t: df. chi2: df. binomial: n, p. poisson: lam. uniform: a, b. "
                                        "scatter/line: x (list), y (list). bar: labels (list), values (list). "
                                        "boxplot/histogram: data (list). Any: title, xlabel, ylabel."
                                    ),
                                },
                                "mermaid": {"type": "string", "description": "For kind=diagram: Mermaid code (flowchart/mindmap) e.g. a decision tree for choosing a test."},
                                "caption": {"type": "string"},
                            },
                            "required": ["kind"],
                        },
                    },
                    "required": ["type", "title", "content", "importance"],
                },
            }
        },
        "required": ["items"],
    },
}


def notes_system(meta: dict, already: list[str]) -> str:
    have = ", ".join(already[-60:]) or "none yet"
    return f"""You are the note-taker for a study session. Read one tutor exchange and record
only what a student should keep for revision.

SESSION
{_context(meta)}

RULES
- Keep definitions, formulas, core concepts, step-by-step procedures, one good worked
  example, and exam tips. Skip chit-chat, restated questions and filler.
- Be concise: notes are for revision, not a transcript.
- Do not repeat items the notes already have: {have}.
- Only add a visual when it truly helps (a distribution curve, a comparison chart, a
  decision flowchart). Most items need none.
- If nothing is worth keeping, return an empty list."""


# ---------- Study guide (built when you click Export) ----------
LAYOUTS = {
    "Full study guide": "A complete guide: overview, each topic explained with examples, formula sheet, mistakes, practice questions.",
    "Cornell notes": "Cornell format: for each topic a cue column (short questions/terms), the notes, and a 1-2 sentence summary.",
    "One-page cheat sheet": "Condensed to the essentials: key points, formulas and tiny examples only. Very terse.",
    "Practice workbook": "Brief recap of each topic, then many practice problems with space to work them out.",
}
EXTRAS = ["Memory tricks", "Explain-it-simply boxes", "Common-mistake warnings",
          "Real-world examples", "Concept map", "Source tags (where each part came from)"]
DEFAULT_STYLE = {
    "layout": "Full study guide", "order": "Rule first, then example", "visuals": "Balanced",
    "practice_count": 6, "difficulty": "Mixed", "length": "Standard",
    "extras": ["Common-mistake warnings", "Concept map", "Source tags (where each part came from)"],
    "custom": "",
}

GUIDE_VISUAL = {
    "type": "object",
    "description": "Optional picture for this topic.",
    "properties": {
        "kind": {"type": "string", "enum": ["chart", "diagram"]},
        "chart_type": {"type": "string", "enum": CHART_TYPES},
        "params": NOTES_TOOL["input_schema"]["properties"]["items"]["items"]["properties"]["visual"]["properties"]["params"],
        "edges": {
            "type": "array",
            "description": "For kind=diagram: a flowchart/decision tree as arrows between short labels (max 12 arrows).",
            "items": {"type": "object", "properties": {
                "from": {"type": "string"}, "to": {"type": "string"}, "label": {"type": "string"}},
                "required": ["from", "to"]},
        },
        "caption": {"type": "string"},
    },
    "required": ["kind"],
}

GUIDE_TOOL = {
    "name": "write_study_guide",
    "description": "Write the finished study guide for this session.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "overview": {"type": "string", "description": "3-5 sentences: what the lesson is about and why it matters."},
            "sections": {
                "type": "array",
                "description": "Topics in a logical teaching order (foundations first), not the order questions were asked.",
                "items": {"type": "object", "properties": {
                    "heading": {"type": "string"},
                    "body": {"type": "string", "description": "Explanation in Markdown (paragraphs, - bullets, 1. steps, **bold**) with LaTeX $...$ inline and $$...$$ on its own line."},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                    "cues": {"type": "array", "items": {"type": "string"}, "description": "2-5 short recall questions or terms for a Cornell cue column."},
                    "summary": {"type": "string", "description": "1-2 sentence summary of this topic."},
                    "simple": {"type": "string", "description": "Optional: the idea explained very simply, as to a friend."},
                    "sources": {"type": "array", "items": {"type": "string"}, "description": "Which session items this came from, e.g. ['Q2', 'N4']."},
                    "connects_to": {"type": "string"},
                    "visual": GUIDE_VISUAL,
                }, "required": ["heading", "body", "key_points", "cues", "summary"]},
            },
            "formula_sheet": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "latex": {"type": "string", "description": "Bare LaTeX, no $ signs. Use only standard commands (frac, sqrt, bar, hat, sum, Greek letters)."},
                "when_to_use": {"type": "string"}}, "required": ["name", "latex", "when_to_use"]}},
            "concept_map": {"type": "array", "description": "Arrows linking the lesson's main ideas (max 14). Short labels.",
                            "items": {"type": "object", "properties": {
                                "from": {"type": "string"}, "to": {"type": "string"}, "label": {"type": "string"}},
                                "required": ["from", "to"]}},
            "common_mistakes": {"type": "array", "items": {"type": "string"}},
            "memory_tricks": {"type": "array", "items": {"type": "string"}},
            "practice_questions": {"type": "array", "items": {"type": "object", "properties": {
                "question": {"type": "string"}, "answer": {"type": "string", "description": "Full worked answer."},
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                "sources": {"type": "array", "items": {"type": "string"}}},
                "required": ["question", "answer", "difficulty"]}},
        },
        "required": ["title", "overview", "sections", "practice_questions"],
    },
}

VERIFY_TOOL = {
    "name": "report_unsupported",
    "description": "List statements in the guide that are NOT supported by the session material.",
    "input_schema": {"type": "object", "properties": {"issues": {"type": "array", "items": {"type": "object", "properties": {
        "where": {"type": "string", "description": "Section heading or 'Practice Q3' etc."},
        "statement": {"type": "string"},
        "why": {"type": "string"}}, "required": ["where", "statement", "why"]}}}, "required": ["issues"]},
}


def style_text(style: dict) -> str:
    s = {**DEFAULT_STYLE, **(style or {})}
    lines = [
        f"- Layout: {s['layout']}. {LAYOUTS.get(s['layout'], '')}",
        f"- Order inside each topic: {s['order']}.",
        f"- Visuals: {s['visuals']} (Heavy = a chart or diagram for most topics; Balanced = where it clearly helps; Minimal = only if essential).",
        f"- Length: {s['length']}.",
        f"- Practice questions: about {s['practice_count']}, difficulty {s['difficulty']}.",
        f"- Include: {', '.join(s['extras']) or 'nothing extra'}.",
    ]
    if s.get("custom", "").strip():
        lines.append(f"- The student's own instructions (follow these): {s['custom'].strip()}")
    return "\n".join(lines)


def guide_system(meta: dict, style: dict | None = None) -> str:
    return f"""You turn a student's study session into a study guide they will read and write on (as a PDF) on an iPad.

SESSION
{_context(meta)}

THE STUDENT'S STUDY STYLE
{style_text(style)}

STRICT RULES
- Use ONLY what is in the session material below (numbered notes N1.. and questions Q1..).
  Do not add topics, facts, formulas or examples that were not covered. You may add a short
  sentence linking two ideas that were both covered.
- If the material is thin on a topic, keep that topic short rather than inventing content.
- Practice questions must test only what was covered, with full worked answers.
- For every section and practice question, list the source ids it came from.
- Teach in a logical order, foundations first. Write clearly for revision.
- Maths: LaTeX with $...$ inline and $$...$$ on its own line. Stick to standard commands.
- Do not use emoji."""


def guide_input(meta: dict, notes: list[dict], history: list[dict]) -> str:
    lines = ["SESSION MATERIAL", "", "SAVED NOTES"]
    for i, n in enumerate(notes, 1):
        lines.append(f"N{i} [{n['type']}, importance {n.get('importance', 1)}] {n['title']}: {n['content']}"
                     + (f"  (formula: {n['latex']})" if n.get("latex") else ""))
    if not notes:
        lines.append("(none)")
    lines.append("\nQUESTIONS THE STUDENT ASKED, WITH THE TUTOR'S ANSWERS")
    q = 0
    for i, m in enumerate(history):
        if m["role"] == "user":
            q += 1
            ans = history[i + 1]["text"] if i + 1 < len(history) else ""
            shot = " [sent a screenshot]" if m.get("images") else ""
            lines.append(f"Q{q}{shot}: {m.get('text') or '(screenshot)'}\nA{q}: {ans[:2500]}\n")
    if q == 0:
        lines.append("(none)")
    return "\n".join(lines)


def _as_list(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return []
    return [x for x in (v or []) if x]


def _clean_guide(g: dict) -> dict:
    for key in ("sections", "formula_sheet", "concept_map", "common_mistakes", "memory_tricks", "practice_questions"):
        g[key] = _as_list(g.get(key))
    g["sections"] = [s for s in g["sections"] if isinstance(s, dict) and s.get("heading")]
    return g


class GuideMixin:
    """Export features shared by every provider. Each provider supplies _force_tool()."""

    def build_study_guide(self, meta, notes, history, model, style=None, deep=True) -> dict:
        data = self._force_tool(guide_system(meta, style), guide_input(meta, notes, history),
                                GUIDE_TOOL, model, 16000, deep)
        if not data or not data.get("sections"):
            raise RuntimeError("The AI didn't return a study guide. Try again.")
        return _clean_guide(data)

    def verify_guide(self, meta, notes, history, guide, model) -> list[dict]:
        text = (guide_input(meta, notes, history) + "\n\nSTUDY GUIDE TO CHECK\n"
                + json.dumps(guide, ensure_ascii=False)[:60000])
        system = ("You fact-check a study guide against the session it was made from. Report only "
                  "statements (facts, formulas, numbers, examples) that are NOT supported by the session "
                  "material. Linking sentences between covered ideas are fine. Return an empty list if all is supported.")
        data = self._force_tool(system, text, VERIFY_TOOL, model, 4000, False)
        return _as_list((data or {}).get("issues"))


class StudyAI(GuideMixin):
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def _force_tool(self, system, text, tool, model, max_tokens, deep):
        resp = self.client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            tools=[tool], tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": text}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return dict(block.input)
        return None

    def stream_answer(self, history: list[dict], meta: dict, model: str,
                      strict: bool, style: str) -> Iterator[str]:
        with self.client.messages.stream(
            model=model,
            max_tokens=4000,
            system=tutor_system(meta, strict, style),
            messages=to_api_messages(history),
        ) as stream:
            for text in stream.text_stream:
                yield text

    def extract_notes(self, question: str, had_images: bool, answer: str,
                      meta: dict, model: str, already: list[str]) -> list[dict]:
        exchange = (
            f"STUDENT{' (sent a screenshot)' if had_images else ''}:\n{question or '(screenshot only)'}"
            f"\n\nTUTOR:\n{answer}"
        )
        resp = self.client.messages.create(
            model=model,
            max_tokens=3000,
            system=notes_system(meta, already),
            tools=[NOTES_TOOL],
            tool_choice={"type": "tool", "name": "record_study_notes"},
            messages=[{"role": "user", "content": exchange}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                items = block.input.get("items", [])
                if isinstance(items, str):  # rare: model returns JSON as a string
                    items = json.loads(items)
                return [i for i in items if i.get("type") in ITEM_TYPES]
        return []


OPENAI_TUTOR_MODELS = ["gpt-6.1-terra", "gpt-6.1-sol", "gpt-6-luna", "gpt-5.6-terra"]
OPENAI_NOTES_MODELS = ["gpt-6-luna", "gpt-6.1-terra", "gpt-5.6-luna"]
OPENAI_EXPORT_MODELS = ["gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-astra", "gpt-5.6-sol"]


class OpenAIStudyAI(GuideMixin):
    """Same tutor + note-taker, running on OpenAI's Responses API."""

    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _input(history: list[dict]) -> list[dict]:
        out = []
        for m in to_api_messages(history):
            if m["role"] == "assistant":
                out.append({"role": "assistant", "content": m["content"]})
                continue
            parts = []
            for block in m["content"]:
                if block["type"] == "image":
                    src = block["source"]
                    parts.append({"type": "input_image", "detail": "high",
                                  "image_url": f"data:{src['media_type']};base64,{src['data']}"})
                else:
                    parts.append({"type": "input_text", "text": block["text"]})
            out.append({"role": "user", "content": parts})
        return out

    def stream_answer(self, history: list[dict], meta: dict, model: str,
                      strict: bool, style: str) -> Iterator[str]:
        stream = self.client.responses.create(
            model=model,
            instructions=tutor_system(meta, strict, style),
            input=self._input(history),
            max_output_tokens=6000,
            stream=True,
        )
        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta
            elif event.type in ("response.failed", "error"):
                raise RuntimeError(getattr(getattr(event, "response", None), "error", None) or getattr(event, "message", "OpenAI error"))

    def extract_notes(self, question: str, had_images: bool, answer: str,
                      meta: dict, model: str, already: list[str]) -> list[dict]:
        exchange = (
            f"STUDENT{' (sent a screenshot)' if had_images else ''}:\n{question or '(screenshot only)'}"
            f"\n\nTUTOR:\n{answer}"
        )
        resp = self.client.responses.create(
            model=model,
            instructions=notes_system(meta, already),
            input=exchange,
            tools=[{"type": "function", "name": NOTES_TOOL["name"], "description": NOTES_TOOL["description"],
                    "parameters": NOTES_TOOL["input_schema"], "strict": False}],
            tool_choice={"type": "function", "name": NOTES_TOOL["name"]},
            max_output_tokens=6000,
        )
        for item in resp.output:
            if item.type == "function_call":
                items = json.loads(item.arguments or "{}").get("items", [])
                return [i for i in items if isinstance(i, dict) and i.get("type") in ITEM_TYPES]
        return []

    def _force_tool(self, system, text, tool, model, max_tokens, deep):
        kwargs = dict(
            model=model, instructions=system, input=text,
            tools=[{"type": "function", "name": tool["name"], "description": tool["description"],
                    "parameters": tool["input_schema"], "strict": False}],
            tool_choice={"type": "function", "name": tool["name"]},
            max_output_tokens=max_tokens + (16000 if deep else 4000),  # room for reasoning
        )
        if deep:
            kwargs["reasoning"] = {"effort": "high"}
        try:
            resp = self.client.responses.create(**kwargs)
        except Exception as e:  # model without adjustable reasoning: retry without it
            if deep and "reasoning" in str(e).lower():
                kwargs.pop("reasoning")
                resp = self.client.responses.create(**kwargs)
            else:
                raise
        for item in resp.output:
            if item.type == "function_call":
                return json.loads(item.arguments or "{}")
        return None


class DemoAI:
    """Canned answers so you can try the app with no API key. Nothing is sent anywhere."""

    ANSWER = (
        "*(Demo mode: this is a sample answer, not a real reply to your question.)*\n\n"
        "A **z-score** tells you how many standard deviations a value is from the mean:\n\n"
        "$$z = \\frac{x - \\mu}{\\sigma}$$\n\n"
        "**Example:** test scores have $\\mu = 70$ and $\\sigma = 8$. A score of 86 gives "
        "$z = (86-70)/8 = 2$, so it's 2 standard deviations above average.\n\n"
        "About 95% of values fall between $z = -1.96$ and $z = 1.96$.\n\n"
        "**Takeaway:** standardise first, then use the normal table."
    )
    NOTES = [
        {"type": "definition", "title": "Z-score", "importance": 3,
         "content": "How many standard deviations a value $x$ lies from the mean."},
        {"type": "formula", "title": "Z-score formula", "importance": 3, "latex": "z = \\frac{x-\\mu}{\\sigma}",
         "content": "Subtract the mean, divide by the standard deviation.",
         "visual": {"kind": "chart", "chart_type": "normal", "caption": "95% of values lie within ±1.96",
                    "params": {"mu": 0, "sigma": 1, "shade_from": -1.96, "shade_to": 1.96, "title": "Standard normal"}}},
        {"type": "example", "title": "Test score z-score", "importance": 2,
         "content": "$\\mu=70,\\ \\sigma=8,\\ x=86 \\Rightarrow z = 2$ (2 SDs above average)."},
        {"type": "procedure", "title": "Choosing z vs t", "importance": 2,
         "content": "1. Is σ known? Use z.\n2. Unknown σ, small n? Use t with n−1 df.",
         "visual": {"kind": "diagram", "mermaid": "flowchart TD\n  A[Population σ known?] -->|Yes| Z[z-test]\n  A -->|No| B[n ≥ 30?]\n  B -->|Yes| Z2[z ≈ t, either works]\n  B -->|No| T[t-test, df = n-1]"}},
        {"type": "tip", "title": "Sketch the curve first", "importance": 1,
         "content": "Shade the area you want before using the table, so you can catch sign mistakes."},
    ]

    def __init__(self, *_):
        self.calls = 0

    def stream_answer(self, *_, **__):
        import time
        for word in self.ANSWER.split(" "):
            time.sleep(0.015)
            yield word + " "

    def extract_notes(self, *_, **__):
        import copy
        return copy.deepcopy(self.NOTES)

    def build_study_guide(self, meta: dict, *_, **__) -> dict:
        return _clean_guide({
            "title": f"{meta.get('subject') or 'Statistics'}: z-scores and the normal curve",
            "overview": "(Demo mode: a sample guide built without the AI.) A z-score puts any value on a common "
                        "scale: how many standard deviations it sits from the mean. It is the bridge from raw data "
                        "to the standard normal table, and it comes back later in confidence intervals and tests.",
            "sections": [
                {"heading": "What a z-score measures",
                 "body": "Subtract the mean, then divide by the standard deviation:\n\n$$z = \\frac{x-\\mu}{\\sigma}$$\n\n"
                         "**Example:** test scores have $\\mu = 70$ and $\\sigma = 8$. A score of $x = 86$ gives "
                         "$z = (86-70)/8 = 2$, so it is 2 standard deviations above average.",
                 "key_points": ["Positive z means above the mean", "Negative z means below the mean",
                                "Units cancel, so you can compare different tests"],
                 "cues": ["What does z tell you?", "Formula for z?", "What does z = -1 mean?"],
                 "summary": "A z-score is the distance from the mean measured in standard deviations.",
                 "simple": "It answers: how unusual is this value compared with everyone else?",
                 "sources": ["Q1", "N1", "N2"],
                 "connects_to": "Once you have z, the normal curve tells you how likely that value is.",
                 "visual": {"kind": "chart", "chart_type": "normal", "caption": "Standard normal curve: middle 95% shaded",
                            "params": {"mu": 0, "sigma": 1, "shade_from": -1.96, "shade_to": 1.96, "xlabel": "z"}}},
                {"heading": "Choosing z or t",
                 "body": "Use **z** when the population standard deviation $\\sigma$ is known. If it is unknown, "
                         "use **t** with $n-1$ degrees of freedom.\n\n1. Check whether $\\sigma$ is given.\n2. If not, use the sample $s$ and the t table.",
                 "key_points": ["sigma known: z", "sigma unknown: t with n-1 df"],
                 "cues": ["When do you use t instead of z?", "Degrees of freedom for one sample?"],
                 "summary": "Known population spread means z; otherwise use t.",
                 "sources": ["N4"],
                 "visual": {"kind": "diagram", "caption": "Decision: z or t?", "edges": [
                     {"from": "Is sigma known?", "to": "Use z", "label": "yes"},
                     {"from": "Is sigma known?", "to": "Use t, df = n-1", "label": "no"}]}},
            ],
            "formula_sheet": [{"name": "Z-score", "latex": "z = \\frac{x-\\mu}{\\sigma}", "when_to_use": "Standardising a single value"}],
            "concept_map": [{"from": "Raw value", "to": "z-score", "label": "standardise"},
                            {"from": "z-score", "to": "Normal table", "label": "look up"},
                            {"from": "Normal table", "to": "Probability"}],
            "common_mistakes": ["Dividing by the variance instead of the standard deviation",
                                "Dropping the minus sign for values below the mean"],
            "memory_tricks": ["z is the 'zoom' number: how many SD-steps away you are"],
            "practice_questions": [
                {"question": "Heights have $\\mu = 170$ cm and $\\sigma = 10$ cm. Find the z-score of 185 cm.",
                 "answer": "$z = (185-170)/10 = 1.5$, so 1.5 standard deviations above the mean.", "difficulty": "easy", "sources": ["N2"]},
                {"question": "What proportion of a normal distribution lies above $z = 1.96$?",
                 "answer": "About 2.5%: 95% lies between -1.96 and 1.96, and the two tails share the other 5%.",
                 "difficulty": "medium", "sources": ["Q1"]},
                {"question": "A sample of 12 has unknown $\\sigma$. Which distribution do you use, and with how many degrees of freedom?",
                 "answer": "The t distribution with $12 - 1 = 11$ degrees of freedom.", "difficulty": "medium", "sources": ["N4"]},
            ],
        })

    def verify_guide(self, *_, **__) -> list[dict]:
        return []


def to_api_messages(history: list[dict], keep_images_for: int = 3) -> list[dict]:
    """Session history -> Anthropic message format (images as base64 blocks).

    Only the screenshots from the last few student turns are re-sent, which keeps
    long sessions fast and cheap; older ones become a short placeholder.
    """
    user_turns = [i for i, m in enumerate(history) if m["role"] == "user"]
    recent = set(user_turns[-keep_images_for:])
    out = []
    for idx, m in enumerate(history):
        if m["role"] == "user":
            imgs = m.get("images", [])
            text = m.get("text") or "Please help me with this screenshot."
            if imgs and idx not in recent:
                imgs, text = [], f"[{len(m['images'])} screenshot(s) shared earlier]\n{text}"
            content = [
                {"type": "image", "source": {"type": "base64", "media_type": img["media_type"], "data": img["data"]}}
                for img in imgs
            ]
            content.append({"type": "text", "text": text})
            out.append({"role": "user", "content": content})
        else:
            out.append({"role": "assistant", "content": m["text"]})
    return out