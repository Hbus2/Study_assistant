# StudyFlow

A live study session: ask questions, paste screenshots, and your study notes build themselves
as you go. When you're done, send them to Notion and study on your iPad.

## Setup (5 minutes)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then fill in your keys
streamlit run app.py
```

**Anthropic API key:** console.anthropic.com → API Keys. This is billed separately from a Claude.ai subscription.

**Notion (optional):**
1. Go to notion.so/profile/integrations → *New integration* → copy the token into `NOTION_TOKEN`.
2. In Notion, make a page called e.g. "Study Notes". Open it, click `•••` → *Connections* → add your integration.
3. Copy that page's link into `NOTION_PARENT_PAGE`.

Each session becomes its own page under it. Syncing again updates that page instead of making a copy.

## How a session works

1. Fill in **Subject / Course / Chapter / Lesson** in the sidebar. These are sent with every question, so the tutor stays on topic.
2. Ask questions. For a screenshot, take it (Win+Shift+S / Cmd+Ctrl+Shift+4) and click **📋 Paste screenshot**, or drop the image into the chat box. Nothing needs saving to disk.
3. After every answer a second AI pass (the note-taker) pulls out definitions, formulas, concepts, procedures, examples and exam tips, and attaches a chart or flowchart when one helps.
4. You decide what gets kept via **Saving notes**:
   - *Ask me before saving*: every item shows up with Save / Skip.
   - *Auto-save core, ask about the rest* (default): ⭐⭐⭐ items save automatically, the rest wait for you.
   - *Auto-save everything*, or *Off* (use 📌 on any answer to pin it).
5. **Sync to Notion** or download Markdown.

## Where the "control" lives

| What | Where |
|---|---|
| What the tutor does, how strict it is about staying on subject | `tutor_system()` in `ai.py` |
| What counts as a note, and the note types | `NOTES_TOOL` schema + `notes_system()` in `ai.py` |
| Which charts can be drawn | `visuals.py`. The AI only picks a chart type and fills in numbers; it never runs code |
| Notion page layout | `notion_sync.py` |

Sessions save automatically to `sessions/` as JSON, and you can reopen them from the sidebar.

## Ideas for next steps
- **Flashcards / quiz mode** generated from the saved notes (spaced repetition).
- A **Notion database** (one row per note, with Subject/Chapter/Type properties) so you can filter across all sessions.
- **Deploy** to Streamlit Community Cloud so it runs on the iPad too (put keys in *Secrets*).
- A global hotkey screenshot helper so a capture goes straight into the current session.
