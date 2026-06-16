"""
chat_store.py  —  Persistent recent chats, including chart/table panels.
Stores up to MAX_CHATS sessions as JSON files in .chat_history/
Each session: { id, title, user, timestamp, messages, model, panels }
"""
import json, uuid, io
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.io as pio

STORE_DIR = Path(__file__).parent / ".chat_history"
MAX_CHATS = 20

def _ensure_dir():
    STORE_DIR.mkdir(exist_ok=True)

def _all_files():
    _ensure_dir()
    return sorted(STORE_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

# ── PANEL SERIALISATION ────────────────────────────────────────────────────
def _serialize_panels(panels):
    out = []
    for p in panels or []:
        item = {"label": p.get("label", "")}
        chart = p.get("chart")
        df    = p.get("df")
        try:
            item["chart"] = pio.to_json(chart) if chart is not None else None
        except Exception:
            item["chart"] = None
        try:
            item["df"] = df.to_json(orient="split", date_format="iso") if df is not None else None
        except Exception:
            item["df"] = None
        out.append(item)
    return out

def _deserialize_panels(panels):
    out = []
    for p in panels or []:
        item = {"label": p.get("label", "")}
        chart_json = p.get("chart")
        df_json    = p.get("df")
        try:
            item["chart"] = pio.from_json(chart_json) if chart_json else None
        except Exception:
            item["chart"] = None
        try:
            item["df"] = pd.read_json(io.StringIO(df_json), orient="split") if df_json else None
        except Exception:
            item["df"] = None
        out.append(item)
    return out

# ── SAVE / LOAD / DELETE ──────────────────────────────────────────────────
def save_chat(user_id: str, messages: list, model_name: str, panels: list = None,
               session_id: str = None) -> str:
    """Save (or overwrite) a chat session. Returns session id."""
    if not messages:
        return ""
    _ensure_dir()
    first_user = next((m["content"] for m in messages if m["role"] == "user"), "Untitled")
    title = first_user[:60] + ("…" if len(first_user) > 60 else "")
    sid = session_id or str(uuid.uuid4())[:8]
    payload = {
        "id":        sid,
        "title":     title,
        "user_id":   user_id,
        "timestamp": datetime.now().isoformat(),
        "model":     model_name,
        "messages":  messages,
        "panels":    _serialize_panels(panels),
    }
    (STORE_DIR / f"{sid}.json").write_text(json.dumps(payload, ensure_ascii=False))
    for old in _all_files()[MAX_CHATS:]:
        old.unlink(missing_ok=True)
    return sid

def load_all(user_id: str) -> list:
    """Lightweight metadata list (no panels), newest first."""
    out = []
    for f in _all_files():
        try:
            data = json.loads(f.read_text())
            if data.get("user_id") == user_id:
                out.append({
                    "id":        data["id"],
                    "title":     data["title"],
                    "timestamp": data["timestamp"],
                    "model":     data.get("model", ""),
                    "msg_count": len([m for m in data["messages"] if m["role"] == "user"]),
                })
        except Exception:
            continue
    return out

def load_chat(session_id: str):
    """Full chat including deserialised panels (charts + dataframes)."""
    path = STORE_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        data["panels"] = _deserialize_panels(data.get("panels", []))
        return data
    except Exception:
        return None

def delete_chat(session_id: str):
    (STORE_DIR / f"{session_id}.json").unlink(missing_ok=True)

def fmt_ts(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%-d %b, %-I:%M %p")
    except Exception:
        return iso
