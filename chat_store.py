"""
chat_store.py  —  Persistent recent chats using Streamlit file-based storage.
Stores up to MAX_CHATS sessions as JSON files in .chat_history/
Each session: { id, title, user, timestamp, messages, model }
"""
import json, os, uuid
from datetime import datetime
from pathlib import Path

STORE_DIR = Path(__file__).parent / ".chat_history"
MAX_CHATS = 20

def _ensure_dir():
    STORE_DIR.mkdir(exist_ok=True)

def _all_files():
    _ensure_dir()
    return sorted(STORE_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

def save_chat(user_id: str, messages: list, model_name: str) -> str:
    """Save current chat. Returns session id."""
    if not messages:
        return ""
    _ensure_dir()
    # derive title from first user message
    first_user = next((m["content"] for m in messages if m["role"] == "user"), "Untitled")
    title = first_user[:60] + ("…" if len(first_user) > 60 else "")
    session_id = str(uuid.uuid4())[:8]
    payload = {
        "id":        session_id,
        "title":     title,
        "user_id":   user_id,
        "timestamp": datetime.now().isoformat(),
        "model":     model_name,
        "messages":  messages,
    }
    path = STORE_DIR / f"{session_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    # prune oldest beyond MAX_CHATS
    files = _all_files()
    for old in files[MAX_CHATS:]:
        old.unlink(missing_ok=True)
    return session_id

def load_all(user_id: str) -> list:
    """Return list of chat metadata (no messages) for this user, newest first."""
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

def load_chat(session_id: str) -> dict | None:
    """Load full chat by id. Returns dict or None."""
    path = STORE_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def delete_chat(session_id: str):
    path = STORE_DIR / f"{session_id}.json"
    path.unlink(missing_ok=True)

def fmt_ts(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%-d %b, %-I:%M %p")
    except Exception:
        return iso
