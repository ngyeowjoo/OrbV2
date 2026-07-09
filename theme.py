"""
theme.py  —  Orb v2 shared theme system

All pages import from here. One source of truth for colours and CSS injection.

Usage:
    from theme import apply_theme, AMBER, get_colours

    apply_theme()          # injects CSS for current mode into the page
    c = get_colours()      # returns dict of current colour values
    AMBER                  # always #D97706 (accent, never changes)
"""
import streamlit as st

AMBER  = "#D97706"

# ── Palette definitions ────────────────────────────────────────────────────────
_THEMES = {
    "light": {
        "BG":           "#F9FAFB",
        "CARD":         "#FFFFFF",
        "BORDER":       "#E5E7EB",
        "TEXT":         "#111827",
        "SUBTEXT":      "#6B7280",
        "USERBG":       "#F3F4F6",
        "SB_BG":        "#111827",
        "SB_TEXT":      "#F9FAFB",
        "SB_SUB":       "#9CA3AF",
        "SB_HVR":       "#1F2937",
        "SB_ACT":       "#374151",
        "INPUT_BG":     "#FFFFFF",
        "SCROLLTRACK":  "#f1f1f1",
        "SCROLLTHUMB":  "#d1d5db",
        "AMBERL":       "#FEF3C7",
        "CODE_BG":      "#F3F4F6",
        "CODE_TEXT":    "#111827",
        "GRID":         "#E5E7EB",
        "PLACEHOLDER":  "#9CA3AF",
        "DEL_BTN":      "#4B5563",
        "SB_BORDER":    "#1F2937",
        "SIGNOUT_BORDER": "#374151",
    },
    "dark": {
        "BG":           "#0F1117",
        "CARD":         "#1A1D27",
        "BORDER":       "#2D3143",
        "TEXT":         "#E5E7EB",
        "SUBTEXT":      "#9CA3AF",
        "USERBG":       "#252836",
        "SB_BG":        "#0A0C14",
        "SB_TEXT":      "#E5E7EB",
        "SB_SUB":       "#6B7280",
        "SB_HVR":       "#1A1D27",
        "SB_ACT":       "#252836",
        "INPUT_BG":     "#1A1D27",
        "SCROLLTRACK":  "#1A1D27",
        "SCROLLTHUMB":  "#374151",
        "AMBERL":       "#451A03",
        "CODE_BG":      "#0D1117",
        "CODE_TEXT":    "#E5E7EB",
        "GRID":         "#2D3143",
        "PLACEHOLDER":  "#4B5563",
        "DEL_BTN":      "#374151",
        "SB_BORDER":    "#0A0C14",
        "SIGNOUT_BORDER": "#2D3143",
    },
}


def get_mode() -> str:
    """Returns 'light' or 'dark'. Reads from session_state; default light."""
    return st.session_state.get("theme_mode", "light")


def get_colours() -> dict:
    """Returns the full colour dict for the current theme."""
    return _THEMES[get_mode()]


def toggle():
    """Flip between light and dark, then rerun."""
    st.session_state["theme_mode"] = "dark" if get_mode() == "light" else "light"
    st.rerun()


def apply_theme(page: str = "main"):
    """
    Injects the full themed CSS for the current mode.
    Call once at the top of each page after st.set_page_config().

    page: "main" | "panel" | "config"
    "main"   — full chat UI (sidebar, chat bubbles, chips, form)
    "panel"  — debug panel styling
    "config" — config page (form fields, textareas, save buttons)
    """
    c    = get_colours()
    dark = get_mode() == "dark"

    # ── Shared base CSS (all pages) ──────────────────────────────────────────
    base = f"""
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; }}
html, body, .stApp {{ background: {c["BG"]} !important; color: {c["TEXT"]}; }}
#MainMenu, footer {{ display: none !important; }}
.block-container {{ max-width: 100% !important; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {c["SB_BG"]} !important;
    border-right: 1px solid {c["SB_BORDER"]} !important;
}}
section[data-testid="stSidebar"] > div {{ background: {c["SB_BG"]} !important; }}
section[data-testid="stSidebar"] * {{ color: {c["SB_TEXT"]} !important; }}
section[data-testid="stSidebar"] .stButton > button {{
    background: transparent !important; border: none !important;
    color: {c["SB_TEXT"]} !important; font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 400 !important;
    padding: 8px 12px !important; border-radius: 8px !important;
    text-align: left !important; width: 100% !important; box-shadow: none !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: {c["SB_HVR"]} !important; color: {c["SB_TEXT"]} !important; border: none !important;
}}
section[data-testid="stSidebar"] .new-chat-btn > button {{
    background: rgba(217,119,6,0.18) !important;
    border: 1px solid rgba(217,119,6,0.35) !important;
    color: {AMBER} !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.84rem !important;
}}
section[data-testid="stSidebar"] .new-chat-btn > button:hover {{
    background: rgba(217,119,6,0.28) !important;
}}
section[data-testid="stSidebar"] .signout-btn > button {{
    background: transparent !important;
    border: 1px solid {c["SIGNOUT_BORDER"]} !important;
    color: {c["SB_SUB"]} !important; border-radius: 8px !important; font-size: 0.80rem !important;
}}
section[data-testid="stSidebar"] .signout-btn > button:hover {{
    border-color: #EF4444 !important; color: #FCA5A5 !important;
    background: rgba(239,68,68,0.08) !important;
}}
section[data-testid="stSidebar"] .del-btn > button {{
    background: transparent !important; border: none !important;
    color: {c["DEL_BTN"]} !important; font-size: 0.70rem !important;
    padding: 3px 6px !important; border-radius: 4px !important;
    box-shadow: none !important; min-width: 0 !important;
}}
section[data-testid="stSidebar"] .del-btn > button:hover {{
    color: #FCA5A5 !important; background: rgba(239,68,68,0.1) !important;
}}
section[data-testid="stSidebar"] .theme-toggle > button {{
    background: {'rgba(255,255,255,0.06)' if dark else 'rgba(0,0,0,0.04)'} !important;
    border: 1px solid {c["BORDER"]} !important;
    color: {c["SB_TEXT"]} !important; border-radius: 20px !important;
    font-size: 0.78rem !important; font-weight: 500 !important;
    padding: 5px 14px !important; width: 100% !important;
    transition: all 0.2s !important;
}}
section[data-testid="stSidebar"] .theme-toggle > button:hover {{
    border-color: {AMBER} !important; color: {AMBER} !important;
    background: rgba(217,119,6,0.12) !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: {c["SCROLLTRACK"]}; }}
::-webkit-scrollbar-thumb {{ background: {c["SCROLLTHUMB"]}; border-radius: 2px; }}

/* ── Inputs ── */
.stTextInput > div > div > input {{
    background: {c["INPUT_BG"]} !important; border: 1px solid {c["BORDER"]} !important;
    border-radius: 10px !important; color: {c["TEXT"]} !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.92rem !important;
    padding: 12px 16px !important; box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: {AMBER} !important; box-shadow: 0 0 0 3px rgba(217,119,6,0.1) !important;
}}
.stTextInput > div > div > input::placeholder {{ color: {c["PLACEHOLDER"]} !important; }}
.stTextInput label {{ display: none !important; }}

.stNumberInput > div > div > input {{
    background: {c["INPUT_BG"]} !important; border: 1px solid {c["BORDER"]} !important;
    border-radius: 8px !important; color: {c["TEXT"]} !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.84rem !important;
}}
.stTextArea > div > div > textarea {{
    background: {c["INPUT_BG"]} !important; border: 1px solid {c["BORDER"]} !important;
    border-radius: 8px !important; color: {c["TEXT"]} !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.80rem !important;
}}

/* ── Buttons ── */
.stButton > button, [data-testid="stFormSubmitButton"] > button {{
    background: {c["CARD"]} !important; border: 1px solid {c["BORDER"]} !important;
    border-radius: 8px !important; color: {c["SUBTEXT"]} !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important;
    font-weight: 500 !important; padding: 7px 14px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important; transition: all 0.15s !important;
}}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
    border-color: {AMBER} !important; color: {AMBER} !important;
    background: {c["AMBERL"]} !important;
}}
.send-btn > div > button, .send-btn [data-testid="stFormSubmitButton"] > button {{
    background: {AMBER} !important; border: none !important;
    border-radius: 8px !important; color: #fff !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.85rem !important;
    font-weight: 600 !important; padding: 9px 22px !important;
    box-shadow: 0 2px 8px rgba(217,119,6,0.28) !important;
}}
.send-btn > div > button:hover, .send-btn [data-testid="stFormSubmitButton"] > button:hover {{
    background: #B45309 !important; color: #fff !important;
}}
.save-btn > div > button {{
    background: {AMBER} !important; border: none !important;
    border-radius: 8px !important; color: #fff !important;
    font-weight: 600 !important; padding: 8px 20px !important;
}}
.save-btn > div > button:hover {{ background: #B45309 !important; }}

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div > div {{
    background: {c["CARD"]} !important; border: 1px solid {c["BORDER"]} !important;
    border-radius: 8px !important; font-family: 'Inter', sans-serif !important;
    font-size: 0.84rem !important; color: {c["TEXT"]} !important;
}}
div[data-testid="stSelectbox"] label {{
    font-size: 0.70rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important;
    color: {c["SUBTEXT"]} !important;
}}
/* Selectbox dropdown popup */
div[data-baseweb="popover"] ul {{
    background: {c["CARD"]} !important; border: 1px solid {c["BORDER"]} !important;
}}
div[data-baseweb="popover"] li {{
    color: {c["TEXT"]} !important;
}}
div[data-baseweb="popover"] li:hover {{
    background: {c["USERBG"]} !important;
}}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: {c["CARD"]}; border: 1px solid {c["BORDER"]}; border-radius: 10px;
    padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}
[data-testid="stMetricLabel"] {{ color: {c["SUBTEXT"]} !important; font-size: 0.75rem !important; }}
[data-testid="stMetricValue"] {{ color: {AMBER} !important; font-size: 1.4rem !important; font-weight:700 !important; }}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {c["BORDER"]} !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}}
/* Target only the outer frame wrapper, not the internal canvas */
[data-testid="stDataFrame"] > div {{
    background: {c["CARD"]} !important;
}}
/* Dataframe toolbar */
[data-testid="stDataFrameResizable"] {{
    background: {c["CARD"]} !important;
}}
div[data-testid="stForm"] {{ border: none !important; padding: 0 !important; }}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: {c["CARD"]} !important; border: 1px solid {c["BORDER"]} !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary {{ color: {c["TEXT"]} !important; }}
[data-testid="stExpander"] summary svg {{ fill: {c["SUBTEXT"]} !important; }}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: {c["CARD"]} !important; border-bottom: 1px solid {c["BORDER"]} !important;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    color: {c["SUBTEXT"]} !important; font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    color: {AMBER} !important;
    border-bottom: 2px solid {AMBER} !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {{
    background: {c["BG"]} !important; padding-top: 16px !important;
}}

/* ── Code blocks ── */
.stCodeBlock, pre, code {{
    background: {c["CODE_BG"]} !important;
    color: {c["CODE_TEXT"]} !important;
    border: 1px solid {c["BORDER"]} !important;
    border-radius: 6px !important;
}}

/* ── Labels ── */
label {{
    font-family: 'Inter', sans-serif !important; color: {c["SUBTEXT"]} !important;
    font-size: 0.76rem !important; font-weight: 600 !important;
}}

/* ── Caption / small text ── */
.stCaption, [data-testid="stCaptionContainer"] {{
    color: {c["SUBTEXT"]} !important;
}}

/* ── Alert / info boxes ── */
[data-testid="stInfo"] {{
    background: {'rgba(217,119,6,0.08)' if dark else '#FEF9EC'} !important;
    border: 1px solid rgba(217,119,6,0.3) !important;
    color: {c["TEXT"]} !important;
}}
[data-testid="stWarning"] {{
    background: {'rgba(239,68,68,0.08)' if dark else '#FEF2F2'} !important;
    color: {c["TEXT"]} !important;
}}
[data-testid="stSuccess"] {{
    background: {'rgba(39,174,96,0.10)' if dark else '#F0FDF4'} !important;
    color: {c["TEXT"]} !important;
}}
"""

    # ── Chat page extras ─────────────────────────────────────────────────────
    chat_css = f"""
.block-container {{ padding: 0 1rem !important; }}

/* Suggestion / clarification cards */
.sug-card > button {{
    background: {c["CARD"]} !important; border: 1px solid {c["BORDER"]} !important;
    border-radius: 10px !important; color: {c["TEXT"]} !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.76rem !important;
    font-weight: 400 !important; padding: 10px 12px !important;
    text-align: center !important; min-height: 46px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important; line-height: 1.35 !important;
}}
.sug-card > button:hover {{
    border-color: {AMBER} !important; background: {c["AMBERL"]} !important; color: {AMBER} !important;
}}

/* Thinking indicator */
@keyframes orbPulse {{
    0%, 100% {{ opacity: 0.3; transform: scale(0.85); }}
    50%      {{ opacity: 1;   transform: scale(1.05); }}
}}
.thinking-dot {{
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: {AMBER}; margin: 0 2px; animation: orbPulse 1.2s ease-in-out infinite;
}}
.thinking-dot:nth-child(2) {{ animation-delay: 0.2s; }}
.thinking-dot:nth-child(3) {{ animation-delay: 0.4s; }}
"""

    # ── Config / panel page extras ───────────────────────────────────────────
    panel_css = f"""
.block-container {{ padding: 1.5rem 2rem !important; }}
"""

    extra = chat_css if page == "main" else panel_css

    st.markdown(f"<style>{base}{extra}</style>", unsafe_allow_html=True)


def render_toggle(key_suffix: str = ""):
    """
    Renders the ☾/☀ toggle button.
    Can be called from any page's sidebar.
    key_suffix prevents duplicate widget keys across pages.
    """
    mode  = get_mode()
    label = "☾  Dark mode" if mode == "light" else "☀  Light mode"
    st.markdown('<div class="theme-toggle">', unsafe_allow_html=True)
    if st.button(label, key=f"theme_toggle_{key_suffix}", use_container_width=True):
        toggle()
    st.markdown('</div>', unsafe_allow_html=True)
