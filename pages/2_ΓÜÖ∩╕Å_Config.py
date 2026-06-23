"""
pages/2_⚙️_Config.py  —  Orb v2 Semantic Layer Config

View and edit the semantic layer (synonyms, thresholds, field aliases,
intent hints) without touching code. Changes are saved back to
semantic_layer.yaml and take effect on the next query.
"""
import streamlit as st
import yaml
from pathlib import Path

st.set_page_config(
    page_title="Orb v2 — Config",
    layout="wide",
    initial_sidebar_state="expanded",
)

AMBER   = "#D97706"
AMBERL  = "#FEF3C7"
BG      = "#F9FAFB"
CARD    = "#FFFFFF"
BORDER  = "#E5E7EB"
TEXT    = "#111827"
SUBTEXT = "#6B7280"
GREEN   = "#059669"
RED     = "#DC2626"
SB_BG   = "#111827"
SB_TEXT = "#F9FAFB"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
html, body, .stApp {{ background: {BG} !important; color: {TEXT}; }}
#MainMenu, footer {{ display: none !important; }}
section[data-testid="stSidebar"] {{ background: {SB_BG} !important; }}
section[data-testid="stSidebar"] * {{ color: {SB_TEXT} !important; }}
.block-container {{ padding: 1.5rem 2rem !important; max-width: 100% !important; }}
.stButton > button {{
    background: {CARD} !important; border: 1px solid {BORDER} !important;
    border-radius: 8px !important; color: {SUBTEXT} !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important;
    font-weight: 500 !important; padding: 6px 14px !important;
}}
.stButton > button:hover {{
    border-color: {AMBER} !important; color: {AMBER} !important; background: {AMBERL} !important;
}}
.save-btn > div > button {{
    background: {AMBER} !important; border: none !important;
    border-radius: 8px !important; color: #fff !important;
    font-weight: 600 !important; padding: 8px 20px !important;
}}
.save-btn > div > button:hover {{ background: #B45309 !important; }}
.stTextInput > div > div > input, .stNumberInput > div > div > input {{
    background: {CARD} !important; border: 1px solid {BORDER} !important;
    border-radius: 8px !important; color: {TEXT} !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.84rem !important;
}}
.stTextArea > div > div > textarea {{
    background: {CARD} !important; border: 1px solid {BORDER} !important;
    border-radius: 8px !important; color: {TEXT} !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.80rem !important;
}}
label {{ font-family: 'Inter', sans-serif !important; color: {SUBTEXT} !important;
         font-size: 0.76rem !important; font-weight: 600 !important; }}
.section-hdr {{
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: {SUBTEXT};
    border-bottom: 2px solid {AMBERL}; padding-bottom: 4px; margin: 20px 0 12px;
}}
</style>
""", unsafe_allow_html=True)

# ── Guard ─────────────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    st.warning("Please sign in to Orb v2 first.")
    st.page_link("app.py", label="← Go to Orb v2", icon="🔮")
    st.stop()

YAML_PATH = Path(__file__).parent.parent / "semantic_layer.yaml"

def load_yaml():
    try:
        return yaml.safe_load(YAML_PATH.read_text()) or {}
    except Exception as e:
        st.error(f"Could not load semantic_layer.yaml: {e}")
        return {}

def save_yaml(data: dict):
    try:
        YAML_PATH.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False))
        # Clear Streamlit cache so semantic.py picks up changes
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Could not save: {e}")
        return False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:8px;">
    <div style="width:32px;height:32px;border-radius:50%;
        background:radial-gradient(circle at 38% 35%,#fff8e0 0%,#F9A602 30%,#c97f00 60%,#3d1f00 100%);
        box-shadow:0 0 10px rgba(217,119,6,0.35);"></div>
    <div>
        <span style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:{TEXT};">
            Orb <span style="color:{AMBER};">v2</span>
        </span>
        <span style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.14em;
                     text-transform:uppercase;color:{SUBTEXT};margin-left:10px;">Semantic Layer Config</span>
    </div>
</div>
<p style="font-family:'Inter',sans-serif;font-size:0.82rem;color:{SUBTEXT};margin-bottom:24px;">
    Edit synonyms, thresholds, field aliases, and intent hints.
    Changes save to <code>semantic_layer.yaml</code> and take effect on the next query — no redeploy needed.
</p>
""", unsafe_allow_html=True)

cfg = load_yaml()

tabs = st.tabs(["📐 Thresholds", "🔤 Synonyms", "🏷️ Field Aliases", "💡 Intent Hints", "📄 Raw YAML"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown(f'<p class="section-hdr">Business Rule Thresholds</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <p style="font-family:'Inter',sans-serif;font-size:0.80rem;color:{SUBTEXT};margin-bottom:16px;">
    These values are used in retrieval logic and injected into every AI system prompt
    so responses are always consistent with your business definitions.</p>
    """, unsafe_allow_html=True)

    t = cfg.get("thresholds", {})
    changes = {}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<p class="section-hdr">Underperformance</p>', unsafe_allow_html=True)
        changes["consecutive_miss_min"] = st.number_input(
            "Consecutive miss threshold (cycles)", min_value=1, max_value=12,
            value=int(t.get("consecutive_miss_min", 3)),
            help="Employees missing targets for this many consecutive cycles or more are flagged")
        changes["consecutive_miss_severe"] = st.number_input(
            "Severe consecutive miss (cycles)", min_value=1, max_value=12,
            value=int(t.get("consecutive_miss_severe", 5)),
            help="Employees at or above this threshold get a critical severity flag")

        st.markdown(f'<p class="section-hdr">Payout Definitions</p>', unsafe_allow_html=True)
        changes["low_payout_pct"] = st.number_input(
            "Low payout — below % of scheme max", min_value=0, max_value=100,
            value=int(t.get("low_payout_pct", 50)),
            help="Payout below this % of the scheme maximum is considered 'low'")
        changes["near_miss_pct"] = st.number_input(
            "Near miss — within % of scheme max", min_value=0, max_value=20,
            value=int(t.get("near_miss_pct", 5)),
            help="Employees within this % of max payout are flagged as near misses")
        changes["high_attainment_pct"] = st.number_input(
            "High attainment — % of scheme max or above", min_value=50, max_value=100,
            value=int(t.get("high_attainment_pct", 90)))
        changes["ranking_min_payout"] = st.number_input(
            "Exclude from rankings — payout at or below", min_value=-1,
            value=int(t.get("ranking_min_payout", 0)),
            help="Set to -1 to include zero-payout employees in rankings")
        changes["top_n_default"] = st.number_input(
            "Default top N (when not specified)", min_value=1, max_value=50,
            value=int(t.get("top_n_default", 10)))

    with col2:
        st.markdown(f'<p class="section-hdr">PMGM / Performance Ratings</p>', unsafe_allow_html=True)
        high_bands_default = "\n".join(t.get("high_pmgm_bands", ["Exceptional","Exceeds Expectations"]))
        high_bands_str = st.text_area(
            "High PMGM bands (one per line)", value=high_bands_default, height=100,
            help="These rating bands are considered 'high performers' in anomaly detection")
        changes["high_pmgm_bands"] = [b.strip() for b in high_bands_str.split("\n") if b.strip()]

        low_bands_default = "\n".join(t.get("low_pmgm_bands", ["Below Expectations","Unsatisfactory"]))
        low_bands_str = st.text_area(
            "Low PMGM bands (one per line)", value=low_bands_default, height=100)
        changes["low_pmgm_bands"] = [b.strip() for b in low_bands_str.split("\n") if b.strip()]

        st.markdown(f'<p class="section-hdr">Anomaly Detection</p>', unsafe_allow_html=True)
        changes["anomaly_high_pmgm_low_payout_pct"] = st.number_input(
            "High PMGM + payout below % of max = anomaly", min_value=0, max_value=100,
            value=int(t.get("anomaly_high_pmgm_low_payout_pct", 50)))
        changes["anomaly_low_pmgm_high_payout_pct"] = st.number_input(
            "Low PMGM + payout at or above % of max = reverse anomaly", min_value=0, max_value=100,
            value=int(t.get("anomaly_low_pmgm_high_payout_pct", 95)))

        st.markdown(f'<p class="section-hdr">Other</p>', unsafe_allow_html=True)
        changes["new_joiner_months"] = st.number_input(
            "New joiner window (months)", min_value=1, max_value=24,
            value=int(t.get("new_joiner_months", 6)))
        changes["proration_threshold"] = st.number_input(
            "Proration threshold (ProrFactor below this = prorated)",
            min_value=0.0, max_value=1.0, step=0.05,
            value=float(t.get("proration_threshold", 1.0)))

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="save-btn">', unsafe_allow_html=True)
    if st.button("Save Thresholds", key="save_thresh"):
        cfg["thresholds"] = changes
        if save_yaml(cfg):
            st.success("✅  Thresholds saved. Active on next query.")
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SYNONYMS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(f'<p class="section-hdr">Synonym Dictionary</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <p style="font-family:'Inter',sans-serif;font-size:0.80rem;color:{SUBTEXT};margin-bottom:16px;">
    Maps user language to canonical terms before routing and vector retrieval.
    Each canonical term can have many phrase variants. One phrase per line.</p>
    """, unsafe_allow_html=True)

    syns = cfg.get("synonyms", {})
    updated_syns = {}

    cols = st.columns(3)
    for i, (canonical, phrases) in enumerate(syns.items()):
        with cols[i % 3]:
            phrases_str = "\n".join(phrases or [])
            new_phrases = st.text_area(
                f"{canonical}", value=phrases_str, height=130,
                key=f"syn_{canonical}",
                help=f"Phrases that mean '{canonical}'")
            updated_syns[canonical] = [p.strip() for p in new_phrases.split("\n") if p.strip()]

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Add new canonical term
    with st.expander("Add new synonym group"):
        new_canon = st.text_input("New canonical term", placeholder="e.g. overtime")
        new_phrases_input = st.text_area("Phrases (one per line)",
                                          placeholder="e.g. extra hours\novertime pay\nOT",
                                          height=100)
        if st.button("Add Group", key="add_syn"):
            if new_canon.strip():
                updated_syns[new_canon.strip()] = [
                    p.strip() for p in new_phrases_input.split("\n") if p.strip()
                ]
                cfg["synonyms"] = updated_syns
                if save_yaml(cfg):
                    st.success(f"✅  Added '{new_canon}'")
                    st.rerun()

    st.markdown('<div class="save-btn">', unsafe_allow_html=True)
    if st.button("Save Synonyms", key="save_syns"):
        cfg["synonyms"] = updated_syns
        if save_yaml(cfg):
            st.success("✅  Synonyms saved. Active on next query.")
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FIELD ALIASES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown(f'<p class="section-hdr">Field Aliases</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <p style="font-family:'Inter',sans-serif;font-size:0.80rem;color:{SUBTEXT};margin-bottom:16px;">
    Maps user-facing terms to actual column names. Used in filter extraction and display.</p>
    """, unsafe_allow_html=True)

    aliases = cfg.get("field_aliases", {})
    alias_items = list(aliases.items())

    # Editable table via text inputs
    updated_aliases = {}
    cols_h = st.columns([2, 2, 1])
    cols_h[0].markdown(f'<p style="font-weight:600;font-size:0.78rem;color:{SUBTEXT};">User Term</p>', unsafe_allow_html=True)
    cols_h[1].markdown(f'<p style="font-weight:600;font-size:0.78rem;color:{SUBTEXT};">Column Name</p>', unsafe_allow_html=True)

    for i, (term, col) in enumerate(alias_items):
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            new_term = st.text_input("t", value=term, key=f"alias_t_{i}", label_visibility="collapsed")
        with c2:
            new_col  = st.text_input("c", value=col,  key=f"alias_c_{i}", label_visibility="collapsed")
        if new_term.strip():
            updated_aliases[new_term.strip()] = new_col.strip()

    with st.expander("Add new alias"):
        nc1, nc2 = st.columns(2)
        with nc1:
            new_alias_term = st.text_input("User term", placeholder="e.g. pay grade")
        with nc2:
            new_alias_col  = st.text_input("Column name", placeholder="e.g. PayGrade")
        if st.button("Add Alias", key="add_alias"):
            if new_alias_term.strip() and new_alias_col.strip():
                updated_aliases[new_alias_term.strip()] = new_alias_col.strip()
                cfg["field_aliases"] = updated_aliases
                if save_yaml(cfg):
                    st.success("✅  Alias added")
                    st.rerun()

    st.markdown('<div class="save-btn">', unsafe_allow_html=True)
    if st.button("Save Field Aliases", key="save_aliases"):
        cfg["field_aliases"] = updated_aliases
        if save_yaml(cfg):
            st.success("✅  Field aliases saved.")
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — INTENT HINTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown(f'<p class="section-hdr">Intent Hints</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <p style="font-family:'Inter',sans-serif;font-size:0.80rem;color:{SUBTEXT};margin-bottom:16px;">
    Phrases that <em>force</em> a specific intent, bypassing the AI router.
    Add business-specific phrasing your executives commonly use. One phrase per line.</p>
    """, unsafe_allow_html=True)

    hints = cfg.get("intent_hints", {})
    updated_hints = {}
    cols = st.columns(3)
    for i, (intent, phrases) in enumerate(hints.items()):
        with cols[i % 3]:
            phrases_str = "\n".join(phrases or [])
            new_phrases = st.text_area(
                f"{intent}", value=phrases_str, height=160,
                key=f"hint_{intent}",
                help=f"Phrases that always route to '{intent}'")
            updated_hints[intent] = [p.strip() for p in new_phrases.split("\n") if p.strip()]

    with st.expander("Add new intent hint group"):
        from ai_engine import INTENT_PATTERNS
        intent_options = list(INTENT_PATTERNS.keys())
        new_hint_intent  = st.selectbox("Intent", intent_options, key="new_hint_intent")
        new_hint_phrases = st.text_area("Phrases (one per line)", height=100, key="new_hint_phrases")
        if st.button("Add Hints", key="add_hints"):
            if new_hint_intent:
                existing = updated_hints.get(new_hint_intent, [])
                new_p = [p.strip() for p in new_hint_phrases.split("\n") if p.strip()]
                updated_hints[new_hint_intent] = list(set(existing + new_p))
                cfg["intent_hints"] = updated_hints
                if save_yaml(cfg):
                    st.success(f"✅  Hints added to '{new_hint_intent}'")
                    st.rerun()

    st.markdown('<div class="save-btn">', unsafe_allow_html=True)
    if st.button("Save Intent Hints", key="save_hints"):
        cfg["intent_hints"] = updated_hints
        if save_yaml(cfg):
            st.success("✅  Intent hints saved. Active on next query.")
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RAW YAML
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown(f'<p class="section-hdr">Raw YAML</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <p style="font-family:'Inter',sans-serif;font-size:0.80rem;color:{SUBTEXT};margin-bottom:16px;">
    Direct edit of the full <code>semantic_layer.yaml</code>.
    Use this for bulk changes or to paste a config from another environment.</p>
    """, unsafe_allow_html=True)

    raw_text = YAML_PATH.read_text() if YAML_PATH.exists() else ""
    edited = st.text_area("semantic_layer.yaml", value=raw_text, height=600)

    col_a, col_b = st.columns([1, 4])
    with col_a:
        st.markdown('<div class="save-btn">', unsafe_allow_html=True)
        if st.button("Save YAML", key="save_raw"):
            try:
                parsed = yaml.safe_load(edited)
                if parsed is None:
                    st.error("YAML is empty.")
                else:
                    YAML_PATH.write_text(edited)
                    st.cache_data.clear()
                    st.success("✅  YAML saved.")
            except yaml.YAMLError as e:
                st.error(f"YAML syntax error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        if st.button("Reset to file on disk", key="reset_raw"):
            st.rerun()
