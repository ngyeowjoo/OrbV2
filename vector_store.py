"""
vector_store.py  —  Phase 2: Semantic retrieval for Orb v2

Uses TF-IDF + cosine similarity (scikit-learn) — no model download,
no internet required, works on Streamlit Cloud immediately.

For production at scale, swap _build_tfidf_index() for a neural embedding
model (all-MiniLM-L6-v2 via sentence-transformers + FAISS) by setting
USE_NEURAL = True and ensuring HuggingFace access.

WHY TF-IDF WORKS WELL HERE
---------------------------
Workforce data is keyword-rich and structured:
  "Bea Reyes Scheme B zero payout Ethics Qualifier failed"
  "Aisyah Torres Scheme A max payout hit 2025-Q2 SG active"

TF-IDF captures these patterns well. The main limitation vs neural
embeddings is synonym handling ("underperform" vs "miss targets") —
mitigated here by enriching each document with synonyms at build time.

ARCHITECTURE
------------
Build phase (once at login, ~0.5s):
  Each employee × cycle row → plain-English sentence → TF-IDF matrix

Query phase (~10ms per query):
  Embed question → cosine similarity against matrix → top-K rows → dataframe
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

VECTOR_STORE_ENABLED = True

# Cache dir hint (used if neural embeddings activated later)
os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME",
    os.path.join(os.path.dirname(__file__), ".model_cache")
)

# ── Module-level index state ───────────────────────────────────────────────────
_vectorizer   = None   # fitted TfidfVectorizer
_fr_matrix    = None   # TF-IDF matrix for Flash Reward rows
_fh_matrix    = None   # TF-IDF matrix for Flash Home rows
_fr_records   = []     # original dicts matching FR matrix rows
_fh_records   = []     # original dicts matching FH matrix rows

# ── SYNONYM EXPANSION ─────────────────────────────────────────────────────────
# Delegated to semantic.py — editable via semantic_layer.yaml without code changes

def _expand(text: str) -> str:
    """Expand text using the semantic layer synonym dictionary."""
    try:
        from semantic import expand_for_vector
        return expand_for_vector(text)
    except Exception:
        return text   # graceful fallback if semantic layer unavailable


# ── TEXT SERIALISERS ──────────────────────────────────────────────────────────
def _fr_to_text(row: dict) -> str:
    name   = row.get("EmployeeName", row.get("EmployeeID", ""))
    eid    = row.get("EmployeeID", "")
    scheme = row.get("Scheme", "")
    cycle  = row.get("Cycle", "")
    payout = float(row.get("TotalCyclePayout", 0) or 0)
    max_p  = float(row.get("SchemeMaxPayout", 1) or 1)
    pct    = round(payout / max_p * 100, 1)
    qual   = row.get("QualifierFailed", "") or ""
    pror   = float(row.get("ProrFactor", 1.0) or 1.0)
    country= row.get("Country", "")
    project= row.get("Project", "")
    rating = row.get("PMGMRating", "") or ""

    parts = [name, eid, scheme, cycle, country, project]
    parts.append(f"payout {payout:.0f} max {max_p:.0f} percent {pct}")
    if payout == 0:
        parts.append("zero payout none empty")
    if pct >= 99:
        parts.append("hit maximum full payout")
    if pct < 50:
        parts.append("low payout underperform miss shortfall")
    if qual:
        parts.append(f"qualifier failed blocked {qual}")
    if pror < 1.0:
        parts.append(f"prorated attendance absent {pror:.0%}")
    if rating:
        parts.append(f"pmgm rating {rating}")
    return _expand(" ".join(str(p) for p in parts if p))


def _fh_to_text(row: dict) -> str:
    name    = row.get("EmployeeName", row.get("EmployeeID", ""))
    eid     = row.get("EmployeeID", "")
    status  = row.get("EmployeeStatus", "") or ""
    country = row.get("Country", "")
    project = row.get("Project", "") or ""
    joined  = str(row.get("JoinDate", ""))[:10]
    last    = str(row.get("LastDate", "") or "")[:10]
    rating  = row.get("PMGMRating", "") or ""

    parts = [name, eid, country, project, status, f"joined {joined}"]
    if last and last != "N":
        parts.append(f"left {last} non-active leaver exited")
    if rating:
        parts.append(f"pmgm rating performance {rating}")
    if status.lower() == "active":
        parts.append("current working employed")
    elif "non" in status.lower():
        parts.append("left resigned terminated exited")
    return _expand(" ".join(str(p) for p in parts if p))


# ── INDEX BUILDER ─────────────────────────────────────────────────────────────
def build_index(fh: pd.DataFrame, fr: pd.DataFrame,
                latest_cycle_only: bool = True):
    """
    Build TF-IDF indices from Flash Home and Flash Reward dataframes.
    Call once at app startup. Runs in ~0.3s for 120 employees.
    """
    global _vectorizer, _fr_matrix, _fh_matrix, _fr_records, _fh_records

    if not VECTOR_STORE_ENABLED:
        return

    # ── Flash Reward ──────────────────────────────────────────────────────────
    fr_idx = fr[fr["Cycle"] == fr["Cycle"].max()].drop_duplicates("EmployeeID") \
             if latest_cycle_only else fr.drop_duplicates(["EmployeeID","Cycle"])

    # Enrich FR with names/country from FH
    if "EmployeeName" not in fr_idx.columns and "EmployeeName" in fh.columns:
        merge_cols = [c for c in ["EmployeeID","EmployeeName","Country","Project","PMGMRating"]
                      if c in fh.columns]
        fr_idx = fr_idx.merge(fh[merge_cols], on="EmployeeID", how="left")

    _fr_records = fr_idx.to_dict("records")
    fr_texts    = [_fr_to_text(r) for r in _fr_records]

    # ── Flash Home ────────────────────────────────────────────────────────────
    _fh_records = fh.to_dict("records")
    fh_texts    = [_fh_to_text(r) for r in _fh_records]

    # ── Fit a single vectorizer on all docs so vocabulary is shared ────────────
    all_texts  = fr_texts + fh_texts
    _vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),    # unigrams + bigrams
        min_df=1,
        max_features=8000,
        sublinear_tf=True,     # log normalization
    )
    all_matrix = _vectorizer.fit_transform(all_texts)

    _fr_matrix = all_matrix[:len(fr_texts)]
    _fh_matrix = all_matrix[len(fr_texts):]

    print(f"vector_store: indexed {len(_fr_records)} FR + {len(_fh_records)} FH docs "
          f"({all_matrix.shape[1]} features)")


# ── RETRIEVER ─────────────────────────────────────────────────────────────────
def vector_retrieve(
    question: str,
    top_k: int = 30,
    source: str = "both",     # "fr" | "fh" | "both"
    country_filter: list = None,
) -> tuple:
    """
    Retrieve the top-K most semantically relevant rows for a question.
    Returns (dataframe, context_string) — same signature as retrieve_data().
    Falls back gracefully if index is not built.
    """
    if _vectorizer is None or not VECTOR_STORE_ENABLED:
        return pd.DataFrame(), "Vector store not ready — using live query."

    try:
        from semantic import expand_for_vector, normalise
        q_expanded = expand_for_vector(normalise(question))
    except Exception:
        q_expanded = question
    q_vec = _vectorizer.transform([q_expanded])

    results_fr = pd.DataFrame()
    results_fh = pd.DataFrame()

    if source in ("fr", "both") and _fr_matrix is not None:
        sims  = cosine_similarity(q_vec, _fr_matrix).flatten()
        top_i = np.argsort(sims)[::-1][:top_k]
        rows  = [_fr_records[i] for i in top_i if sims[i] > 0.0]
        if rows:
            results_fr = pd.DataFrame(rows)
            if country_filter and "ALL" not in country_filter and "Country" in results_fr.columns:
                results_fr = results_fr[results_fr["Country"].isin(country_filter)]

    if source in ("fh", "both") and _fh_matrix is not None:
        sims  = cosine_similarity(q_vec, _fh_matrix).flatten()
        top_i = np.argsort(sims)[::-1][:top_k]
        rows  = [_fh_records[i] for i in top_i if sims[i] > 0.0]
        if rows:
            results_fh = pd.DataFrame(rows)
            if country_filter and "ALL" not in country_filter and "Country" in results_fh.columns:
                results_fh = results_fh[results_fh["Country"].isin(country_filter)]

    # Merge if both sources
    if not results_fr.empty and not results_fh.empty:
        fh_extra = [c for c in results_fh.columns
                    if c not in results_fr.columns or c == "EmployeeID"]
        combined = results_fr.merge(results_fh[fh_extra], on="EmployeeID", how="left")
    elif not results_fr.empty:
        combined = results_fr
    elif not results_fh.empty:
        combined = results_fh
    else:
        return pd.DataFrame(), "No matching records found."

    context = (
        f"Semantic retrieval — top {len(combined)} most relevant records:\n"
        f"{combined.to_string(index=False)}"
    )
    return combined, context


# ── STATUS ────────────────────────────────────────────────────────────────────
def status() -> dict:
    return {
        "enabled":    VECTOR_STORE_ENABLED,
        "backend":    "TF-IDF + cosine similarity (sklearn)",
        "fr_indexed": len(_fr_records),
        "fh_indexed": len(_fh_records),
        "ready":      _vectorizer is not None,
    }
