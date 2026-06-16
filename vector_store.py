"""
vector_store.py  —  Phase 2: Semantic vector retrieval for Orb v2

STATUS: SCAFFOLD — not yet active. Activate by:
  1. pip install faiss-cpu sentence-transformers
  2. Set VECTOR_STORE_ENABLED = True
  3. Call build_index() on app startup (after load_data())
  4. Replace retrieve_data() calls with vector_retrieve() in ai_engine.py

WHY THIS EXISTS
---------------
The current architecture queries the full dataset on every turn and dumps
up to 111 rows into the LLM context window. This works for 120 employees
but breaks at scale (10,000+ employees × 6 cycles = 60,000+ rows).

Vector retrieval solves two problems:
  1. Context window management — sends only the 20-30 most relevant rows
  2. Semantic matching — finds "employees who struggled" without needing
     exact regex keywords like "underperform" or "below target"

HOW IT WORKS
------------
Build phase (once at startup, ~5s):
  Each row in Flash Reward and Flash Home is converted to a plain-English
  sentence and embedded as a 384-dimensional vector using a local model.

  Flash Reward row → "Aisyah Torres (E0001), Scheme A, 2025-Q2: achieved
    82.3 on Revenue Target (target 100), payout 1200.50, no qualifier issues"

  Flash Home row → "Aisyah Torres (E0001): Active employee in SG,
    Project Alpha, joined 2021-03-15, PMGM rating: Meets Expectations"

  All vectors stored in a FAISS flat index (exact search, no approximation
  needed at this scale).

Query phase (per turn, ~50ms):
  1. Embed the user's question
  2. Retrieve top-K most similar rows from the index
  3. Reconstruct the matching rows as a dataframe
  4. Pass to Claude for reasoning

CHUNKING STRATEGY
-----------------
Each logical "chunk" is one employee × one cycle (Flash Reward) or one
employee record (Flash Home). We do NOT chunk by metric row — that would
split related information. Instead, all metrics for one employee × cycle
are concatenated into a single sentence before embedding.

This means:
  - Flash Reward index: ~111 entries (one per employee in latest cycle)
  - Flash Home index: ~120 entries (one per employee)
  - Total index size: ~231 vectors × 384 dims ≈ tiny (< 1MB)

At 10,000 employees × 6 cycles: ~60,000 vectors ≈ 90MB — still fine for
in-memory FAISS on a standard cloud instance.

HYBRID APPROACH (recommended for production)
--------------------------------------------
Vector search narrows candidates → live join fills in detail.
  1. Vector search → top-20 EmployeeIDs
  2. Live DB query → SELECT * WHERE EmployeeID IN (top-20 ids)
  3. Claude reasoning on the tight result set

This gives semantic flexibility without stale data problems.
"""

VECTOR_STORE_ENABLED = False   # ← set to True to activate

try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

import pandas as pd
from typing import Optional

_MODEL        = None   # lazy-loaded sentence transformer
_FR_INDEX     = None   # FAISS index for Flash Reward
_FH_INDEX     = None   # FAISS index for Flash Home
_FR_RECORDS   = []     # original rows matching FR index positions
_FH_RECORDS   = []     # original rows matching FH index positions


# ── TEXT SERIALISERS ──────────────────────────────────────────────────────────

def _fr_row_to_text(row: pd.Series) -> str:
    """Convert one Flash Reward employee × cycle row to a searchable sentence."""
    name    = row.get("EmployeeName", row.get("EmployeeID", "Unknown"))
    eid     = row.get("EmployeeID", "")
    scheme  = row.get("Scheme", "")
    cycle   = row.get("Cycle", "")
    payout  = row.get("TotalCyclePayout", 0)
    max_p   = row.get("SchemeMaxPayout", 0)
    pct     = round(payout / max_p * 100, 1) if max_p else 0
    qual    = row.get("QualifierFailed", "")
    pror    = row.get("ProrFactor", 1.0)

    parts = [
        f"{name} ({eid}), {scheme}, {cycle}:",
        f"payout {payout:,.0f} of max {max_p:,.0f} ({pct}%)",
    ]
    if qual:
        parts.append(f"qualifier failed: {qual}")
    if pror < 1.0:
        parts.append(f"attendance prorated at {pror:.0%}")
    if payout == 0:
        parts.append("zero payout this cycle")
    if pct >= 99:
        parts.append("hit maximum payout")
    return " | ".join(parts)


def _fh_row_to_text(row: pd.Series) -> str:
    """Convert one Flash Home employee record to a searchable sentence."""
    name    = row.get("EmployeeName", row.get("EmployeeID", "Unknown"))
    eid     = row.get("EmployeeID", "")
    status  = row.get("EmployeeStatus", "")
    country = row.get("Country", "")
    project = row.get("Project", "")
    joined  = str(row.get("JoinDate", ""))[:10]
    last    = str(row.get("LastDate", "")) if pd.notna(row.get("LastDate")) else ""
    rating  = row.get("PMGMRating", "")

    parts = [
        f"{name} ({eid}): {status} employee",
        f"country {country}, {project}",
        f"joined {joined}",
    ]
    if last:
        parts.append(f"left {last[:10]}")
    if rating:
        parts.append(f"PMGM: {rating}")
    return " | ".join(parts)


# ── INDEX BUILDER ─────────────────────────────────────────────────────────────

def build_index(fh: pd.DataFrame, fr: pd.DataFrame, latest_cycle_only: bool = True):
    """
    Build FAISS indices from Flash Home and Flash Reward dataframes.
    Call once at app startup after load_data().

    Parameters
    ----------
    fh               : Flash Home dataframe
    fr               : Flash Reward dataframe
    latest_cycle_only: Only index the latest cycle to keep context tight
    """
    global _MODEL, _FR_INDEX, _FH_INDEX, _FR_RECORDS, _FH_RECORDS

    if not VECTOR_STORE_ENABLED:
        return
    if not _DEPS_AVAILABLE:
        print("vector_store: faiss-cpu or sentence-transformers not installed. Skipping.")
        return

    print("vector_store: loading embedding model...")
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")   # 80MB, fast, 384-dim

    # ── Flash Reward ──
    if latest_cycle_only:
        fr_to_index = fr[fr["Cycle"] == fr["Cycle"].max()].drop_duplicates("EmployeeID")
    else:
        fr_to_index = fr.drop_duplicates(["EmployeeID", "Cycle"])

    # Merge names for richer text
    if "EmployeeName" not in fr_to_index.columns and "EmployeeName" in fh.columns:
        fr_to_index = fr_to_index.merge(
            fh[["EmployeeID","EmployeeName"]], on="EmployeeID", how="left"
        )

    _FR_RECORDS = fr_to_index.to_dict("records")
    fr_texts    = [_fr_row_to_text(pd.Series(r)) for r in _FR_RECORDS]
    fr_vecs     = _MODEL.encode(fr_texts, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(fr_vecs)
    _FR_INDEX   = faiss.IndexFlatIP(fr_vecs.shape[1])   # inner product on normalised = cosine
    _FR_INDEX.add(fr_vecs)

    # ── Flash Home ──
    _FH_RECORDS = fh.to_dict("records")
    fh_texts    = [_fh_row_to_text(pd.Series(r)) for r in _FH_RECORDS]
    fh_vecs     = _MODEL.encode(fh_texts, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(fh_vecs)
    _FH_INDEX   = faiss.IndexFlatIP(fh_vecs.shape[1])
    _FH_INDEX.add(fh_vecs)

    print(f"vector_store: indexed {len(_FR_RECORDS)} FR rows + {len(_FH_RECORDS)} FH rows")


# ── RETRIEVER ─────────────────────────────────────────────────────────────────

def vector_retrieve(
    question: str,
    top_k: int = 30,
    source: str = "both",   # "fr" | "fh" | "both"
    country_filter: Optional[list] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Retrieve the top-K most semantically relevant rows for a question.

    Returns (dataframe, context_string) matching the signature of retrieve_data().
    Falls back to an empty dataframe if the index is not built.

    Parameters
    ----------
    question       : User's natural language question
    top_k          : Number of rows to retrieve per source
    source         : Which index(es) to search
    country_filter : If provided, filters results to these countries
    """
    if not VECTOR_STORE_ENABLED or _MODEL is None:
        return pd.DataFrame(), "Vector store not enabled."

    q_vec = _MODEL.encode([question], show_progress_bar=False).astype("float32")
    faiss.normalize_L2(q_vec)

    results_fr = pd.DataFrame()
    results_fh = pd.DataFrame()

    if source in ("fr", "both") and _FR_INDEX is not None and _FR_INDEX.ntotal > 0:
        scores, idxs = _FR_INDEX.search(q_vec, min(top_k, _FR_INDEX.ntotal))
        rows = [_FR_RECORDS[i] for i in idxs[0] if i >= 0]
        results_fr = pd.DataFrame(rows)
        if country_filter and "ALL" not in country_filter and "Country" in results_fr.columns:
            results_fr = results_fr[results_fr["Country"].isin(country_filter)]

    if source in ("fh", "both") and _FH_INDEX is not None and _FH_INDEX.ntotal > 0:
        scores, idxs = _FH_INDEX.search(q_vec, min(top_k, _FH_INDEX.ntotal))
        rows = [_FH_RECORDS[i] for i in idxs[0] if i >= 0]
        results_fh = pd.DataFrame(rows)
        if country_filter and "ALL" not in country_filter and "Country" in results_fh.columns:
            results_fh = results_fh[results_fh["Country"].isin(country_filter)]

    # Join on EmployeeID if both sources retrieved
    if not results_fr.empty and not results_fh.empty:
        fh_cols = [c for c in results_fh.columns if c not in results_fr.columns or c == "EmployeeID"]
        combined = results_fr.merge(results_fh[fh_cols], on="EmployeeID", how="left")
    elif not results_fr.empty:
        combined = results_fr
    else:
        combined = results_fh

    if combined.empty:
        return combined, "No matching records found."

    context = (
        f"Semantic retrieval: top {len(combined)} rows most relevant to your question.\n"
        f"{combined.to_string(index=False)}"
    )
    return combined, context


# ── STATUS ────────────────────────────────────────────────────────────────────

def status() -> dict:
    return {
        "enabled":      VECTOR_STORE_ENABLED,
        "deps_ok":      _DEPS_AVAILABLE,
        "fr_indexed":   _FR_INDEX.ntotal if _FR_INDEX else 0,
        "fh_indexed":   _FH_INDEX.ntotal if _FH_INDEX else 0,
        "model_loaded": _MODEL is not None,
    }
