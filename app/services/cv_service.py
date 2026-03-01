"""
CV matching service – orchestrates the full CV-to-job matching flow.

Flow:
  1. Normalize CV text
  2. Generate 512-dim embedding (Bedrock Titan V2 - reuses existing service)
  3. Insert CV + embedding into S3 (joblab-cv-store bucket)
  4. Perform vector similarity search on Supabase job_chunks (via RPC)
  5. Enrich matches with job metadata from Supabase jobs table
  6. Store top matches back in S3
  7. Return structured response
"""

import io
import logging
import time
from typing import Any

import requests
from PyPDF2 import PdfReader

from app.config import get_settings
from app.schemas.cv_match import CVMatchResponse, JobMatch
from app.services.embeddings import embed_text
from app.services.s3_cv_store import insert_cv, update_matches

logger = logging.getLogger(__name__)


# ── PDF extraction ─────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF file.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file bytes.

    Returns
    -------
    str
        Extracted text from all pages, joined by newlines.
    """
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        text_pages = []
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if page_text.strip():
                text_pages.append(page_text)
        result = "\n".join(text_pages)
        logger.info("PDF extracted  pages=%d  chars=%d", len(pdf_reader.pages), len(result))
        return result
    except Exception as exc:
        logger.exception("PDF extraction failed")
        raise ValueError(f"Failed to parse PDF: {exc}") from exc


# ── Supabase helpers (reuse same auth pattern as joblab_tools) ──────────────


def _headers() -> dict[str, str]:
    """Auth headers for Supabase REST (service_role, bypasses RLS)."""
    settings = get_settings()
    return {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _base_url() -> str:
    return f"{get_settings().supabase_url.rstrip('/')}/rest/v1"


# ── Text normalization ──────────────────────────────────────────────────────

def _normalize_cv_text(raw_text: str) -> str:
    """
    Basic CV text normalization:
    - Strip leading/trailing whitespace
    - Collapse multiple whitespace/newlines
    - Truncate to 10,000 chars for embedding safety
    """
    import re
    text = raw_text.strip()
    text = re.sub(r"\s+", " ", text)
    return text[:10000]


# ── Main matching logic ────────────────────────────────────────────────────

def match_cv(
    cv_text: str,
    countries: list[str] | None = None,
    job_levels: list[str] | None = None,
    job_functions: list[str] | None = None,
    platforms: list[str] | None = None,
    is_remote: bool | None = None,
    role_keyword: str = "",
) -> CVMatchResponse:
    """
    Full CV matching pipeline with optional filtering.

    Parameters
    ----------
    cv_text : str
        Raw CV / resume text.
    countries : list[str], optional
        Filter by specific countries.
    job_levels : list[str], optional
        Filter by seniority levels.
    job_functions : list[str], optional
        Filter by job functions.
    platforms : list[str], optional
        Filter by platforms.
    is_remote : bool, optional
        Filter by remote work.
    role_keyword : str
        Filter by keyword in job title.

    Returns
    -------
    CVMatchResponse
        Contains cv_id and list of top-10 matched jobs with similarity scores.
    """
    start = time.time()
    logger.info("CV matching started  chars=%d", len(cv_text))

    # Step 1 — Normalize
    normalized = _normalize_cv_text(cv_text)
    logger.info("CV normalized  chars=%d", len(normalized))

    # Step 2 — Generate embedding (reuses existing Bedrock Titan V2 service)
    embedding = embed_text(normalized)
    embed_elapsed = round(time.time() - start, 3)
    logger.info("CV embedding generated  dims=%d  time=%.3fs", len(embedding), embed_elapsed)

    # Step 3 — Insert CV into S3
    cv_id = insert_cv(cv_text, embedding)
    logger.info("CV stored in S3  cv_id=%s", cv_id)

    # Step 4 — Vector similarity search on Supabase job_chunks via RPC
    settings = get_settings()
    rpc_url = f"{settings.supabase_url.rstrip('/')}/rest/v1/rpc/match_job_chunks"
    # Fetch more results when filters are active to compensate for filtering
    has_filters = bool(countries or job_levels or job_functions or platforms or is_remote is not None or role_keyword)
    match_count = 100 if has_filters else 30
    payload = {
        "query_embedding": embedding,
        "match_count": match_count,
    }

    rpc_resp = requests.post(rpc_url, headers=_headers(), json=payload, timeout=15)
    rpc_resp.raise_for_status()
    chunk_rows: list[dict[str, Any]] = rpc_resp.json()
    logger.info("Supabase vector search returned %d chunks", len(chunk_rows))

    # Deduplicate by job_id (keep highest similarity per job)
    best_by_job: dict[str, float] = {}
    for row in chunk_rows:
        jid = row.get("job_id")
        sim = float(row.get("similarity", 0))
        if jid and (jid not in best_by_job or sim > best_by_job[jid]):
            best_by_job[jid] = sim

    # Sort by similarity descending - keep more jobs for filtering
    all_job_ids_sorted = sorted(best_by_job.keys(), key=lambda j: best_by_job[j], reverse=True)

    if not all_job_ids_sorted:
        logger.warning("No matching jobs found for CV  cv_id=%s", cv_id)
        update_matches(cv_id, [])
        return CVMatchResponse(cv_id=cv_id, matches=[])

    # Step 5 — Fetch metadata for top candidates (more than we need for filtering)
    # Take top 50 for filtering, or all if fewer
    candidates = all_job_ids_sorted[:min(50, len(all_job_ids_sorted))]
    meta_url = f"{_base_url()}/jobs"
    meta_qs = {
        "select": "job_id,actual_role,company_name,country,location,url,posted_date,job_level_std,job_function_std,job_type_filled,platform,is_remote",
        "job_id": f"in.({','.join(candidates)})",
    }
    meta_resp = requests.get(meta_url, headers=_headers(), params=meta_qs, timeout=15)
    meta_resp.raise_for_status()
    meta_rows: list[dict[str, Any]] = meta_resp.json()

    job_meta: dict[str, dict[str, Any]] = {}
    for row in meta_rows:
        job_meta[row["job_id"]] = row

    # Step 5b — Filter out jobs older than 30 days
    from datetime import datetime, timedelta, timezone
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    recent_job_ids = [
        jid for jid in candidates
        if jid in job_meta and (
            not job_meta[jid].get("posted_date")
            or job_meta[jid]["posted_date"] >= cutoff_date
        )
    ]
    logger.info(
        "Filtered jobs by 30-day recency: %d → %d",
        len(candidates), len(recent_job_ids),
    )

    # Step 5c — Apply user-specified filter criteria with fallback logic
    def passes_filters(jid: str) -> bool:
        meta = job_meta.get(jid, {})
        if countries and meta.get("country") not in countries:
            return False
        if job_levels and meta.get("job_level_std") not in job_levels:
            return False
        if job_functions and meta.get("job_function_std") not in job_functions:
            return False
        if platforms and meta.get("platform") not in platforms:
            return False
        if is_remote is not None:
            if bool(meta.get("is_remote")) != is_remote:
                return False
        if role_keyword:
            if role_keyword.lower() not in meta.get("actual_role", "").lower():
                return False
        return True

    strict_matches = []
    relaxed_matches = []

    if has_filters:
        logger.info(
            "Applying user filters: countries=%s levels=%s functions=%s platforms=%s remote=%s keyword=%s",
            countries, job_levels, job_functions, platforms, is_remote, role_keyword,
        )
        # Separate jobs into strict matches and fallback candidates
        for jid in recent_job_ids:
            if passes_filters(jid):
                strict_matches.append(jid)
            else:
                relaxed_matches.append(jid)
        
        logger.info(
            "User filters applied: %d strict matches, %d relaxed available",
            len(strict_matches), len(relaxed_matches),
        )
        
        # If we have fewer than 5 strict matches, add relaxed ones to reach 10 total
        MIN_STRICT = 5
        TARGET_TOTAL = 10
        
        if len(strict_matches) < MIN_STRICT:
            needed = TARGET_TOTAL - len(strict_matches)
            logger.info(
                "Only %d strict matches (< %d) - adding %d relaxed matches as fallback",
                len(strict_matches), MIN_STRICT, min(needed, len(relaxed_matches)),
            )
        
        final_job_ids = strict_matches[:TARGET_TOTAL]
        final_relaxed_flags = {jid: False for jid in final_job_ids}
        
        # Add relaxed matches if needed
        if len(final_job_ids) < TARGET_TOTAL and relaxed_matches:
            additional = relaxed_matches[:TARGET_TOTAL - len(final_job_ids)]
            for jid in additional:
                final_job_ids.append(jid)
                final_relaxed_flags[jid] = True
    else:
        # No filters - just take top 10
        final_job_ids = recent_job_ids[:10]
        final_relaxed_flags = {jid: False for jid in final_job_ids}
        logger.info("No user filters - taking top %d recent jobs", len(final_job_ids))

    # Step 6 — Build structured matches
    matches: list[JobMatch] = []
    matches_json: list[dict[str, Any]] = []

    for jid in final_job_ids:
        meta = job_meta.get(jid, {})
        similarity = round(best_by_job[jid], 4)
        is_remote_val = meta.get("is_remote")
        if isinstance(is_remote_val, int):
            is_remote_val = bool(is_remote_val)
        is_relaxed = final_relaxed_flags.get(jid, False)
        
        match = JobMatch(
            job_id=jid,
            title=meta.get("actual_role", "Unknown"),
            company=meta.get("company_name", "Unknown"),
            similarity=similarity,
            country=meta.get("country", ""),
            location=meta.get("location", ""),
            url=meta.get("url", ""),
            posted_date=meta.get("posted_date", ""),
            job_level_std=meta.get("job_level_std", "") or "",
            job_function_std=meta.get("job_function_std", "") or "",
            job_type_filled=meta.get("job_type_filled", "") or "",
            platform=meta.get("platform", "") or "",
            is_remote=is_remote_val or False,
            relaxed_criteria=is_relaxed,
        )
        matches.append(match)
        matches_json.append({
            "job_id": jid,
            "title": meta.get("actual_role", "Unknown"),
            "company": meta.get("company_name", "Unknown"),
            "country": meta.get("country", ""),
            "location": meta.get("location", ""),
            "url": meta.get("url", ""),
            "posted_date": meta.get("posted_date", ""),
            "job_level_std": meta.get("job_level_std", "") or "",
            "job_function_std": meta.get("job_function_std", "") or "",
            "job_type_filled": meta.get("job_type_filled", "") or "",
            "platform": meta.get("platform", "") or "",
            "is_remote": is_remote_val or False,
            "similarity": similarity,
        })

    # Step 7 — Store matches in S3
    update_matches(cv_id, matches_json)
    logger.info("CV matching completed  cv_id=%s  matches=%d", cv_id, len(matches))

    total_elapsed = round(time.time() - start, 3)
    logger.info(
        "CV match pipeline  cv_id=%s  total_time=%.3fs  matches=%d",
        cv_id, total_elapsed, len(matches),
    )

    return CVMatchResponse(cv_id=cv_id, matches=matches)
