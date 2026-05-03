"""
llm_utils.py

Rate-limit-aware LLM invocation with exponential backoff.

Place at kg_agents/utils/llm_utils.py

Usage:
    from kg_agents.utils.llm_utils import invoke_with_backoff
    response = invoke_with_backoff(self.llm, prompt, logger=logger, agent="EntityExtractionAgent")
"""

import re
import time
from typing import Any, Optional


# Maximum total wait time before giving up (seconds)
MAX_TOTAL_WAIT = 120

# Initial wait on first rate limit hit (seconds)
INITIAL_WAIT = 5

# Multiplier for each successive retry
BACKOFF_MULTIPLIER = 2


def invoke_with_backoff(
    llm,
    prompt:   str,
    logger=None,
    agent:    str = "LLM",
    chunk_id: str = "",
) -> Optional[Any]:
    """
    Invoke an LLM with exponential backoff on rate limit errors.

    Returns the LLM response on success, or None if all retries exhausted.
    Raises non-rate-limit exceptions immediately.

    Parameters
    ----------
    llm     : any LangChain chat model
    prompt  : str prompt to send
    logger  : PipelineLogger | None
    agent   : str name for logging
    chunk_id: str for logging context
    """
    wait        = INITIAL_WAIT
    total_wait  = 0

    while True:
        try:
            return llm.invoke(prompt)

        except Exception as e:
            err_str = str(e)

            # ── Rate limit error ─────────────────────────────────────────
            if _is_rate_limit(e):

                # Try to parse the suggested wait time from error message
                suggested = _parse_suggested_wait(err_str)
                actual_wait = min(suggested or wait, MAX_TOTAL_WAIT - total_wait)

                if total_wait >= MAX_TOTAL_WAIT or actual_wait <= 0:
                    msg = (f"Rate limit: max wait {MAX_TOTAL_WAIT}s exhausted "
                           f"on chunk '{chunk_id}'. Returning None.")
                    print(f"  [RateLimit] {msg}")
                    if logger:
                        logger.warning(agent, "rate_limit_exhausted", {
                            "chunk_id":   chunk_id,
                            "total_wait": total_wait,
                            "message":    msg,
                        })
                    return None

                print(f"  [RateLimit] {agent} — waiting {actual_wait}s "
                      f"(total waited: {total_wait}s)...")
                if logger:
                    logger.warning(agent, "rate_limit_backoff", {
                        "chunk_id":   chunk_id,
                        "wait_s":     actual_wait,
                        "total_wait": total_wait,
                    })

                time.sleep(actual_wait)
                total_wait += actual_wait
                wait = min(wait * BACKOFF_MULTIPLIER, 60)

            # ── Any other error — raise immediately ───────────────────────
            else:
                raise


def _is_rate_limit(exc: Exception) -> bool:
    """Detect rate limit errors across openai, groq, and generic HTTP 429."""
    err_str = str(exc).lower()
    return (
        "rate_limit" in err_str
        or "rate limit" in err_str
        or "429" in err_str
        or "tokens per day" in err_str
        or "too many requests" in err_str
        or "insufficient_quota" in err_str
        or "you exceeded your current quota" in err_str
        or type(exc).__name__ in (
            "RateLimitError", "RateLimitException",
            "AuthenticationError",   # OpenAI quota exceeded sometimes shows as this
        )
    )


def _parse_suggested_wait(err_str: str) -> Optional[float]:
    """
    Parse the suggested wait time from a rate limit error message.
    Groq errors say: 'Please try again in 55m25.536s'
    OpenAI errors say: 'Please try again in 20s' or 'retry after 30'
    """
    # Pattern: Nm Ns  (minutes and seconds)
    m = re.search(r'(\d+)m\s*([\d.]+)s', err_str)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))

    # Pattern: Ns (seconds only)
    m = re.search(r'(?:in|after)\s*([\d.]+)\s*s', err_str)
    if m:
        return float(m.group(1))

    # Pattern: retry after N
    m = re.search(r'retry after\s+(\d+)', err_str, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return None