"""
chunk_classifier.py

Rule-based chunk type classifier. Runs before coreference — zero LLM calls.

Chunk types:
    content      — mathematical exposition, definitions, proofs, algorithms
                   → run full pipeline
    bibliography — reference lists, book titles, citations
                   → skip entirely
    frontmatter  — TOC, preface, acknowledgements, foreword, index
                   → skip entirely
    exercise     — problem sets, exercises, homework
                   → run pipeline but flag as exercise context

Classification is purely deterministic — heading signals + text pattern
signals. No ambiguity, no LLM, runs in microseconds.
"""

import re
from typing import Tuple


# ── Heading signals ──────────────────────────────────────────────────────────

_SKIP_HEADINGS = {
    # Bibliography / references
    "references", "bibliography", "further reading", "suggested reading",
    "additional reading", "suggested texts", "recommended reading",
    "works cited", "citations",
    # Frontmatter
    "table of contents", "contents", "preface", "foreword",
    "acknowledgements", "acknowledgments", "about the authors",
    "about the author", "about this book", "introduction to this volume",
    "editors", "contributors", "list of contributors",
    # Index / appendix admin
    "index", "notation", "list of symbols", "list of figures",
    "list of tables", "list of algorithms",
    # Copyright / publisher
    "copyright", "colophon", "permissions", "isbn",
}

_EXERCISE_HEADINGS = {
    "exercises", "exercise", "problems", "problem set",
    "homework", "practice problems", "review questions",
    "questions", "assignments",
}

_CONTENT_HEADINGS = {
    # Strong positive signals for real mathematical content
    "theorem", "lemma", "proof", "definition", "algorithm",
    "remark", "example", "proposition", "corollary",
    "method", "scheme", "analysis", "convergence", "stability",
    "discretization", "approximation", "error", "numerical",
    "accuracy", "interpolation", "iteration", "eigenvalue",
    "equation", "differential", "integral", "boundary",
    "matrix", "linear", "nonlinear", "optimization",
    "arithmetic", "representation", "rounding", "precision",
    "floating", "cancellation", "conditioning",
    # Partial heading matches — if heading contains ANY of these
    # substrings it's very likely real content
    "rolle", "taylor", "mean value", "intermediate value",
    "gauss", "euler", "runge", "kutta", "jacobi", "newton",
}


# ── Text pattern signals ─────────────────────────────────────────────────────

# Author-like patterns: "Lastname, F." or "Lastname, Firstname"
_AUTHOR_PATTERN = re.compile(
    r'[A-Z][a-z]{2,},\s+[A-Z][\w]*\.?\s+[A-Z]?', re.MULTILINE
)

# Citation-like: year in parentheses (1987) or [12] or ISBN
_CITATION_PATTERN = re.compile(
    r'\((?:19|20)\d{2}\)|\[\d{1,3}\]|ISBN[\s\-]?\d'
)

# TOC-like: "Chapter N ... page" or "N.N heading ... dots ... N"
_TOC_PATTERN = re.compile(
    r'(?:Chapter|Section)\s+\d|\.{4,}|\d+\s*\.\s*\d+\s+[A-Z][a-z]'
)

# LaTeX math signals — strong indicator of real mathematical content
_MATH_PATTERN = re.compile(
    r'\\(?:frac|begin|sum|int|partial|nabla|alpha|beta|lambda|mathbf|mathcal'
    r'|leq|geq|cdot|times|rightarrow|Rightarrow|infty|Delta|Omega|equation'
    r'|align|matrix|bmatrix|pmatrix)\b'
)

# Publisher / edition patterns
_PUBLISHER_PATTERN = re.compile(
    r'(?:Press|Publisher|Edition|Volume|Society|SIAM|Springer|Wiley|Cambridge'
    r'|Academic|Dover|McGraw|Addison|Elsevier)\b'
)


def classify_chunk(chunk: dict) -> Tuple[str, str]:
    """
    Classify a chunk and return (chunk_type, reason).

    Parameters
    ----------
    chunk : dict with keys 'heading', 'content', 'chunk_id'

    Returns
    -------
    (type_str, reason_str)
        type_str : "content" | "bibliography" | "frontmatter" | "exercise"
        reason_str : human-readable explanation for the debugger UI
    """
    heading = chunk.get("heading", "").strip().lower()
    text    = chunk.get("content", "")
    text_l  = text.lower()

    # ── 1. Heading exact match (fastest, most reliable) ──────────────────
    if heading in _SKIP_HEADINGS:
        category = "bibliography" if any(
            w in heading for w in ("reference", "bibliography", "reading", "cited")
        ) else "frontmatter"
        return category, f"heading '{heading}' matched skip list"

    if heading in _EXERCISE_HEADINGS:
        return "exercise", f"heading '{heading}' matched exercise list"

    # ── 2. Heading contains content keyword → fast-accept ────────────────
    for kw in _CONTENT_HEADINGS:
        if kw in heading:
            return "content", f"heading contains content keyword '{kw}'"

    # ── 3. Text pattern scoring ───────────────────────────────────────────
    skip_score    = 0
    content_score = 0

    # Strong LaTeX math → content
    math_hits = len(_MATH_PATTERN.findall(text))
    if math_hits >= 3:
        return "content", f"strong math signal ({math_hits} LaTeX math commands)"
    content_score += math_hits

    # Mathematical text patterns without LaTeX commands — catches plain-text
    # math like "f(a)=f(b)=0", "x ≤ 10^-10", "C[a,b]"
    plain_math = len(re.findall(
        r'[a-z]\([a-z]\)|[a-z]_[0-9]|≤|≥|∈|→|∞|\^[0-9]|\\leq|\\geq|f\(|g\(',
        text
    ))
    content_score += plain_math

    # Domain-specific terms in text body (not heading) → content signal
    domain_terms = len(re.findall(
        r'\b(?:theorem|lemma|proof|algorithm|convergence|stability|error|'
        r'discretization|approximation|floating point|roundoff|rounding|'
        r'precision|arithmetic|matrix|eigenvalue|interpolation|derivative|'
        r'integral|differential|boundary|numerical)\b',
        text_l
    ))
    content_score += domain_terms

    # Author patterns → bibliography/frontmatter
    author_hits = len(_AUTHOR_PATTERN.findall(text))
    skip_score += author_hits * 2

    # Citation patterns → bibliography
    citation_hits = len(_CITATION_PATTERN.findall(text))
    skip_score += citation_hits * 2

    # TOC patterns → frontmatter
    toc_hits = len(_TOC_PATTERN.findall(text))
    skip_score += toc_hits * 2

    # Publisher patterns → frontmatter/bibliography
    pub_hits = len(_PUBLISHER_PATTERN.findall(text))
    skip_score += pub_hits

    # Short chunks without ANY content signal get a small penalty,
    # but NOT enough to override content_score on its own.
    # Previous threshold was 300 chars which was too aggressive.
    if len(text.strip()) < 150 and math_hits == 0 and content_score == 0:
        skip_score += 2

    # ── 4. Score-based decision ───────────────────────────────────────────
    # Content score overrides: if we found domain terms or math in the text,
    # never classify as skip regardless of skip_score.
    if content_score >= 2:
        return "content", f"content signals found (content={content_score}, skip={skip_score})"

    if skip_score >= 5 and content_score >= 0:
        # Determine subcategory
        if citation_hits > 2 or author_hits > 3:
            return "bibliography", (
                f"text signals: {author_hits} author patterns, "
                f"{citation_hits} citations, {pub_hits} publisher mentions"
            )
        return "frontmatter", (
            f"text signals: {toc_hits} TOC patterns, "
            f"{author_hits} author patterns, score={skip_score}"
        )
        
    # Not needed
    if skip_score > content_score * 3 and content_score == 0:
        return "frontmatter", f"skip_score={skip_score} >> content_score={content_score}"

    # ── 5. Default: treat as content ─────────────────────────────────────
    return "content", f"no skip signals (skip={skip_score}, content={content_score})"


def should_skip(chunk: dict) -> Tuple[bool, str, str]:
    """
    Convenience wrapper.

    Returns (skip: bool, chunk_type: str, reason: str)
    """
    chunk_type, reason = classify_chunk(chunk)
    skip = chunk_type in ("bibliography", "frontmatter")
    return skip, chunk_type, reason