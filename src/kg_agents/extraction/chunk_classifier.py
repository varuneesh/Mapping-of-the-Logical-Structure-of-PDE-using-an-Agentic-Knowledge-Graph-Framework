import re
from typing import Tuple


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
    "theorem", "lemma", "proof", "definition", "algorithm",
    "remark", "example", "proposition", "corollary",
    "method", "scheme", "analysis", "convergence", "stability",
    "discretization", "approximation", "error", "numerical",
    "accuracy", "interpolation", "iteration", "eigenvalue",
    "equation", "differential", "integral", "boundary",
    "matrix", "linear", "nonlinear", "optimization",
    "arithmetic", "representation", "rounding", "precision",
    "floating", "cancellation", "conditioning",
    "rolle", "taylor", "mean value", "intermediate value",
    "gauss", "euler", "runge", "kutta", "jacobi", "newton",
}

_AUTHOR_PATTERN = re.compile(
    r'[A-Z][a-z]{2,},\s+[A-Z][\w]*\.?\s+[A-Z]?', re.MULTILINE
)

_CITATION_PATTERN = re.compile(
    r'\((?:19|20)\d{2}\)|\[\d{1,3}\]|ISBN[\s\-]?\d'
)

_TOC_PATTERN = re.compile(
    r'(?:Chapter|Section)\s+\d|\.{4,}|\d+\s*\.\s*\d+\s+[A-Z][a-z]'
)

_MATH_PATTERN = re.compile(
    r'\\(?:frac|begin|sum|int|partial|nabla|alpha|beta|lambda|mathbf|mathcal'
    r'|leq|geq|cdot|times|rightarrow|Rightarrow|infty|Delta|Omega|equation'
    r'|align|matrix|bmatrix|pmatrix)\b'
)

_PUBLISHER_PATTERN = re.compile(
    r'(?:Press|Publisher|Edition|Volume|Society|SIAM|Springer|Wiley|Cambridge'
    r'|Academic|Dover|McGraw|Addison|Elsevier)\b'
)


def classify_chunk(chunk: dict) -> Tuple[str, str]:
    
    heading = chunk.get("heading", "").strip().lower()
    text    = chunk.get("content", "")
    text_l  = text.lower()

    if heading in _SKIP_HEADINGS:
        category = "bibliography" if any(
            w in heading for w in ("reference", "bibliography", "reading", "cited")
        ) else "frontmatter"
        return category, f"heading '{heading}' matched skip list"

    if heading in _EXERCISE_HEADINGS:
        return "exercise", f"heading '{heading}' matched exercise list"

    for kw in _CONTENT_HEADINGS:
        if kw in heading:
            return "content", f"heading contains content keyword '{kw}'"

    skip_score    = 0
    content_score = 0

    math_hits = len(_MATH_PATTERN.findall(text))
    if math_hits >= 3:
        return "content", f"strong math signal ({math_hits} LaTeX math commands)"
    content_score += math_hits

    plain_math = len(re.findall(
        r'[a-z]\([a-z]\)|[a-z]_[0-9]|≤|≥|∈|→|∞|\^[0-9]|\\leq|\\geq|f\(|g\(',
        text
    ))
    content_score += plain_math

    domain_terms = len(re.findall(
        r'\b(?:theorem|lemma|proof|algorithm|convergence|stability|error|'
        r'discretization|approximation|floating point|roundoff|rounding|'
        r'precision|arithmetic|matrix|eigenvalue|interpolation|derivative|'
        r'integral|differential|boundary|numerical)\b',
        text_l
    ))
    content_score += domain_terms

    author_hits = len(_AUTHOR_PATTERN.findall(text))
    skip_score += author_hits * 2

    citation_hits = len(_CITATION_PATTERN.findall(text))
    skip_score += citation_hits * 2

    toc_hits = len(_TOC_PATTERN.findall(text))
    skip_score += toc_hits * 2

    pub_hits = len(_PUBLISHER_PATTERN.findall(text))
    skip_score += pub_hits

    if len(text.strip()) < 150 and math_hits == 0 and content_score == 0:
        skip_score += 2

    if content_score >= 2:
        return "content", f"content signals found (content={content_score}, skip={skip_score})"

    if skip_score >= 5 and content_score >= 0:
        if citation_hits > 2 or author_hits > 3:
            return "bibliography", (
                f"text signals: {author_hits} author patterns, "
                f"{citation_hits} citations, {pub_hits} publisher mentions"
            )
        return "frontmatter", (
            f"text signals: {toc_hits} TOC patterns, "
            f"{author_hits} author patterns, score={skip_score}"
        )
        
    if skip_score > content_score * 3 and content_score == 0:
        return "frontmatter", f"skip_score={skip_score} >> content_score={content_score}"

    return "content", f"no skip signals (skip={skip_score}, content={content_score})"


def should_skip(chunk: dict) -> Tuple[bool, str, str]:

    chunk_type, reason = classify_chunk(chunk)
    skip = chunk_type in ("bibliography", "frontmatter")
    return skip, chunk_type, reason