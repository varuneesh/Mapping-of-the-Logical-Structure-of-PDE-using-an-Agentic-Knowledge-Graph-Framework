"""
Shared utilities for all evaluation scripts.
All data paths are resolved relative to this file's location.
"""
import json
import os
from difflib import SequenceMatcher
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
# UPLOADS = "/mnt/user-data/uploads"
ROOT_DIR = Path.cwd().parent
GRAPH_PATH         = ROOT_DIR / "data" / "graph_memory.json"
DOC_MEMORY_PATH    = ROOT_DIR / "data" / "document_memory.json"
ASCHER_SUMMARY     = ROOT_DIR / "logs" / "ascher_greif_run_1_auto_run_20260502_013713_summary.json"
MORTON_SUMMARY     = ROOT_DIR / "logs" / "morton_pde_run_1_auto_run_20260503_214810_summary.json"
ASCHER_JSONL       = ROOT_DIR / "logs" / "ascher_greif_run_1_auto_run_20260502_013713.jsonl"
MORTON_JSONL       = ROOT_DIR / "logs" / "morton_pde_run_1_auto_run_20260503_214810.jsonl"

# ── Loaders ──────────────────────────────────────────────────────────────────
def load_graph():
    with open(GRAPH_PATH) as f:
        return json.load(f)

def load_summary(path):
    with open(path) as f:
        return json.load(f)

def load_jsonl(path):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events

# ── Fuzzy matching (token-sort ratio, mirrors rapidfuzz token_sort_ratio) ────
def _token_sort(s):
    return " ".join(sorted(s.lower().split()))

def token_sort_ratio(a, b):
    """Return 0-100 fuzzy score using sorted-token comparison."""
    return SequenceMatcher(None, _token_sort(a), _token_sort(b)).ratio() * 100

def best_fuzzy_match(term, candidates, threshold=80):
    """
    Return (score, best_candidate) for a term against a list of candidates.
    Returns (0, None) if nothing exceeds threshold.
    """
    best_score, best_cand = 0, None
    for cand in candidates:
        s = token_sort_ratio(term, cand)
        if s > best_score:
            best_score, best_cand = s, cand
    if best_score >= threshold:
        return best_score, best_cand
    return best_score, None   # return best_score even if below threshold for analysis

# ── Graph helpers ─────────────────────────────────────────────────────────────
def node_book_counts(node_data):
    """Return (ascher_count, morton_count) of source chunks for a node."""
    a = sum(1 for s in node_data["sources"] if "Ascher" in s)
    m = sum(1 for s in node_data["sources"] if "Morton" in s)
    return a, m

def is_cross_book(node_data):
    a, m = node_book_counts(node_data)
    return a > 0 and m > 0

def is_ascher_only(node_data):
    a, m = node_book_counts(node_data)
    return a > 0 and m == 0

def is_morton_only(node_data):
    a, m = node_book_counts(node_data)
    return a == 0 and m > 0

# ── Core ontology types (pre-extension baseline) ──────────────────────────────
CORE_TYPES = {
    "NumericalMethod", "MathematicalObject", "ProblemType",
    "TheoreticalProperty", "Theorem", "ErrorConcept", "Definition",
}

EXTENDED_TYPES = {
    # Ascher extensions
    "FloatingPointStandard", "NumericalConcept", "MatrixConcept",
    "OrthogonalConcept", "MatrixConditionConcept", "KrylovSubspaceConcept",
    "MatrixOperationConcept", "NormConcept", "MatrixPropertyConcept",
    "CriticalPointConcept", "ConstraintConcept", "SolutionProperty",
    "DerivativeConcept",
    # Morton extensions
    "TimeSteppingMethod", "FiniteDifferenceMethod", "FourierAnalysisConcept",
    "StabilityConcept",
}
