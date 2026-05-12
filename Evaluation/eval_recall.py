"""
Eval 1 — Entity Recall Against Book Indices (Ascher-Greif & Morton-Mayers)
==========================================================================
Uses the fuzzy recall results already embedded in files.txt (the evaluation
output) as ground truth for the index terms, and re-derives all statistics
directly from graph_memory.json so the thesis numbers are reproducible.

Index terms are defined here from the book indices used in the original
recall.py run.  The recall computation is re-run from scratch against the
live graph so results are always up-to-date.
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import load_graph, token_sort_ratio, best_fuzzy_match, node_book_counts

THRESHOLD = 70  # fuzzy match threshold (matches original recall.py)

# ── Index terms (from book indices, used in the original evaluation) ──────────
# These are exactly the terms evaluated in files.txt
ASCHER_INDEX_TERMS = [
    "Adams method", "Adams-Bashforth method", "Adams-Moulton method",
    "adaptive quadrature", "advection equation", "A-stability",
    "backward difference operator", "backward error analysis",
    "backward substitution", "BFGS algorithm", "Big-O notation",
    "BiCGSTAB", "bisection method", "boundary value problem",
    "Burgers equation", "B-spline", "Bezier polynomial",
    "Chebyshev points", "Cholesky decomposition", "condition number",
    "conjugate gradient method", "convergence", "corrected trapezoidal rule",
    "Crank-Nicolson scheme", "critical point", "cubic spline interpolation",
    "differentiation matrix", "differential equation", "discrete cosine transform",
    "discrete Fourier transform", "eigenvector", "energy norm",
    "fast Fourier transform", "finite element method", "finite volume method",
    "fixed point iteration", "forward Euler method", "forward substitution",
    "Fourier transform", "Gaussian elimination", "Gaussian quadrature",
    "Gibbs phenomenon", "Givens rotation", "Hamiltonian system",
    "heat equation", "Hessian matrix", "Householder reflection",
    "IEEE floating point standard", "initial value problem",
    "intermediate value theorem", "inverse iteration", "Jacobi method",
    "Jacobian matrix", "Karush-Kuhn-Tucker conditions", "Krylov subspace method",
    "Lagrange polynomial", "Lagrange polynomial interpolation",
    "LU decomposition", "line search", "linear programming",
    "local truncation error", "mean value theorem", "midpoint rule",
    "multigrid method", "Newton's method", "nonlinear least squares",
    "normal equations", "ordinary differential equation", "partial differential equation",
    "polynomial interpolation", "power method", "predictor-corrector",
    "pseudo-inverse", "QR decomposition", "quasi-Newton method",
    "rate of convergence", "Rayleigh quotient", "Rayleigh quotient iteration",
    "regularization", "Richardson extrapolation", "Rolle theorem",
    "Romberg integration", "roundoff error", "Runge-Kutta method",
    "secant method", "Simpson rule", "singular value decomposition",
    "Sobolev space", "spline interpolation", "stability",
    "steepest descent method", "symmetric positive definite matrix",
    "Taylor series", "Theta notation", "Thomas algorithm",
    "trapezoidal rule", "truncation error", "wavelet",
    "saddle point", "quadrature", "least squares",
    "interpolation", "basis function", "finite difference method",
    "automatic differentiation", "numerical differentiation", "numerical integration",
    "trust region", "backward differentiation formulas", "Hermite polynomial",
    "weighted least squares", "active set method", "Lanczos algorithm",
    "piecewise polynomial", "barycentric interpolation", "floating point arithmetic",
    "modified Gram-Schmidt", "iterative method", "Chebyshev polynomial",
    "descent direction", "divided difference", "fixed point theorem",
    "radial basis function", "double precision", "upwind method",
    "Verlet method", "stiff ODE", "least squares via normal equations",
    "Bayesian method", "variational form", "constrained optimization",
    "convex function", "continuous Gram-Schmidt", "approximate minimum degree",
    "continuation", "principal component analysis", "barrier method",
    "leapfrog method", "machine precision", "forward difference operator",
    "eigenvalue", "least squares via QR", "incomplete Cholesky",
    "absolute stability", "Arnoldi algorithm", "least squares via SVD",
    "gradient descent", "partial pivoting", "augmented Lagrangian method",
    "adaptive algorithm", "ill-posed problem", "Gram-Schmidt algorithm",
    "absolute stability region", "continuous least squares", "incomplete LU",
    "backward Euler", "Delaunay triangulation", "generalized minimum residual",
    "machine epsilon", "successive over-relaxation", "backtracking",
    "bandwidth", "Butcher tableau", "conditioning",
    "Legendre polynomial", "Gauss-Newton method",
]

MORTON_INDEX_TERMS = [
    "ADI methods", "advection equation", "amplification factor",
    "amplification matrix", "asymptotic error analysis", "backward differences",
    "boundary conditions", "box scheme", "Burgers equation",
    "central differences", "CFL condition", "characteristics",
    "chequerboard mode", "Cholesky factors", "classical solution",
    "coarse grid correction", "comparison function", "conjugate gradient method",
    "conservation laws", "consistency", "consistent data",
    "control volume", "convergence", "convection-diffusion problems",
    "corner discontinuity", "Crank-Nicolson scheme", "cylindrical symmetry",
    "damping", "derivative bounds", "difference notation",
    "difference operators", "difference scheme", "diffusion equation",
    "diagonally dominant matrix", "discontinuities", "dispersion relation",
    "dissipative difference schemes", "domain decomposition",
    "domain of dependence", "Douglas-Rachford scheme", "Dufort-Frankel scheme",
    "efficiency of difference schemes", "elliptic equations",
    "energy method", "Engquist-Osher scheme", "error analysis",
    "explicit scheme", "finite element method", "finite volume methods",
    "flux functions", "forward differences", "Fourier analysis",
    "Fourier coefficient", "Fourier modes", "Fourier transform",
    "Galerkin method", "generalised solution", "Gauss-Seidel iteration",
    "geometric integrators", "GMRES algorithm", "group velocity",
    "Hamiltonian PDE", "hat function", "heat equation",
    "ICCG algorithm", "implicit scheme", "incomplete Cholesky",
    "instability", "interface conditions", "irreducible matrix",
    "iteration matrix", "Jacobi iteration", "Kreiss matrix theorem",
    "Krylov subspace methods", "l2 norm", "Lax equivalence theorem",
    "Lax-Wendroff scheme", "leap-frog scheme", "linear algebraic equations",
    "LOD scheme", "maximum norm", "maximum principle",
    "mesh Peclet number", "modified equation analysis", "monotonicity preserving",
    "multi-symplectic PDE", "multigrid method", "MUSCL schemes",
    "natural boundary condition", "norms", "order of accuracy",
    "oscillating modes", "Parseval relation", "parabolic equations",
    "Peaceman-Rachford scheme", "Petrov-Galerkin methods", "phase error",
    "Poisson bracket", "PPM scheme", "practical stability",
    "preconditioning", "prolongation", "rarefaction wave",
    "recurrence relations", "refinement path", "residual",
    "restriction", "Riemann invariants", "Roe scheme",
    "Runge-Kutta time-stepping", "search direction", "semi-discrete methods",
    "separation of variables", "shape functions", "shock",
    "singular perturbation problems", "smoothing", "sparse matrices",
    "spectral radius", "spurious mode", "stability",
    "stencil", "streamline diffusion methods", "strong stability",
    "summation by parts", "test function", "theta-method",
    "Thomas algorithm", "three-time-level scheme", "total variation",
    "trial functions", "triangular elements", "tridiagonal matrix",
    "truncation error", "TV-stability", "TVD",
    "two-grid method", "unconditional stability", "upwind scheme",
    "V-cycles", "variational formulation", "von Neumann condition",
    "W-cycles", "wave equation", "weak solutions",
    "weighted Jacobi", "well-posed problem",
    "Dirichlet boundary conditions", "boundary-fitted mesh",
    "area preserving", "Cholesky factors",
]

# ── Node type categories for breakdown ────────────────────────────────────────
TYPE_CATEGORY = {
    "NumericalMethod":      "Algorithm / Method",
    "MathematicalObject":   "Mathematical Object",
    "ProblemType":          "Problem Type",
    "TheoreticalProperty":  "Theoretical Property",
    "Theorem":              "Theorem / Result",
    "MatrixConcept":        "Matrix Concept",
    "ErrorConcept":         "Error Concept",
    "Definition":           "Definition",
    "MatrixPropertyConcept":"Matrix Concept",
    "MatrixOperationConcept":"Matrix Concept",
    "ConstraintConcept":    "Constraint / Optimisation",
    "NumericalConcept":     "Numerical Concept",
    "OrthogonalConcept":    "Mathematical Object",
    "DerivativeConcept":    "Mathematical Object",
    "NormConcept":          "Mathematical Object",
    "CriticalPointConcept": "Constraint / Optimisation",
    "SolutionProperty":     "Theoretical Property",
    "FiniteDifferenceMethod":"Algorithm / Method",
    "TimeSteppingMethod":   "Algorithm / Method",
    "FourierAnalysisConcept":"Mathematical Object",
    "StabilityConcept":     "Theoretical Property",
    "KrylovSubspaceConcept":"Algorithm / Method",
    "MatrixConditionConcept":"Matrix Concept",
    "FloatingPointStandard":"Definition",
}


def run_recall(index_terms, graph_nodes, label, threshold=THRESHOLD):
    """
    Fuzzy recall of index_terms against graph node names.
    Returns a results dict with matched/missed lists and summary stats.
    """
    node_names = list(graph_nodes.keys())
    matched, missed = [], []

    for term in index_terms:
        best_score, best_cand = 0, None
        for name in node_names:
            s = token_sort_ratio(term, name)
            if s > best_score:
                best_score, best_cand = s, name

        if best_score >= threshold:
            sal = graph_nodes[best_cand]["salience"]
            node_type = graph_nodes[best_cand]["type"]
            matched.append({
                "term": term,
                "matched_to": best_cand,
                "score": round(best_score, 6),
                "salience": sal,
                "type": node_type,
                "category": TYPE_CATEGORY.get(node_type, node_type),
            })
        else:
            matched_str = best_cand if best_cand else "—"
            missed.append({
                "term": term,
                "closest": matched_str,
                "best_score": round(best_score, 6),
            })

    total = len(index_terms)
    n_matched = len(matched)
    recall_pct = 100.0 * n_matched / total if total else 0

    return {
        "label": label,
        "threshold": threshold,
        "total_terms": total,
        "matched": n_matched,
        "missed": len(missed),
        "recall_pct": round(recall_pct, 1),
        "matched_list": sorted(matched, key=lambda x: -x["score"]),
        "missed_list":  sorted(missed, key=lambda x: -x["best_score"]),
    }


def category_breakdown(matched_list, missed_list):
    """Per-category recall breakdown."""
    from collections import defaultdict
    cat_matched = defaultdict(int)
    cat_total   = defaultdict(int)

    for m in matched_list:
        cat = m["category"]
        cat_matched[cat] += 1
        cat_total[cat]   += 1
    for m in missed_list:
        # missed items don't have a type — we can't break them down by type
        # so we only report on matched categories
        pass
    # Build rows sorted by total desc
    rows = []
    for cat, total in sorted(cat_total.items(), key=lambda x: -x[1]):
        n = cat_matched[cat]
        rows.append({"category": cat, "matched": n, "total": total,
                     "pct": round(100.0*n/total, 1) if total else 0})
    return rows


def top20_salience_vs_index(graph_nodes, matched_list):
    """
    Cross-reference the top-20 nodes by salience against the matched index terms.
    Returns rows showing which top-20 nodes were index-matched.
    """
    top20 = sorted(graph_nodes.items(), key=lambda x: -x[1]["salience"])[:20]
    matched_names = {m["matched_to"] for m in matched_list}
    rows = []
    for name, data in top20:
        rows.append({
            "name": name,
            "salience": data["salience"],
            "type": data["type"],
            "in_index": name in matched_names,
        })
    return rows


def missed_failure_analysis(missed_list):
    """
    Group missed terms into failure-mode categories.
    """
    categories = {
        "Close miss (65-69)": [],
        "Moderate miss (60-64)": [],
        "Far miss (<60)": [],
    }
    for m in missed_list:
        s = m["best_score"]
        if s >= 65:
            categories["Close miss (65-69)"].append(m)
        elif s >= 60:
            categories["Moderate miss (60-64)"].append(m)
        else:
            categories["Far miss (<60)"].append(m)
    return categories


def print_recall_report(result):
    print(f"\n{'='*70}")
    print(f"RECALL: {result['label']}")
    print(f"{'='*70}")
    print(f"  Index terms evaluated : {result['total_terms']}")
    print(f"  Matched (>= {result['threshold']})     : {result['matched']}  ({result['recall_pct']}%)")
    print(f"  Missed                : {result['missed']}  ({100-result['recall_pct']:.1f}%)")

    print(f"\n  -- TOP 20 MATCHED TERMS (by score) --")
    for m in result["matched_list"][:20]:
        print(f"    [{m['score']:>10.6f}]  '{m['term']}'  →  '{m['matched_to']}'  (sal={m['salience']:.3f}, {m['category']})")

    print(f"\n  -- MISSED TERMS BY FAILURE MODE --")
    failure = missed_failure_analysis(result["missed_list"])
    for mode, items in failure.items():
        print(f"  {mode}: {len(items)} terms")
        for m in items[:8]:
            print(f"    [{m['best_score']:>6.2f}]  '{m['term']}'  →  closest: '{m['closest']}'")

    print(f"\n  -- CATEGORY BREAKDOWN --")
    breakdown = category_breakdown(result["matched_list"], result["missed_list"])
    print(f"  {'Category':<32} {'Matched':>7}  {'Recall':>7}")
    print(f"  {'-'*50}")
    for row in breakdown:
        print(f"  {row['category']:<32} {row['matched']:>7}  {row['pct']:>6.1f}%")


def main():
    g = load_graph()
    nodes = g["nodes"]

    ascher_result = run_recall(ASCHER_INDEX_TERMS, nodes, "Ascher-Greif")
    morton_result = run_recall(MORTON_INDEX_TERMS, nodes, "Morton-Mayers")

    print_recall_report(ascher_result)
    print_recall_report(morton_result)

    # Top-20 salience vs index presence
    print(f"\n{'='*70}")
    print("TOP-20 SALIENCE NODES vs INDEX PRESENCE")
    print(f"{'='*70}")
    print(f"  {'Node':<45} {'Salience':>8}  {'Type':<28} {'In Index'}")
    print(f"  {'-'*90}")
    a_top20 = top20_salience_vs_index(nodes, ascher_result["matched_list"] + morton_result["matched_list"])
    for row in a_top20:
        tick = "✓" if row["in_index"] else "✗"
        print(f"  {row['name']:<45} {row['salience']:>8.4f}  {row['type']:<28} {tick}")

    # Summary
    print(f"\n{'='*70}")
    print("RECALL SUMMARY")
    print(f"{'='*70}")
    print(f"  Ascher-Greif  : {ascher_result['recall_pct']}%  ({ascher_result['matched']}/{ascher_result['total_terms']} terms matched)")
    print(f"  Morton-Mayers : {morton_result['recall_pct']}%  ({morton_result['matched']}/{morton_result['total_terms']} terms matched)")
    print(f"  Fuzzy threshold: {THRESHOLD}/100")

    return {"ascher": ascher_result, "morton": morton_result}


if __name__ == "__main__":
    main()
