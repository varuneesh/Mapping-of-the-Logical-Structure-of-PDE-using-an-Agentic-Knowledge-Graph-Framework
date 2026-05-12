"""
Eval 4 — Ontology Evolution Analysis
======================================
Reconstructs the timeline of ontology extensions from both JSONL logs,
analyses which classes were added and when, how many nodes were reclassified
after each update, how many relations were rescued, and validates the semantic
coherence of the extended types.
"""

import sys, os, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import (load_graph, load_jsonl,
                        ASCHER_JSONL, MORTON_JSONL, EXTENDED_TYPES)


# ── Ontology class descriptions (from thesis — for semantic coherence check) ──
CLASS_DESCRIPTIONS = {
    # Ascher extensions
    "FloatingPointStandard":   "Standards or specifications for floating-point arithmetic (e.g. IEEE 754)",
    "NumericalConcept":        "Fundamental numerical computing concepts (precision, machine epsilon, rounding)",
    "ErrorConcept":            "Quantified or named error types in numerical computation",
    "MatrixConcept":           "Named matrix classes with special structural properties",
    "OrthogonalConcept":       "Orthogonality concepts — orthogonal projections, orthonormal bases",
    "MatrixConditionConcept":  "Condition number and conditioning concepts for matrices",
    "KrylovSubspaceConcept":   "Krylov subspace and related iterative solver concepts",
    "MatrixOperationConcept":  "Named matrix factorisation or decomposition operations",
    "NormConcept":             "Named vector and matrix norms",
    "MatrixPropertyConcept":   "Named scalar or structural properties of matrices",
    "CriticalPointConcept":    "Critical points, saddle points, local/global extrema in optimisation",
    "ConstraintConcept":       "Constraint types in constrained optimisation (active set, KKT, etc.)",
    "SolutionProperty":        "Properties of solutions to numerical problems (uniqueness, feasibility)",
    "DerivativeConcept":       "Named derivative or differentiation concepts (Jacobian, Hessian, gradient)",
    # Morton extensions
    "TimeSteppingMethod":      "Methods for advancing solutions in time (explicit/implicit time integration)",
    "FiniteDifferenceMethod":  "Specific finite difference discretisation schemes",
    "FourierAnalysisConcept":  "Fourier analysis tools used in stability and error analysis of PDE solvers",
    "StabilityConcept":        "Specific stability concepts for PDE schemes (von Neumann, CFL, TVD)",
}


def extract_ontology_timeline(jsonl_events, book_label):
    """Extract extension events and reclassification events from JSONL."""
    extensions = []
    reclassifications = []
    relations_rescued = []
    inter_chunk_passes = []

    for event in jsonl_events:
        agent = event.get("agent", "")
        ev    = event.get("event", "")
        chunk = event.get("chunk_id", "")
        data  = event.get("data", {})
        ts    = event.get("ts", "")

        if ev == "extensions_applied":
            # Extract chunk number for x-axis positioning
            chunk_num = _chunk_number(chunk)
            extensions.append({
                "chunk_id":    chunk,
                "chunk_num":   chunk_num,
                "timestamp":   ts,
                "version":     data.get("version", "?"),
                "new_classes": data.get("new_classes", []),
                "new_relations": data.get("new_relations", []),
            })

        elif agent == "ReclassificationPass" and ev == "relations_rescued":
            reclassifications.append({
                "chunk_id":    chunk,
                "chunk_num":   _chunk_number(chunk),
                "rescued":     data.get("count", 0),
            })

        elif agent == "InterChunkRelationExtractor" and ev == "completed":
            inter_chunk_passes.append({
                "chunk_id":  chunk,
                "chunk_num": _chunk_number(chunk),
                "inserted":  data.get("inserted", data.get("relations_inserted", 0)),
            })

    return {
        "book": book_label,
        "extensions": extensions,
        "reclassifications": reclassifications,
        "inter_chunk_passes": inter_chunk_passes,
        "total_new_classes": sum(len(e["new_classes"]) for e in extensions),
        "total_rescued":     sum(r["rescued"] for r in reclassifications),
        "total_inter_chunk": sum(p["inserted"] for p in inter_chunk_passes),
    }


def _chunk_number(chunk_id):
    """Extract a numeric position from a chunk_id string for timeline plotting."""
    import re
    nums = re.findall(r"chunk_(\d+)", chunk_id)
    return int(nums[-1]) if nums else 0


def analyze_ontology_evolution(ascher_events, morton_events, graph):
    nodes = graph["nodes"]

    ascher_timeline = extract_ontology_timeline(ascher_events, "Ascher-Greif")
    morton_timeline = extract_ontology_timeline(morton_events, "Morton-Mayers")

    # ── All unique classes added across both runs ──────────────────────────
    all_classes = set()
    for e in ascher_timeline["extensions"]:
        all_classes.update(e["new_classes"])
    for e in morton_timeline["extensions"]:
        all_classes.update(e["new_classes"])

    # ── How many graph nodes are of each extended type? ────────────────────
    from collections import Counter
    node_type_counts = Counter(d["type"] for d in nodes.values())
    extended_node_counts = {t: node_type_counts.get(t, 0) for t in all_classes}

    # ── Classes that appeared in both runs (shared discovery) ─────────────
    ascher_classes = set()
    for e in ascher_timeline["extensions"]:
        ascher_classes.update(e["new_classes"])
    morton_classes = set()
    for e in morton_timeline["extensions"]:
        morton_classes.update(e["new_classes"])
    shared = ascher_classes & morton_classes

    # ── Semantic coherence: for each class, sample the nodes assigned to it ─
    class_node_examples = defaultdict(list)
    for name, data in nodes.items():
        t = data["type"]
        if t in all_classes:
            class_node_examples[t].append((name, data["salience"]))
    # Sort examples by salience desc, keep top 5
    for t in class_node_examples:
        class_node_examples[t] = sorted(class_node_examples[t], key=lambda x: -x[1])[:5]

    return {
        "ascher": ascher_timeline,
        "morton": morton_timeline,
        "all_classes_added": sorted(all_classes),
        "ascher_only_classes": sorted(ascher_classes - morton_classes),
        "morton_only_classes": sorted(morton_classes - ascher_classes),
        "shared_classes": sorted(shared),
        "extended_node_counts": extended_node_counts,
        "class_node_examples": dict(class_node_examples),
        "class_descriptions": CLASS_DESCRIPTIONS,
    }


def print_ontology_report(result):
    print(f"\n{'='*70}")
    print("EVAL 4 — ONTOLOGY EVOLUTION ANALYSIS")
    print(f"{'='*70}")

    for timeline_key in ["ascher", "morton"]:
        tl = result[timeline_key]
        print(f"\n  -- {tl['book'].upper()} ONTOLOGY TIMELINE --")
        print(f"  Extensions applied : {len(tl['extensions'])}")
        print(f"  New classes total  : {tl['total_new_classes']}")
        print(f"  Relations rescued  : {tl['total_rescued']}")
        print(f"  Inter-chunk passes : {len(tl['inter_chunk_passes'])}")
        print()
        for ext in tl["extensions"]:
            classes_str = ", ".join(ext["new_classes"]) if ext["new_classes"] else "(none)"
            print(f"    v{ext['version']}  @chunk {ext['chunk_num']:<5}  +classes: {classes_str}")
        if tl["reclassifications"]:
            print(f"\n    Reclassification rescues per version:")
            for r in tl["reclassifications"]:
                print(f"      @chunk {r['chunk_num']:<5}  rescued {r['rescued']} relations")

    print(f"\n  -- ALL NEW CLASSES ACROSS BOTH RUNS ({len(result['all_classes_added'])}) --")
    print(f"  Ascher-only  ({len(result['ascher_only_classes'])}): {', '.join(result['ascher_only_classes'])}")
    print(f"  Morton-only  ({len(result['morton_only_classes'])}): {', '.join(result['morton_only_classes'])}")
    print(f"  Shared       ({len(result['shared_classes'])}): {', '.join(result['shared_classes']) or '(none)'}")

    print(f"\n  -- EXTENDED CLASSES: NODE COUNTS & EXAMPLES --")
    for cls in sorted(result["all_classes_added"]):
        count = result["extended_node_counts"].get(cls, 0)
        desc  = result["class_descriptions"].get(cls, "")
        examples = result["class_node_examples"].get(cls, [])
        example_str = ", ".join(f"'{n}'" for n, _ in examples[:3])
        print(f"\n  {cls}  ({count} nodes)")
        if desc:
            print(f"    Desc: {desc}")
        if example_str:
            print(f"    e.g.: {example_str}")

    print(f"\n  -- INTER-CHUNK RELATION EXTRACTION PASSES --")
    for timeline_key in ["ascher", "morton"]:
        tl = result[timeline_key]
        total_inserted = tl["total_inter_chunk"]
        passes = len(tl["inter_chunk_passes"])
        print(f"  {tl['book']:<20}: {passes} passes, {total_inserted} relations inserted")


def main():
    ascher_events = load_jsonl(ASCHER_JSONL)
    morton_events = load_jsonl(MORTON_JSONL)
    g = load_graph()
    result = analyze_ontology_evolution(ascher_events, morton_events, g)
    print_ontology_report(result)
    return result


if __name__ == "__main__":
    main()
