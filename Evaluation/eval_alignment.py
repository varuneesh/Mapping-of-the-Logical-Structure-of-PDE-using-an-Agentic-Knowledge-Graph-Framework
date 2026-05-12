"""
Eval 2 — Cross-Book Alignment Quality
======================================
Analyses the 85 nodes that were successfully aligned across both Ascher-Greif
and Morton-Mayers, demonstrating that the alignment agent works across different
authors, notation styles, and writing conventions.

Key claims this evaluation supports:
  - 85 nodes merged across both books (17.7% of the 480-node graph)
  - Merged nodes have higher salience than single-book nodes (salience boost)
  - Specific high-value cross-book merges (truncation error, stability, etc.)
  - The type distribution of cross-book nodes vs single-book nodes
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import (load_graph, node_book_counts, is_cross_book,
                        is_ascher_only, is_morton_only)
import statistics


def analyze_alignment(graph):
    nodes = graph["nodes"]
    edges = graph["edges"]

    # ── 1. Partition nodes ─────────────────────────────────────────────────
    cross  = {n: d for n, d in nodes.items() if is_cross_book(d)}
    ascher = {n: d for n, d in nodes.items() if is_ascher_only(d)}
    morton = {n: d for n, d in nodes.items() if is_morton_only(d)}

    # ── 2. Salience comparison ─────────────────────────────────────────────
    def sal_stats(node_dict):
        sals = [d["salience"] for d in node_dict.values()]
        if not sals:
            return {}
        return {
            "count": len(sals),
            "mean":  round(statistics.mean(sals), 4),
            "median": round(statistics.median(sals), 4),
            "max":   round(max(sals), 4),
            "stdev": round(statistics.stdev(sals), 4) if len(sals) > 1 else 0,
        }

    cross_stats  = sal_stats(cross)
    ascher_stats = sal_stats(ascher)
    morton_stats = sal_stats(morton)

    # Salience boost: ratio of cross-book mean to single-book means
    single_sals = [d["salience"] for d in ascher.values()] + \
                  [d["salience"] for d in morton.values()]
    single_mean = statistics.mean(single_sals) if single_sals else 0
    boost = cross_stats["mean"] / single_mean if single_mean else 0

    # ── 3. Type distribution of cross-book nodes ───────────────────────────
    from collections import Counter
    cross_types  = Counter(d["type"] for d in cross.values())
    ascher_types = Counter(d["type"] for d in ascher.values())
    morton_types = Counter(d["type"] for d in morton.values())

    # ── 4. Top cross-book nodes by salience ────────────────────────────────
    top_cross = sorted(cross.items(), key=lambda x: -x[1]["salience"])

    def cross_row(name, data):
        a, m = node_book_counts(data)
        return {
            "name": name,
            "salience": data["salience"],
            "type": data["type"],
            "ascher_chunks": a,
            "morton_chunks": m,
            "total_chunks":  a + m,
            "type_conflict": data.get("type_conflict", False),
        }

    top_cross_rows = [cross_row(n, d) for n, d in top_cross[:30]]

    # ── 5. Hypothetical single-book salience (for boost illustration) ───────
    # If a cross-book node had only its Ascher sources, what would its salience be?
    # salience = mean_conf * log(1 + source_count)
    import math
    def hypothetical_salience(data, book_filter):
        sources = [s for s in data["sources"] if book_filter in s]
        if not sources:
            return 0.0
        confs = data["confidence_scores"]
        # Approximation: use same mean confidence (we don't have per-source conf)
        mean_conf = statistics.mean(confs) if confs else 0.9
        return mean_conf * math.log(1 + len(sources))

    boost_examples = []
    for name, data in top_cross[:15]:
        sal_actual = data["salience"]
        sal_a = hypothetical_salience(data, "Ascher")
        sal_m = hypothetical_salience(data, "Morton")
        boost_examples.append({
            "name": name,
            "actual_salience":       round(sal_actual, 4),
            "ascher_only_salience":  round(sal_a, 4),
            "morton_only_salience":  round(sal_m, 4),
            "boost_vs_ascher":       round(sal_actual / sal_a, 2) if sal_a > 0 else None,
            "boost_vs_morton":       round(sal_actual / sal_m, 2) if sal_m > 0 else None,
        })

    # ── 6. Cross-book edges ────────────────────────────────────────────────
    cross_names = set(cross.keys())
    edges_both_cross = [e for e in edges
                        if e["source"] in cross_names and e["target"] in cross_names]
    edges_cross_to_ascher = [e for e in edges
                             if e["source"] in cross_names and e["target"] in ascher
                             or e["target"] in cross_names and e["source"] in ascher]
    edges_cross_to_morton = [e for e in edges
                             if e["source"] in cross_names and e["target"] in morton
                             or e["target"] in cross_names and e["source"] in morton]

    return {
        "totals": {
            "total_nodes": len(nodes),
            "cross_book":   len(cross),
            "ascher_only":  len(ascher),
            "morton_only":  len(morton),
            "cross_pct":    round(100 * len(cross) / len(nodes), 1),
        },
        "salience_stats": {
            "cross_book":   cross_stats,
            "ascher_only":  ascher_stats,
            "morton_only":  morton_stats,
            "single_book_mean": round(single_mean, 4),
            "cross_book_salience_boost": round(boost, 2),
        },
        "type_distribution": {
            "cross_book":  dict(cross_types.most_common()),
            "ascher_only": dict(ascher_types.most_common(10)),
            "morton_only": dict(morton_types.most_common(10)),
        },
        "top_cross_book_nodes": top_cross_rows,
        "salience_boost_examples": boost_examples,
        "cross_book_edges": {
            "both_endpoints_cross":  len(edges_both_cross),
            "cross_to_ascher_only":  len(edges_cross_to_ascher),
            "cross_to_morton_only":  len(edges_cross_to_morton),
        },
    }


def print_alignment_report(result):
    t = result["totals"]
    s = result["salience_stats"]

    print(f"\n{'='*70}")
    print("EVAL 2 — CROSS-BOOK ALIGNMENT QUALITY")
    print(f"{'='*70}")

    print(f"\n  -- NODE PARTITION --")
    print(f"  Total nodes       : {t['total_nodes']}")
    print(f"  Cross-book merged : {t['cross_book']}  ({t['cross_pct']}% of graph)")
    print(f"  Ascher-only       : {t['ascher_only']}")
    print(f"  Morton-only       : {t['morton_only']}")

    print(f"\n  -- SALIENCE COMPARISON --")
    for key, label in [("cross_book","Cross-book nodes"), ("ascher_only","Ascher-only nodes"), ("morton_only","Morton-only nodes")]:
        st = s[key]
        if st:
            print(f"  {label:<24}: mean={st['mean']:.4f}, median={st['median']:.4f}, max={st['max']:.4f}, σ={st['stdev']:.4f}")
    print(f"  Single-book mean  : {s['single_book_mean']:.4f}")
    print(f"  Cross-book boost  : {s['cross_book_salience_boost']:.2f}×  (cross-book mean / single-book mean)")

    print(f"\n  -- TYPE DISTRIBUTION (CROSS-BOOK NODES) --")
    for t_type, count in sorted(result["type_distribution"]["cross_book"].items(), key=lambda x: -x[1]):
        print(f"    {t_type:<35} {count:>4}")

    print(f"\n  -- TOP 20 CROSS-BOOK NODES (by salience) --")
    print(f"  {'Name':<45} {'Sal':>6}  {'A':>4} {'M':>4}  {'Type'}")
    print(f"  {'-'*85}")
    for row in result["top_cross_book_nodes"][:20]:
        conflict = " *" if row["type_conflict"] else ""
        print(f"  {row['name']:<45} {row['salience']:>6.3f}  {row['ascher_chunks']:>4} {row['morton_chunks']:>4}  {row['type']}{conflict}")

    print(f"\n  -- SALIENCE BOOST EXAMPLES (actual vs hypothetical single-book) --")
    print(f"  {'Name':<45} {'Actual':>7}  {'A-only':>7}  {'M-only':>7}  {'Boost(A)':>9}  {'Boost(M)':>9}")
    print(f"  {'-'*100}")
    for row in result["salience_boost_examples"][:15]:
        ba = f"{row['boost_vs_ascher']:.2f}×" if row["boost_vs_ascher"] else "  —  "
        bm = f"{row['boost_vs_morton']:.2f}×" if row["boost_vs_morton"] else "  —  "
        print(f"  {row['name']:<45} {row['actual_salience']:>7.4f}  {row['ascher_only_salience']:>7.4f}  {row['morton_only_salience']:>7.4f}  {ba:>9}  {bm:>9}")

    print(f"\n  -- CROSS-BOOK EDGE CONNECTIVITY --")
    ce = result["cross_book_edges"]
    print(f"  Edges between two cross-book nodes : {ce['both_endpoints_cross']}")
    print(f"  Edges: cross-book → Ascher-only    : {ce['cross_to_ascher_only']}")
    print(f"  Edges: cross-book → Morton-only    : {ce['cross_to_morton_only']}")


def main():
    g = load_graph()
    result = analyze_alignment(g)
    print_alignment_report(result)
    return result


if __name__ == "__main__":
    main()
