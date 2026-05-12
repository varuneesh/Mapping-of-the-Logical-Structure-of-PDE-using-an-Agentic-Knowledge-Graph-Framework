"""
Eval 3 — Graph Structure Analysis
===================================
Intrinsic evaluation of the graph's structure: node type distribution,
salience distribution, edge density, relation type distribution,
multi-source edges, and type conflicts.
"""

import sys, os, math, statistics
sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import load_graph, is_cross_book, is_ascher_only, is_morton_only
from collections import Counter


def analyze_structure(graph):
    nodes = graph["nodes"]
    edges = graph["edges"]
    N = len(nodes)
    E = len(edges)

    # ── 1. Node type distribution ─────────────────────────────────────────
    type_counts = Counter(d["type"] for d in nodes.values())

    # ── 2. Salience distribution ──────────────────────────────────────────
    saliences = [d["salience"] for d in nodes.values()]
    sal_stats = {
        "max":    round(max(saliences), 4),
        "mean":   round(statistics.mean(saliences), 4),
        "median": round(statistics.median(saliences), 4),
        "min":    round(min(saliences), 4),
        "stdev":  round(statistics.stdev(saliences), 4),
    }
    sal_buckets = {
        ">=3.0": sum(1 for s in saliences if s >= 3.0),
        ">=2.0": sum(1 for s in saliences if s >= 2.0),
        ">=1.0": sum(1 for s in saliences if s >= 1.0),
        "<1.0":  sum(1 for s in saliences if s < 1.0),
    }

    # ── 3. Top 20 nodes by salience ───────────────────────────────────────
    top20 = sorted(nodes.items(), key=lambda x: -x[1]["salience"])[:20]
    top20_rows = []
    for name, data in top20:
        a = sum(1 for s in data["sources"] if "Ascher" in s)
        m = sum(1 for s in data["sources"] if "Morton" in s)
        top20_rows.append({
            "name": name,
            "salience": data["salience"],
            "type": data["type"],
            "ascher_chunks": a,
            "morton_chunks": m,
            "type_conflict": data.get("type_conflict", False),
        })

    # ── 4. Edge density ───────────────────────────────────────────────────
    avg_degree = round(2 * E / N, 2) if N else 0   # undirected equivalent
    avg_out_degree = round(E / N, 2) if N else 0

    # Out-degree distribution
    out_deg = Counter(e["source"] for e in edges)
    in_deg  = Counter(e["target"] for e in edges)
    max_out = max(out_deg.values()) if out_deg else 0
    max_in  = max(in_deg.values())  if in_deg  else 0
    top_out = out_deg.most_common(10)
    top_in  = in_deg.most_common(10)

    # ── 5. Relation type distribution ─────────────────────────────────────
    rel_counts = Counter(e["relation"] for e in edges)

    # ── 6. Multi-source edges ─────────────────────────────────────────────
    multi_source = [e for e in edges if len(e.get("sources", [])) > 1]
    inter_chunk  = [e for e in edges if e.get("inter_chunk", False)]

    multi_src_rels = Counter(e["relation"] for e in multi_source)
    multi_src_mean_sources = (
        statistics.mean(len(e["sources"]) for e in multi_source)
        if multi_source else 0
    )

    # ── 7. Type conflicts ─────────────────────────────────────────────────
    conflict_nodes = {n: d for n, d in nodes.items() if d.get("type_conflict", False)}
    conflict_type_pairs = Counter()
    for data in conflict_nodes.values():
        observed = tuple(sorted(data.get("observed_types", [])))
        if len(observed) >= 2:
            conflict_type_pairs[observed] += 1

    # ── 8. Source count distribution ──────────────────────────────────────
    source_counts = [len(d["sources"]) for d in nodes.values()]
    sc_dist = {
        "single_chunk": sum(1 for s in source_counts if s == 1),
        "2_to_5":       sum(1 for s in source_counts if 2 <= s <= 5),
        "6_to_20":      sum(1 for s in source_counts if 6 <= s <= 20),
        ">20":          sum(1 for s in source_counts if s > 20),
    }

    return {
        "counts": {"nodes": N, "edges": E},
        "type_distribution": dict(type_counts.most_common()),
        "salience_stats": sal_stats,
        "salience_buckets": sal_buckets,
        "top20_nodes": top20_rows,
        "edge_density": {
            "avg_out_degree": avg_out_degree,
            "avg_degree_undirected": avg_degree,
            "max_out_degree": max_out,
            "max_in_degree":  max_in,
            "top_out_degree": top_out,
            "top_in_degree":  top_in,
        },
        "relation_distribution": dict(rel_counts.most_common()),
        "multi_source_edges": {
            "count": len(multi_source),
            "pct_of_total": round(100 * len(multi_source) / E, 1) if E else 0,
            "mean_sources_per_multi": round(multi_src_mean_sources, 2),
            "top_relations": dict(multi_src_rels.most_common(5)),
        },
        "inter_chunk_edges": {
            "count": len(inter_chunk),
            "pct_of_total": round(100 * len(inter_chunk) / E, 1) if E else 0,
        },
        "type_conflicts": {
            "count": len(conflict_nodes),
            "pct_of_nodes": round(100 * len(conflict_nodes) / N, 1) if N else 0,
            "conflict_pair_counts": {str(k): v for k, v in conflict_type_pairs.most_common(10)},
        },
        "source_count_distribution": sc_dist,
    }


def print_structure_report(result):
    print(f"\n{'='*70}")
    print("EVAL 3 — GRAPH STRUCTURE ANALYSIS")
    print(f"{'='*70}")

    c = result["counts"]
    print(f"\n  Total nodes : {c['nodes']}")
    print(f"  Total edges : {c['edges']}")

    print(f"\n  -- NODE TYPE DISTRIBUTION --")
    total_n = c["nodes"]
    for t, count in result["type_distribution"].items():
        bar = "█" * int(30 * count / total_n)
        print(f"  {t:<35} {count:>4}  ({100*count/total_n:>5.1f}%)  {bar}")

    print(f"\n  -- SALIENCE DISTRIBUTION --")
    ss = result["salience_stats"]
    print(f"  max={ss['max']:.4f}  mean={ss['mean']:.4f}  median={ss['median']:.4f}  min={ss['min']:.4f}  σ={ss['stdev']:.4f}")
    for bucket, count in result["salience_buckets"].items():
        print(f"    Nodes {bucket:<6}: {count}")

    print(f"\n  -- SOURCE COUNT DISTRIBUTION --")
    for bucket, count in result["source_count_distribution"].items():
        print(f"    {bucket:<14}: {count}")

    print(f"\n  -- TOP 20 NODES BY SALIENCE --")
    print(f"  {'Name':<45} {'Sal':>6}  {'A':>4} {'M':>4}  Type")
    print(f"  {'-'*80}")
    for row in result["top20_nodes"]:
        conflict = "*" if row["type_conflict"] else " "
        print(f"  {conflict}{row['name']:<44} {row['salience']:>6.4f}  {row['ascher_chunks']:>4} {row['morton_chunks']:>4}  {row['type']}")

    print(f"\n  -- EDGE DENSITY --")
    d = result["edge_density"]
    print(f"  Mean out-degree         : {d['avg_out_degree']}")
    print(f"  Mean degree (undirected): {d['avg_degree_undirected']}")
    print(f"  Max out-degree          : {d['max_out_degree']}")
    print(f"  Max in-degree           : {d['max_in_degree']}")
    print(f"  Top nodes by out-degree:")
    for name, deg in d["top_out_degree"]:
        print(f"    {name:<45} out={deg}")
    print(f"  Top nodes by in-degree:")
    for name, deg in d["top_in_degree"]:
        print(f"    {name:<45} in={deg}")

    print(f"\n  -- RELATION TYPE DISTRIBUTION --")
    total_e = c["edges"]
    for rel, count in result["relation_distribution"].items():
        bar = "█" * int(25 * count / total_e)
        print(f"  {rel:<30} {count:>5}  ({100*count/total_e:>5.1f}%)  {bar}")

    print(f"\n  -- MULTI-SOURCE EDGES --")
    ms = result["multi_source_edges"]
    print(f"  Count              : {ms['count']}  ({ms['pct_of_total']}% of all edges)")
    print(f"  Mean sources/edge  : {ms['mean_sources_per_multi']}")
    print(f"  Top relations      : {ms['top_relations']}")

    print(f"\n  -- INTER-CHUNK EDGES --")
    ic = result["inter_chunk_edges"]
    print(f"  Count : {ic['count']}  ({ic['pct_of_total']}% of all edges)")

    print(f"\n  -- TYPE CONFLICTS --")
    tc = result["type_conflicts"]
    print(f"  Nodes with type conflicts : {tc['count']}  ({tc['pct_of_nodes']}% of nodes)")
    print(f"  Most common conflict pairs:")
    for pair, count in tc["conflict_pair_counts"].items():
        print(f"    {pair:<60} : {count}")


def main():
    g = load_graph()
    result = analyze_structure(g)
    print_structure_report(result)
    return result


if __name__ == "__main__":
    main()
