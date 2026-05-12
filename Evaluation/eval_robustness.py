"""
Eval 5 — Pipeline Robustness
==============================
Comparative analysis of Ascher-Greif vs Morton-Mayers runs across all key
robustness metrics: chunk processing, entity/relation consistency rates,
retry behaviour, rate-limit handling, coreference, and runtime.
"""

import sys, os, json, statistics
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import load_summary, load_jsonl, ASCHER_SUMMARY, MORTON_SUMMARY, ASCHER_JSONL, MORTON_JSONL


def analyze_run(summary, jsonl_events, label):
    chunks = summary["chunks"]
    N = len(chunks)

    # ── Chunk classification breakdown ────────────────────────────────────
    classifier_only = [c for c in chunks if c.get("agent_trace") == ["classifier"]]
    content_chunks  = [c for c in chunks if "entity_extraction" in c.get("agent_trace", [])]
    exercise_flag   = [c for c in chunks if "exercise" in " ".join(c.get("agent_trace", [])).lower()]

    # ── Status breakdown ──────────────────────────────────────────────────
    ok_chunks    = [c for c in chunks if c["status"] == "ok"]
    error_chunks = [c for c in chunks if c["status"] == "error"]

    # Error sub-types
    rate_limit_chunks = [c for c in error_chunks
                         if any("rate_limit_exhausted" in str(p.get("event",""))
                                for p in c.get("problems", []))]
    val_fail_chunks   = [c for c in error_chunks
                         if any("validation_failed" in str(p.get("event",""))
                                for p in c.get("problems", []))]
    val_but_output    = [c for c in val_fail_chunks
                         if c.get("entities_consistent", 0) > 0]

    # ── Retry behaviour ───────────────────────────────────────────────────
    retry_traces = [c for c in chunks
                    if isinstance(c.get("agent_trace"), list)
                    and (c["agent_trace"].count("relation_extraction") > 1
                         or c["agent_trace"].count("entity_extraction") > 1)]
    max_retries_in_trace = 0
    for c in retry_traces:
        t = c.get("agent_trace", [])
        re_count = t.count("relation_extraction") - 1
        ee_count = t.count("entity_extraction") - 1
        max_retries_in_trace = max(max_retries_in_trace, re_count + ee_count)

    # ── Extraction metrics ─────────────────────────────────────────────────
    total_extracted   = sum(c.get("entities_extracted", 0) for c in chunks)
    total_consistent  = sum(c.get("entities_consistent", 0) for c in chunks)
    total_rel_ext     = sum(c.get("relations_extracted", 0) for c in chunks)
    total_rel_cons    = sum(c.get("relations_consistent", 0) for c in chunks)
    total_coref       = sum(c.get("coref_resolutions", 0) for c in chunks)

    ent_consistency   = round(100 * total_consistent / total_extracted, 1) if total_extracted else 0
    rel_consistency   = round(100 * total_rel_cons / total_rel_ext, 1) if total_rel_ext else 0

    # ── Runtime ───────────────────────────────────────────────────────────
    total_elapsed = summary["total_elapsed_s"]
    content_times = [c["elapsed_s"] for c in content_chunks if c["elapsed_s"] > 0]
    mean_chunk_time = round(statistics.mean(content_times), 1) if content_times else 0
    median_chunk_time = round(statistics.median(content_times), 1) if content_times else 0
    max_chunk_time    = round(max(content_times), 1) if content_times else 0

    # ── Validation failure analysis from JSONL ────────────────────────────
    domain_range_violations = defaultdict(int)
    unknown_entity_errors = 0
    for event in jsonl_events:
        if event.get("event") == "validation_failed":
            msg = event.get("data", {})
            if isinstance(msg, dict):
                for err in msg.get("relation_errors", []):
                    if "Domain-range violation" in err:
                        # extract relation name
                        import re
                        m = re.search(r"'([^']+)'", err)
                        if m:
                            domain_range_violations[m.group(1)] += 1
                    if "Unknown target entity" in err:
                        unknown_entity_errors += 1
            elif isinstance(msg, str):
                if "Domain-range violation" in msg:
                    import re
                    for m in re.finditer(r"Domain-range violation: '([^']+)'", msg):
                        domain_range_violations[m.group(1)] += 1

    # ── Agent error counts ────────────────────────────────────────────────
    agent_errors = summary.get("agent_error_counts", {})

    # ── Coreference breakdown ─────────────────────────────────────────────
    coref_chunks = [c for c in content_chunks if c.get("coref_resolutions", 0) > 0]
    coref_per_chunk = [c["coref_resolutions"] for c in coref_chunks]
    mean_coref = round(statistics.mean(coref_per_chunk), 1) if coref_per_chunk else 0

    return {
        "label": label,
        "chunk_counts": {
            "total":         N,
            "skipped":       len(classifier_only),
            "content":       len(content_chunks),
            "error_status":  len(error_chunks),
            "ok_status":     len(ok_chunks),
        },
        "error_breakdown": {
            "rate_limit_exhausted":    len(rate_limit_chunks),
            "validation_failed":       len(val_fail_chunks),
            "validation_but_produced": len(val_but_output),
        },
        "retry_behaviour": {
            "chunks_with_retry_trace": len(retry_traces),
            "max_retries_single_chunk": max_retries_in_trace,
            "explicit_retry_count_field": sum(c.get("retries", 0) for c in chunks),
        },
        "extraction_metrics": {
            "entities_extracted":  total_extracted,
            "entities_consistent": total_consistent,
            "entity_consistency_pct": ent_consistency,
            "relations_extracted":  total_rel_ext,
            "relations_consistent": total_rel_cons,
            "relation_consistency_pct": rel_consistency,
            "coref_resolutions":   total_coref,
            "coref_chunks":        len(coref_chunks),
            "mean_coref_per_chunk": mean_coref,
        },
        "runtime": {
            "total_elapsed_s": total_elapsed,
            "total_elapsed_h": round(total_elapsed / 3600, 2),
            "mean_content_chunk_s": mean_chunk_time,
            "median_content_chunk_s": median_chunk_time,
            "max_content_chunk_s": max_chunk_time,
        },
        "validation_errors": {
            "top_domain_range_violations": dict(
                sorted(domain_range_violations.items(), key=lambda x: -x[1])[:8]
            ),
            "unknown_entity_errors": unknown_entity_errors,
        },
        "agent_error_counts": agent_errors,
    }


def print_robustness_report(ascher, morton):
    print(f"\n{'='*70}")
    print("EVAL 5 — PIPELINE ROBUSTNESS")
    print(f"{'='*70}")

    # Comparative table
    def row(label, a_val, m_val, width=38):
        print(f"  {label:<{width}}  {str(a_val):>14}  {str(m_val):>14}")

    header = f"  {'Metric':<38}  {'Ascher-Greif':>14}  {'Morton-Mayers':>14}"
    print(f"\n{header}")
    print(f"  {'-'*68}")

    # Chunk counts
    row("Total chunks",            ascher["chunk_counts"]["total"],   morton["chunk_counts"]["total"])
    row("Skipped (front/back matter)", ascher["chunk_counts"]["skipped"], morton["chunk_counts"]["skipped"])
    row("Content chunks processed", ascher["chunk_counts"]["content"], morton["chunk_counts"]["content"])
    row("Status: ok",              ascher["chunk_counts"]["ok_status"],    morton["chunk_counts"]["ok_status"])
    row("Status: error",           ascher["chunk_counts"]["error_status"], morton["chunk_counts"]["error_status"])

    print(f"  {'-'*68}")
    row("Error: rate-limit exhausted",   ascher["error_breakdown"]["rate_limit_exhausted"],  morton["error_breakdown"]["rate_limit_exhausted"])
    row("Error: validation failed",      ascher["error_breakdown"]["validation_failed"],     morton["error_breakdown"]["validation_failed"])
    row("  (but still produced output)", ascher["error_breakdown"]["validation_but_produced"], morton["error_breakdown"]["validation_but_produced"])

    print(f"  {'-'*68}")
    row("Chunks with retry traces",      ascher["retry_behaviour"]["chunks_with_retry_trace"], morton["retry_behaviour"]["chunks_with_retry_trace"])
    row("Max retries in single chunk",   ascher["retry_behaviour"]["max_retries_single_chunk"], morton["retry_behaviour"]["max_retries_single_chunk"])

    print(f"  {'-'*68}")
    row("Entities extracted",            ascher["extraction_metrics"]["entities_extracted"],  morton["extraction_metrics"]["entities_extracted"])
    row("Entities consistent",           ascher["extraction_metrics"]["entities_consistent"], morton["extraction_metrics"]["entities_consistent"])
    row("Entity consistency rate",
        f"{ascher['extraction_metrics']['entity_consistency_pct']}%",
        f"{morton['extraction_metrics']['entity_consistency_pct']}%")
    row("Relations extracted",           ascher["extraction_metrics"]["relations_extracted"],  morton["extraction_metrics"]["relations_extracted"])
    row("Relations consistent",          ascher["extraction_metrics"]["relations_consistent"], morton["extraction_metrics"]["relations_consistent"])
    row("Relation consistency rate",
        f"{ascher['extraction_metrics']['relation_consistency_pct']}%",
        f"{morton['extraction_metrics']['relation_consistency_pct']}%")
    row("Coreference resolutions",       ascher["extraction_metrics"]["coref_resolutions"], morton["extraction_metrics"]["coref_resolutions"])
    row("Mean coref/chunk",              ascher["extraction_metrics"]["mean_coref_per_chunk"], morton["extraction_metrics"]["mean_coref_per_chunk"])

    print(f"  {'-'*68}")
    row("Total runtime (hours)",
        f"{ascher['runtime']['total_elapsed_h']}h",
        f"{morton['runtime']['total_elapsed_h']}h")
    row("Mean time/content chunk (s)",   ascher["runtime"]["mean_content_chunk_s"], morton["runtime"]["mean_content_chunk_s"])
    row("Max time/chunk (s)",            ascher["runtime"]["max_content_chunk_s"],  morton["runtime"]["max_content_chunk_s"])

    # Validation error breakdown
    for run in [ascher, morton]:
        print(f"\n  -- TOP VALIDATION ERRORS: {run['label']} --")
        for rel, count in run["validation_errors"]["top_domain_range_violations"].items():
            print(f"    {rel:<40} : {count} violations")
        unk = run["validation_errors"]["unknown_entity_errors"]
        if unk:
            print(f"    Unknown entity errors                    : {unk}")

    # Agent error counts
    for run in [ascher, morton]:
        print(f"\n  -- AGENT ERROR COUNTS: {run['label']} --")
        for agent, count in run["agent_error_counts"].items():
            print(f"    {agent:<35} : {count}")

    # Runtime interpretation
    print(f"\n  -- RUNTIME INTERPRETATION --")
    a_h = ascher["runtime"]["total_elapsed_h"]
    m_h = morton["runtime"]["total_elapsed_h"]
    a_n = ascher["chunk_counts"]["content"]
    m_n = morton["chunk_counts"]["content"]
    a_s = ascher["runtime"]["mean_content_chunk_s"]
    m_s = morton["runtime"]["mean_content_chunk_s"]
    speedup = round(m_s / a_s, 1) if a_s > 0 else "—"
    print(f"  Morton chunks are {speedup}× slower on average ({m_s}s vs {a_s}s)")
    print(f"  Likely cause: Morton content chunks are denser (PDE-heavy, fewer skipped sections)")
    print(f"  Ascher: {a_n} content chunks × {a_s}s = {a_h}h total")
    print(f"  Morton: {m_n} content chunks × {m_s}s = {m_h}h total")


def main():
    ascher_summary = load_summary(ASCHER_SUMMARY)
    morton_summary = load_summary(MORTON_SUMMARY)
    ascher_events  = load_jsonl(ASCHER_JSONL)
    morton_events  = load_jsonl(MORTON_JSONL)

    ascher = analyze_run(ascher_summary, ascher_events, "Ascher-Greif")
    morton = analyze_run(morton_summary, morton_events, "Morton-Mayers")

    print_robustness_report(ascher, morton)
    return {"ascher": ascher, "morton": morton}


if __name__ == "__main__":
    main()
