"""
eval_master.py — Run all evaluations and produce a single JSON report.
Usage: python eval_master.py [--json output.json]
"""

import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

from eval_utils import load_graph, load_jsonl, ASCHER_JSONL, MORTON_JSONL

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None, help="Write combined results to this JSON file")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("KNOWLEDGE GRAPH EVALUATION — MASTER REPORT")
    print("="*70)

    # ── Eval 1: Recall ────────────────────────────────────────────────────
    print("\n>>> Running Eval 1: Entity Recall")
    from eval_recall import main as recall_main
    recall_results = recall_main()

    # ── Eval 2: Alignment ─────────────────────────────────────────────────
    print("\n>>> Running Eval 2: Cross-Book Alignment")
    from eval_alignment import main as alignment_main
    alignment_results = alignment_main()

    # ── Eval 3: Graph Structure ───────────────────────────────────────────
    print("\n>>> Running Eval 3: Graph Structure")
    from eval_graph_structure import main as structure_main
    structure_results = structure_main()

    # ── Eval 4: Ontology Evolution ────────────────────────────────────────
    print("\n>>> Running Eval 4: Ontology Evolution")
    from eval_ontology import main as ontology_main
    ontology_results = ontology_main()

    # ── Eval 5: Robustness ────────────────────────────────────────────────
    print("\n>>> Running Eval 5: Pipeline Robustness")
    from eval_robustness import main as robustness_main
    robustness_results = robustness_main()

    # ── Combined summary ──────────────────────────────────────────────────
    combined = {
        "eval_1_recall":           recall_results,
        "eval_2_alignment":        alignment_results,
        "eval_3_graph_structure":  structure_results,
        "eval_4_ontology":         ontology_results,
        "eval_5_robustness":       robustness_results,
    }

    if args.json:
        # Convert non-serialisable objects
        def make_serialisable(obj):
            if isinstance(obj, dict):
                return {k: make_serialisable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serialisable(i) for i in obj]
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)

        with open(args.json, "w") as f:
            json.dump(make_serialisable(combined), f, indent=2)
        print(f"\n✓ Combined results written to: {args.json}")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    return combined


if __name__ == "__main__":
    main()
