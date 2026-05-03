"""
logger.py

Structured logger for the KG construction pipeline.

Every agent calls logger.log(agent, event, data) — the logger writes to:
  1. A rotating JSON-lines file (one record per event) — machine readable,
     easy to grep/filter after a run.
  2. A human-readable run summary that collects per-chunk outcomes so you
     can scan the whole run at a glance and spot which chunks caused problems.

Usage in any agent:
    from kg_agents.utils.logger import PipelineLogger
    # Logger is passed in via state["logger"] — same instance across all chunks.

    logger = state["logger"]
    logger.log("EntityExtractionAgent", "extracted", {
        "count": len(entities),
        "entities": [e["name"] for e in entities]
    })
    logger.log("EntityExtractionAgent", "warning", {
        "message": "LLM returned no entities for this chunk"
    })

After a full run, call:
    logger.save_summary()   — writes human-readable summary JSON
    logger.print_problem_chunks()  — prints only chunks with warnings/errors
"""

import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from collections import defaultdict


# ── Log levels ───────────────────────────────────────────────────────────────
INFO    = "INFO"
WARNING = "WARNING"
ERROR   = "ERROR"
DEBUG   = "DEBUG"

# Events that count as a problem (surfaced in problem chunk report)
_PROBLEM_EVENTS = {
    "validation_failed", "retry_triggered", "max_retries_exceeded",
    "parse_error", "llm_error", "type_conflict", "coref_failed",
    "candidate_rejected", "alignment_failed", "warning", "error"
}


class PipelineLogger:
    """
    Structured per-chunk, per-agent logger for the KG pipeline.

    Parameters
    ----------
    run_name : str
        Identifier for this run (e.g. "ascher_greif_full_run").
    log_dir  : str | Path
        Directory where log files are written.
    """

    def __init__(self, run_name: str, log_dir: str = "logs"):

        self.run_name  = run_name
        self.log_dir   = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.jsonl_path = self.log_dir / f"{run_name}_{timestamp}.jsonl"
        self.summary_path = self.log_dir / f"{run_name}_{timestamp}_summary.json"

        # In-memory structures
        self._current_chunk_id: Optional[str] = None
        self._chunk_records: Dict[str, Dict]  = {}   # chunk_id → summary record
        self._run_start = time.time()

        # Open JSONL file for appending
        self._file = open(self.jsonl_path, "a", encoding="utf-8")

        self._write_raw({
            "event":    "run_started",
            "run_name": run_name,
            "time":     datetime.now().isoformat()
        })

        print(f"[Logger] Run '{run_name}' → {self.jsonl_path}")

    # ------------------------------------------------------------------ #
    #  Chunk lifecycle                                                     #
    # ------------------------------------------------------------------ #

    def start_chunk(self, chunk_id: str, heading: str = "") -> None:
        """Call at the start of each chunk before any agent runs."""
        self._current_chunk_id = chunk_id
        self._chunk_records[chunk_id] = {
            "chunk_id":    chunk_id,
            "heading":     heading,
            "start_time":  time.time(),
            "agents_run":  [],
            "problems":    [],          # list of {agent, event, message}
            "entities_extracted":   0,
            "relations_extracted":  0,
            "entities_consistent":  0,
            "relations_consistent": 0,
            "coref_resolutions":    0,
            "retries":              0,
            "status":               "running",
        }
        self._write_raw({
            "chunk_id": chunk_id,
            "event":    "chunk_started",
            "heading":  heading
        })

    def end_chunk(self, chunk_id: str, final_state: Dict = None) -> None:
        """Call after graph_builder completes for this chunk."""
        rec = self._chunk_records.get(chunk_id, {})
        rec["elapsed_s"] = round(time.time() - rec.get("start_time", time.time()), 2)
        rec["status"]    = "error" if rec["problems"] else "ok"

        if final_state:
            rec["entities_extracted"]   = len(final_state.get("entities", []))
            rec["relations_extracted"]  = len(final_state.get("relationships", []))
            rec["entities_consistent"]  = len(final_state.get("consistent_entities", []))
            rec["relations_consistent"] = len(final_state.get("consistent_relationships", []))
            rec["coref_resolutions"]    = len(final_state.get("coreference_annotations", {}))
            rec["agent_trace"]          = final_state.get("agent_trace", [])

        self._write_raw({
            "chunk_id": chunk_id,
            "event":    "chunk_ended",
            "status":   rec["status"],
            "elapsed_s": rec.get("elapsed_s")
        })

    # ------------------------------------------------------------------ #
    #  Core logging method                                                 #
    # ------------------------------------------------------------------ #

    def log(
        self,
        agent:   str,
        event:   str,
        data:    Dict[str, Any] = None,
        level:   str = INFO,
    ) -> None:
        """
        Log a structured event from an agent.

        Parameters
        ----------
        agent : str   e.g. "EntityExtractionAgent"
        event : str   e.g. "extracted", "retry_triggered", "parse_error"
        data  : dict  any additional context
        level : str   INFO | WARNING | ERROR | DEBUG
        """
        chunk_id = self._current_chunk_id or "no_chunk"
        record   = {
            "ts":       datetime.now().isoformat(),
            "chunk_id": chunk_id,
            "agent":    agent,
            "event":    event,
            "level":    level,
            "data":     data or {}
        }

        self._write_raw(record)

        # Update chunk summary
        if chunk_id in self._chunk_records:
            rec = self._chunk_records[chunk_id]

            if agent not in rec["agents_run"]:
                rec["agents_run"].append(agent)

            if event in _PROBLEM_EVENTS or level in (WARNING, ERROR):
                rec["problems"].append({
                    "agent":   agent,
                    "event":   event,
                    "message": data.get("message", str(data)) if data else ""
                })

            if event == "retry_triggered":
                rec["retries"] += 1

        # Console output — only warnings and errors to keep terminal clean
        if level in (WARNING, ERROR):
            print(f"  [{level}] [{agent}] {event}: "
                  f"{data.get('message', '') if data else ''}")

    # ------------------------------------------------------------------ #
    #  Convenience methods                                                 #
    # ------------------------------------------------------------------ #

    def info(self, agent: str, event: str, data: Dict = None) -> None:
        self.log(agent, event, data, INFO)

    def warning(self, agent: str, event: str, data: Dict = None) -> None:
        self.log(agent, event, data, WARNING)

    def error(self, agent: str, event: str, data: Dict = None) -> None:
        self.log(agent, event, data, ERROR)

    def log_exception(self, agent: str, exc: Exception, context: Dict = None) -> None:
        """Log a caught exception with full traceback."""
        self.log(agent, "exception", {
            "exception_type": type(exc).__name__,
            "message":        str(exc),
            "traceback":      traceback.format_exc(),
            **(context or {})
        }, ERROR)

    # ------------------------------------------------------------------ #
    #  Summary and reporting                                               #
    # ------------------------------------------------------------------ #

    def save_summary(self) -> Path:
        """
        Write the full run summary JSON — one record per chunk plus
        aggregate stats. Call once after all chunks are processed.
        """
        total_chunks   = len(self._chunk_records)
        problem_chunks = [
            r for r in self._chunk_records.values() if r["problems"]
        ]
        ok_chunks = total_chunks - len(problem_chunks)

        # Aggregate agent-level error counts
        agent_errors: Dict[str, int] = defaultdict(int)
        for rec in problem_chunks:
            for p in rec["problems"]:
                agent_errors[p["agent"]] += 1

        summary = {
            "run_name":       self.run_name,
            "total_chunks":   total_chunks,
            "ok_chunks":      ok_chunks,
            "problem_chunks": len(problem_chunks),
            "total_elapsed_s": round(time.time() - self._run_start, 2),
            "agent_error_counts": dict(agent_errors),
            "chunks": list(self._chunk_records.values())
        }

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[Logger] Summary saved → {self.summary_path}")
        print(f"  Total chunks : {total_chunks}")
        print(f"  OK           : {ok_chunks}")
        print(f"  Problems     : {len(problem_chunks)}")
        if agent_errors:
            print("  Errors by agent:")
            for agent, count in sorted(
                agent_errors.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"    {agent:40s} {count} problem(s)")

        return self.summary_path

    def print_problem_chunks(self, max_chunks: int = 20) -> None:
        """
        Print a concise report of all chunks that had warnings or errors.
        Call this during development to quickly spot which chunks to investigate.
        """
        problems = [
            r for r in self._chunk_records.values() if r["problems"]
        ]

        if not problems:
            print("[Logger] No problem chunks — clean run.")
            return

        print(f"\n[Logger] {len(problems)} problem chunk(s):\n")

        for rec in problems[:max_chunks]:
            print(f"  Chunk : {rec['chunk_id']}")
            print(f"  Section: {rec.get('heading', '')}")
            print(f"  Retries: {rec['retries']}")
            for p in rec["problems"]:
                print(f"    ✗ [{p['agent']}] {p['event']}: {p['message']}")
            print()

    def get_agent_summary(self) -> Dict[str, Dict]:
        """
        Return per-agent statistics across the full run.
        Useful for identifying which agent is the most frequent source of issues.
        """
        agent_stats: Dict[str, Dict] = defaultdict(lambda: {
            "chunks_involved": 0,
            "total_problems":  0,
            "problem_events":  defaultdict(int)
        })

        for rec in self._chunk_records.values():
            agents_with_problems = set()
            for p in rec["problems"]:
                a = p["agent"]
                agent_stats[a]["total_problems"] += 1
                agent_stats[a]["problem_events"][p["event"]] += 1
                agents_with_problems.add(a)
            for a in agents_with_problems:
                agent_stats[a]["chunks_involved"] += 1

        return {k: dict(v) for k, v in agent_stats.items()}

    def close(self) -> None:
        """Flush and close the JSONL log file."""
        self._write_raw({
            "event":      "run_ended",
            "elapsed_s":  round(time.time() - self._run_start, 2)
        })
        self._file.close()

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _write_raw(self, record: Dict) -> None:
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()   # ensure nothing is lost if pipeline crashes