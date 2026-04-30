import json
import os
import shutil
from pathlib import Path

from src.parser import parse_all_transcripts
from src.agents_module import (
    run_qa_scoring,
    run_compliance_extraction,
    run_coaching,
    MODEL,
    PROVIDER,
)
from src.dashboard import compute_dashboard, dashboard_to_markdown
from src.team_trend import compute_team_trend
from src.escalation import get_escalation_cases
from src.rebuttal import generate_rebuttal_coaching, rebuttal_to_markdown

STAGES = [
    "INIT",
    "INPUTS_LOADED",
    "TRANSCRIPTS_PARSED",
    "QA_SCORED",
    "COMPLIANCE_EXTRACTED",
    "COACHING_GENERATED",
    "DASHBOARD_COMPUTED",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
]

STATE_FILE = "pipeline_state.json"


def _save_state(stage: str):
    with open(STATE_FILE, "w") as f:
        json.dump({"stage": stage}, f, indent=2)
    print(f"[PIPELINE] -- {stage}")


def coaching_to_markdown(coaching_results: list) -> str:
    lines = ["# Agent Coaching Notes\n"]
    for notes in coaching_results:
        agent = notes.get("agent_name", "Unknown")
        call_id = notes.get("call_id", "Unknown")
        lines.append(f"## {agent} — {call_id}\n")
        lines.append("### What Went Well\n")
        for item in notes.get("what_went_well", []):
            lines.append(f"**{item.get('point', '')}**")
            lines.append(f'> "{item.get("agent_quote", "")}"')
            lines.append(f"*{item.get('context', '')}*\n")
        lines.append("### Development Areas\n")
        for item in notes.get("development_areas", []):
            lines.append(f"**{item.get('area', '')}**")
            lines.append(f'> "{item.get("agent_quote", "")}"')
            lines.append(f"Suggested: *\"{item.get('suggested_alternative', '')}\"*")
            lines.append(f"*{item.get('why', '')}*\n")
        lines.append("---\n")
    return "\n".join(lines)


def run_pipeline():
    # Clean all generated artifacts so pipeline always regenerates from raw inputs
    for f in [
        "llm_calls.jsonl", "qa_scores.json", "compliance_flags.json",
        "coaching_notes.md", "dashboard_summary.md", "calibration_check.json",
        "team_trend.json", "escalation_cases.json", "rebuttal_coaching.md",
        "pipeline_state.json",
    ]:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists("parsed_transcripts"):
        shutil.rmtree("parsed_transcripts")

    _save_state("INIT")
    print("\n[PIPELINE] Starting pipeline run...\n")

    # INIT -> INPUTS_LOADED
    print("[1/9] Loading inputs...")
    with open("qa_framework.json") as f:
        framework = json.load(f)
    _save_state("INPUTS_LOADED")

    # INPUTS_LOADED -> TRANSCRIPTS_PARSED
    print("[2/9] Parsing transcripts...")
    parsed_transcripts = parse_all_transcripts("transcripts", "parsed_transcripts")
    _save_state("TRANSCRIPTS_PARSED")

    # TRANSCRIPTS_PARSED -> QA_SCORED
    # Stage 1: QA Evaluator agent — one call per transcript
    print("[3/9] QA scoring — QA Evaluator agent (Stage 1)...")
    qa_scores = []
    for pt in parsed_transcripts:
        print(f"  Scoring {pt['call_id']}...")
        qa_scores.append(run_qa_scoring(pt, framework))
    with open("qa_scores.json", "w") as f:
        json.dump(qa_scores, f, indent=2)
    _save_state("QA_SCORED")

    # QA_SCORED -> COMPLIANCE_EXTRACTED
    # Stage 2: Compliance Analyst agent — one call per transcript
    print("[4/9] Compliance extraction — Compliance Analyst agent (Stage 2)...")
    compliance_results = []
    for pt in parsed_transcripts:
        print(f"  Analysing {pt['call_id']}...")
        compliance_results.append(run_compliance_extraction(pt))
    with open("compliance_flags.json", "w") as f:
        json.dump(compliance_results, f, indent=2)
    _save_state("COMPLIANCE_EXTRACTED")

    # COMPLIANCE_EXTRACTED -> COACHING_GENERATED
    # Stage 3: Coach agent — one call per transcript, NO QA scores passed
    print("[5/9] Coaching — Coach agent (Stage 3, no QA scores)...")
    coaching_results = []
    for pt in parsed_transcripts:
        print(f"  Coaching {pt['call_id']}...")
        coaching_results.append(run_coaching(pt))
    with open("coaching_notes.md", "w") as f:
        f.write(coaching_to_markdown(coaching_results))
    _save_state("COACHING_GENERATED")

    # COACHING_GENERATED -> DASHBOARD_COMPUTED
    # Dashboard requires both QA_SCORED and COMPLIANCE_EXTRACTED (enforced by stage ordering above)
    print("[6/9] Computing dashboard...")
    dashboard_rows = compute_dashboard(qa_scores, compliance_results)
    with open("dashboard_summary.md", "w") as f:
        f.write(dashboard_to_markdown(dashboard_rows))

    print("[6b] Team performance trend (SHOULD)...")
    team_trend = compute_team_trend(qa_scores)
    with open("team_trend.json", "w") as f:
        json.dump(team_trend, f, indent=2)

    print("[6c] Auto-escalation cases (STRETCH)...")
    escalation_cases = get_escalation_cases(qa_scores, compliance_results, parsed_transcripts)
    with open("escalation_cases.json", "w") as f:
        json.dump(escalation_cases, f, indent=2)

    print("[6d] Rebuttal coaching (STRETCH)...")
    rebuttal_scenarios = generate_rebuttal_coaching(parsed_transcripts)
    if rebuttal_scenarios:
        with open("rebuttal_coaching.md", "w") as f:
            f.write(rebuttal_to_markdown(rebuttal_scenarios))

    print("[6e] Calibration check (SHOULD)...")
    from src.calibration import run_calibration_check
    calibration = run_calibration_check(qa_scores)
    with open("calibration_check.json", "w") as f:
        json.dump(calibration, f, indent=2)

    _save_state("DASHBOARD_COMPUTED")
    _save_state("VALIDATION_COMPLETE")
    _save_state("RESULTS_FINALISED")

    print("\n[PIPELINE] Pipeline complete! Run: python validate.py")


if __name__ == "__main__":
    run_pipeline()
