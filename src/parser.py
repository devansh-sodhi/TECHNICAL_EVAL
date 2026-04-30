import re
import json
import os
from pathlib import Path


def parse_transcript_file(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    header_match = re.search(r"---\s*(T-\d+)\s*\(Agent:\s*([^)]+)\)\s*---", content)
    if not header_match:
        raise ValueError(f"Could not parse header from {filepath}")

    call_id = header_match.group(1).strip()
    agent_name = header_match.group(2).strip()

    turns = []
    turn_number = 0
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("---"):
            continue
        m = re.match(r"^([^:]+):\s+(.+)$", line)
        if m:
            turn_number += 1
            turns.append({
                "call_id": call_id,
                "agent_name": agent_name,
                "turn_number": turn_number,
                "speaker": m.group(1).strip(),
                "text": m.group(2).strip(),
            })

    estimated_duration_minutes = max(1, round(len(turns) / 3))

    full_text = content.lower()
    if "withdrawal" in full_text:
        issue_type = "withdrawal_inquiry"
    elif "suspend" in full_text or "fraud" in full_text:
        issue_type = "account_suspension"
    elif "leverage" in full_text or "forex" in full_text:
        issue_type = "trading_information"
    else:
        issue_type = "general_inquiry"

    resolution_status = "unresolved"
    if any(w in full_text for w in ["have a great day", "have a lovely day"]):
        resolution_status = "resolved"
    elif any(w in full_text for w in ["escalat", "senior reviewer"]):
        resolution_status = "escalated"

    escalation_signals = []
    for turn in turns:
        if turn["speaker"] != agent_name:
            for keyword in ["report", "complain", "unacceptable", "frustrated", "terrible"]:
                if keyword in turn["text"].lower():
                    escalation_signals.append({
                        "turn_number": turn["turn_number"],
                        "signal": keyword,
                        "text": turn["text"],
                    })

    return {
        "call_id": call_id,
        "agent_name": agent_name,
        "turns": turns,
        "estimated_duration_minutes": estimated_duration_minutes,
        "issue_type": issue_type,
        "resolution_status": resolution_status,
        "escalation_signals": escalation_signals,
    }


def parse_all_transcripts(transcripts_dir: str, output_dir: str) -> list:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for txt_file in sorted(Path(transcripts_dir).glob("*.txt")):
        parsed = parse_transcript_file(str(txt_file))
        output_path = os.path.join(output_dir, f"{parsed['call_id']}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)
        print(f"  Parsed {txt_file.name} -> {output_path}")
        results.append(parsed)
    return results
