import json
import os
import re
import sys
from pathlib import Path

errors = []
passes = []


def _ok(msg):
    passes.append(msg)
    print(f"  OK   {msg}")


def _fail(msg):
    errors.append(msg)
    print(f"  FAIL {msg}")


def _check(condition, ok_msg, fail_msg):
    _ok(ok_msg) if condition else _fail(fail_msg)


def validate():
    print("=" * 60)
    print("Pipeline Validation")
    print("=" * 60)

    # 1. Required artifacts
    print("\n[1] Required artifacts")
    for artifact in [
        "qa_framework.json", "qa_scores.json", "compliance_flags.json",
        "coaching_notes.md", "dashboard_summary.md", "llm_calls.jsonl",
    ]:
        _check(os.path.exists(artifact), f"exists: {artifact}", f"MISSING: {artifact}")

    # 2. JSON validity
    print("\n[2] JSON validity")
    for jf in ["qa_framework.json", "qa_scores.json", "compliance_flags.json"]:
        if os.path.exists(jf):
            try:
                json.load(open(jf))
                _ok(f"valid JSON: {jf}")
            except Exception as e:
                _fail(f"invalid JSON: {jf} — {e}")

    # 3. All transcripts parsed
    print("\n[3] Transcript parsing")
    transcript_dir = Path("transcripts")
    if transcript_dir.exists():
        txt_files = list(transcript_dir.glob("*.txt"))
        _check(len(txt_files) > 0, f"{len(txt_files)} transcript(s) found", "No transcripts")
        for txt_file in txt_files:
            content = open(txt_file, encoding="utf-8").read()
            m = re.search(r"T-\d+", content)
            if m:
                parsed_path = f"parsed_transcripts/{m.group(0)}.json"
                _check(os.path.exists(parsed_path), f"parsed: {parsed_path}", f"missing: {parsed_path}")

    # 4. Load llm_calls.jsonl
    llm_calls = []
    if os.path.exists("llm_calls.jsonl"):
        for line in open("llm_calls.jsonl"):
            line = line.strip()
            if line:
                try:
                    llm_calls.append(json.loads(line))
                except Exception:
                    _fail(f"Invalid JSON in llm_calls.jsonl: {line[:80]}")

    # 5. Per-transcript Stage 1, 2, 3 records
    print("\n[4] Per-transcript LLM call records")
    qa_scores = json.load(open("qa_scores.json")) if os.path.exists("qa_scores.json") else []
    call_ids = [q["call_id"] for q in qa_scores]
    _check(len(call_ids) > 0, f"{len(call_ids)} transcript(s) scored", "qa_scores.json empty")

    for cid in call_ids:
        _check(any(c["call_id"] == cid and c["stage"] == "QA_SCORING" for c in llm_calls),
               f"Stage 1 QA log: {cid}", f"Missing Stage 1 log: {cid}")
        _check(any(c["call_id"] == cid and c["stage"] == "COMPLIANCE_EXTRACTION" for c in llm_calls),
               f"Stage 2 compliance log: {cid}", f"Missing Stage 2 log: {cid}")
        _check(any(c["call_id"] == cid and c["stage"] == "COACHING_GENERATION" for c in llm_calls),
               f"Stage 3 coaching log: {cid}", f"Missing Stage 3 log: {cid}")

    # 6. Coaching calls: qa_scores_included must be False
    print("\n[5] Coaching calls exclude QA scores")
    for call in [c for c in llm_calls if c["stage"] == "COACHING_GENERATION"]:
        _check(call.get("qa_scores_included") is False,
               f"qa_scores_included=false: {call['call_id']}",
               f"qa_scores_included NOT false: {call['call_id']}")

    # 7. QA dimensions match framework
    print("\n[6] QA dimensions match framework")
    if os.path.exists("qa_framework.json") and qa_scores:
        framework = json.load(open("qa_framework.json"))
        framework_dim_ids = {d["id"] for d in framework["dimensions"]}
        for qa in qa_scores:
            scored_ids = {d["dimension_id"] for d in qa["dimensions"]}
            _check(framework_dim_ids == scored_ids,
                   f"All dims scored: {qa['call_id']}",
                   f"Dim mismatch {qa['call_id']}: {framework_dim_ids} vs {scored_ids}")

    # 8. Auto-fail conditions checked
    print("\n[7] Auto-fail conditions")
    if os.path.exists("qa_framework.json") and qa_scores:
        framework = json.load(open("qa_framework.json"))
        num_conditions = len(framework["auto_fail_conditions"])
        for qa in qa_scores:
            n = len(qa.get("auto_fail_checks", []))
            _check(n == num_conditions,
                   f"All {num_conditions} auto-fails checked: {qa['call_id']}",
                   f"Expected {num_conditions}, got {n} for {qa['call_id']}")

    # 9. Weighted score verification
    print("\n[8] Weighted score calculation")
    if os.path.exists("qa_framework.json") and qa_scores:
        framework = json.load(open("qa_framework.json"))
        weights = {d["id"]: d["weight"] for d in framework["dimensions"]}
        for qa in qa_scores:
            recalc = round(sum(
                (d["score"] / 10.0) * weights.get(d["dimension_id"], 0) * 100
                for d in qa["dimensions"]
            ), 2)
            _check(abs(recalc - qa["weighted_score"]) < 0.02,
                   f"Weighted score correct: {qa['call_id']} ({qa['weighted_score']})",
                   f"Score mismatch {qa['call_id']}: expected {recalc}, got {qa['weighted_score']}")

    # 10. Grade thresholds
    print("\n[9] Grade thresholds")
    for qa in qa_scores:
        s, af = qa["weighted_score"], qa["auto_fail_triggered"]
        expected = "F" if af else ("A" if s >= 85 else "B" if s >= 70 else "C" if s >= 55 else "D" if s >= 40 else "F")
        _check(qa["grade"] == expected,
               f"Grade correct: {qa['call_id']} = {qa['grade']}",
               f"Grade wrong: {qa['call_id']} expected {expected}, got {qa['grade']}")

    # 11. Coaching notes quality
    print("\n[10] Coaching notes")
    if os.path.exists("coaching_notes.md"):
        content = open("coaching_notes.md").read()
        _check('"' in content, "Coaching notes contain quoted text", "Coaching notes lack quotes")
        _check("##" in content, "Coaching notes have sections", "Coaching notes lack structure")
        for cid in call_ids:
            _check(cid in content, f"Coaching note for {cid}", f"Missing note for {cid}")

    # 12. Dashboard completeness
    print("\n[11] Dashboard completeness")
    if os.path.exists("dashboard_summary.md"):
        content = open("dashboard_summary.md").read()
        for qa in qa_scores:
            _check(qa["call_id"] in content, f"Dashboard row: {qa['call_id']}", f"Missing row: {qa['call_id']}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {len(passes)} passed, {len(errors)} failed")
    if errors:
        print("\nFailed checks:")
        for e in errors:
            print(f"  x {e}")
        return False
    print("\nAll validation checks passed!")
    return True


if __name__ == "__main__":
    sys.exit(0 if validate() else 1)
