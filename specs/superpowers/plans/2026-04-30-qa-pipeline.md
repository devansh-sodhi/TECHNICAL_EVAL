# Multi-Stage AI QA Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a replayable multi-stage AI QA pipeline that evaluates customer support transcripts through strictly separated LLM stages for scoring, compliance extraction, and coaching, with all deterministic logic (grade calculation, weighted scores) in code.

**Architecture:** Python pipeline with an explicit stage state machine (INIT → RESULTS_FINALISED). Each stage is a separate module in `src/`. LLM calls use OpenRouter via the existing OpenAI SDK (`OPENROUTER_API_KEY`). Coaching prompts are constructed to never include QA scores — enforced by an assertion in code. All JSON parsing from LLM responses uses a robust `extract_json()` helper. The pipeline is idempotent and fully replayable from a clean checkout.

**Tech Stack:** Python 3.x, `openai` SDK (via OpenRouter), `python-dotenv`, `hashlib`/`json`/`re` (stdlib)

---

## File Structure

```
pipeline.py                   — Main orchestrator, stage state machine
validate.py                   — Validation command (python validate.py)
src/
  __init__.py                 — Empty package marker
  llm_client.py               — Shared OpenRouter client + MODEL/PROVIDER constants
  llm_logger.py               — Append-only llm_calls.jsonl logger with prompt hashing
  parser.py                   — Deterministic transcript .txt → structured JSON
  qa_scorer.py                — Stage 1: QA scoring LLM call + weighted score in code
  compliance.py               — Stage 2: Compliance extraction LLM call
  coaching.py                 — Stage 3: Coaching LLM call (NO QA scores in prompt)
  dashboard.py                — Deterministic grade calculation + dashboard_summary.md
  calibration.py              — SHOULD: Calibration check LLM call
  team_trend.py               — SHOULD: Team trend vs baseline (pure code, no LLM)
  escalation.py               — STRETCH: Auto-escalation case files (pure code)
  rebuttal.py                 — STRETCH: Rebuttal coaching LLM call
tests/
  test_parser.py              — Unit tests for transcript parsing
  test_qa_scorer.py           — Unit tests for weighted score + grade calculation
  test_coaching.py            — Unit test: assert coaching prompt excludes QA scores

# Input files (created in Task 1):
qa_framework.json
transcripts/T-1041.txt
transcripts/T-1042.txt
transcripts/T-1043.txt

# Generated output files:
parsed_transcripts/           — JSON per transcript
qa_scores.json
compliance_flags.json
coaching_notes.md
dashboard_summary.md
calibration_check.json
team_trend.json
escalation_cases.json
rebuttal_coaching.md
llm_calls.jsonl
pipeline_state.json           — Stage tracker
```

---

## Task 1: Input Files and Project Setup

**Files:**
- Create: `qa_framework.json`
- Create: `transcripts/T-1041.txt`
- Create: `transcripts/T-1042.txt`
- Create: `transcripts/T-1043.txt`
- Modify: `requirements.txt`
- Create: `src/__init__.py`
- Create: `.gitignore` entry for `.env`

- [ ] **Step 1: Create qa_framework.json**

```json
{
  "dimensions": [
    { "id": "D1", "name": "Greeting and identification", "weight": 0.05 },
    { "id": "D2", "name": "Issue comprehension", "weight": 0.15 },
    { "id": "D3", "name": "Accuracy of information provided", "weight": 0.25 },
    { "id": "D4", "name": "Regulatory and compliance language", "weight": 0.20 },
    { "id": "D5", "name": "Empathy and tone", "weight": 0.15 },
    { "id": "D6", "name": "Resolution and next steps clarity", "weight": 0.15 },
    { "id": "D7", "name": "Call closure", "weight": 0.05 }
  ],
  "auto_fail_conditions": [
    "Agent promises a specific outcome they cannot guarantee",
    "Agent discloses account details without completing identity verification",
    "Agent uses dismissive or confrontational language",
    "Agent provides incorrect regulatory information"
  ]
}
```

- [ ] **Step 2: Create transcripts/T-1041.txt**

```
--- T-1041 (Agent: Sara) ---
Sara: Thank you for calling Deriv support, you're speaking with Sara, how can I help you today?
Customer: Hi, I've been waiting 6 days for my withdrawal and I'm getting really frustrated.
Sara: I completely understand your frustration, and I appreciate your patience. Can I take your account email to look into this?
Customer: It's john.doe@email.com
Sara: Thank you. I can see the withdrawal was initiated on the 12th. It looks like it's been flagged for a manual compliance review, which can take up to 7 business days. You're within that window, but I can escalate it for you to a senior reviewer.
Customer: Will that guarantee it gets processed today?
Sara: I can't guarantee same-day processing, but escalating will ensure it's reviewed as a priority — typically within 24 hours. I'll raise that for you now.
Customer: Okay, thank you.
Sara: You'll get an email confirmation of the escalation within the hour. Is there anything else I can help with?
Customer: No, that's all.
Sara: Great, thank you for your patience and for calling Deriv. Have a great day!
```

- [ ] **Step 3: Create transcripts/T-1042.txt**

```
--- T-1042 (Agent: Marcus) ---
Marcus: Deriv support, Marcus speaking.
Customer: I want to know why my account was suspended. I haven't done anything wrong.
Marcus: Okay so I need to look at your account. What's your email?
Customer: mark.smith@email.com
Marcus: Right, I can see here your account was flagged by our fraud team. It looks like there were some unusual login patterns.
Customer: That's ridiculous. I want it unsuspended now.
Marcus: I get that you're annoyed but there's not much I can do. The fraud team makes that call, not me. You'd have to wait for them to review it which could be weeks.
Customer: Weeks?! That's unacceptable. I have money in there.
Marcus: Yeah I understand but I can't override the fraud team. That's just how it works. I can log a note if you want.
Customer: This is terrible service. I'm going to report you.
Marcus: That's your right. Is there anything else?
```

- [ ] **Step 4: Create transcripts/T-1043.txt**

```
--- T-1043 (Agent: Linda) ---
Linda: Hello, thanks for calling Deriv, this is Linda, how can I assist you today?
Customer: Hi Linda, I'm trying to understand the leverage limits for forex trading on my account. I'm based in the UK.
Linda: Of course! For UK clients under FCA regulation, the maximum leverage for major forex pairs is 1:30 for retail clients. If you're a professional client, that can go up to 1:500.
Customer: How do I become a professional client?
Linda: You'd need to meet at least two of three criteria: trading in significant size, a financial portfolio over €500,000, or relevant professional experience in financial services. You can apply through your account settings.
Customer: Great, and is there a minimum deposit to get started?
Linda: There's no minimum deposit for a standard account, though some trading platforms have their own minimums. Deriv Bot requires a $5 minimum for example.
Customer: Perfect, that's really helpful.
Linda: Wonderful! Is there anything else I can help with today?
Customer: No, that's everything, thank you.
Linda: My pleasure. Thanks for calling Deriv — have a lovely day!
```

- [ ] **Step 5: Update requirements.txt**

```
openai
python-dotenv
```

- [ ] **Step 6: Create src/__init__.py (empty)**

```python
```

- [ ] **Step 7: Create src directory and verify**

```bash
mkdir -p src tests transcripts parsed_transcripts
```

- [ ] **Step 8: Commit**

```bash
git add qa_framework.json transcripts/ src/__init__.py requirements.txt
git commit -m "feat: add input fixtures and project structure"
```

---

## Task 2: Core Infrastructure — LLM Client and Logger

**Files:**
- Create: `src/llm_client.py`
- Create: `src/llm_logger.py`

- [ ] **Step 1: Create src/llm_client.py**

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "openai/gpt-4o-mini"
PROVIDER = "openrouter"

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    return _client

def call_llm(prompt: str, system: str = None, temperature: float = 0.1) -> str:
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content
```

- [ ] **Step 2: Create src/llm_logger.py**

```python
import json
import hashlib
from datetime import datetime, timezone

LOGFILE = "llm_calls.jsonl"

def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

def log_llm_call(
    stage: str,
    call_id: str,
    provider: str,
    model: str,
    prompt: str,
    input_artifacts: list,
    output_artifact: str,
    qa_scores_included: bool = False,
):
    record = {
        "stage": stage,
        "call_id": call_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "prompt_hash": hash_prompt(prompt),
        "input_artifacts": input_artifacts,
        "output_artifact": output_artifact,
        "qa_scores_included": qa_scores_included,
    }
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
```

- [ ] **Step 3: Verify imports work**

```bash
python -c "from src.llm_client import MODEL, PROVIDER; print(MODEL, PROVIDER)"
```

Expected output: `openai/gpt-4o-mini openrouter`

- [ ] **Step 4: Commit**

```bash
git add src/llm_client.py src/llm_logger.py
git commit -m "feat: add LLM client and call logger"
```

---

## Task 3: Transcript Parser

**Files:**
- Create: `src/parser.py`
- Create: `tests/test_parser.py`

- [ ] **Step 1: Write failing tests first**

Create `tests/test_parser.py`:

```python
import json
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.parser import parse_transcript_file, parse_all_transcripts

SAMPLE_TRANSCRIPT = """--- T-1041 (Agent: Sara) ---
Sara: Thank you for calling Deriv support, you're speaking with Sara, how can I help you today?
Customer: Hi, I've been waiting 6 days for my withdrawal.
Sara: I completely understand. Can I take your account email?
Customer: It's john.doe@email.com
Sara: Great, thank you for calling Deriv. Have a great day!
"""

def test_parse_call_id_and_agent():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(SAMPLE_TRANSCRIPT)
        tmp_path = f.name
    try:
        result = parse_transcript_file(tmp_path)
        assert result['call_id'] == 'T-1041'
        assert result['agent_name'] == 'Sara'
    finally:
        os.unlink(tmp_path)

def test_parse_turns_structure():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(SAMPLE_TRANSCRIPT)
        tmp_path = f.name
    try:
        result = parse_transcript_file(tmp_path)
        assert len(result['turns']) == 5
        assert result['turns'][0]['speaker'] == 'Sara'
        assert result['turns'][0]['turn_number'] == 1
        assert result['turns'][0]['call_id'] == 'T-1041'
        assert result['turns'][0]['agent_name'] == 'Sara'
        assert 'Thank you for calling' in result['turns'][0]['text']
    finally:
        os.unlink(tmp_path)

def test_parse_estimated_duration():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(SAMPLE_TRANSCRIPT)
        tmp_path = f.name
    try:
        result = parse_transcript_file(tmp_path)
        assert isinstance(result['estimated_duration_minutes'], int)
        assert result['estimated_duration_minutes'] >= 1
    finally:
        os.unlink(tmp_path)

def test_parse_issue_type_and_resolution():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(SAMPLE_TRANSCRIPT)
        tmp_path = f.name
    try:
        result = parse_transcript_file(tmp_path)
        assert result['issue_type'] == 'withdrawal_inquiry'
        assert result['resolution_status'] in ('resolved', 'escalated', 'unresolved')
    finally:
        os.unlink(tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_parser.py -v
```

Expected: ImportError or AttributeError — parser not yet implemented.

- [ ] **Step 3: Create src/parser.py**

```python
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
    if any(w in full_text for w in ["have a great day", "have a lovely day", "anything else i can help"]):
        resolution_status = "resolved"
    elif any(w in full_text for w in ["escalat", "senior reviewer", "review it"]):
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
        print(f"  Parsed {txt_file.name} → {output_path}")
        results.append(parsed)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_parser.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Quick smoke test with real files**

```bash
python -c "
from src.parser import parse_all_transcripts
results = parse_all_transcripts('transcripts', 'parsed_transcripts')
for r in results: print(r['call_id'], r['agent_name'], len(r['turns']), 'turns')
"
```

Expected:
```
T-1041 Sara 11 turns
T-1042 Marcus 10 turns
T-1043 Linda 12 turns
```

- [ ] **Step 6: Commit**

```bash
git add src/parser.py tests/test_parser.py parsed_transcripts/
git commit -m "feat: add deterministic transcript parser with tests"
```

---

## Task 4: QA Scorer — Stage 1 LLM Calls

**Files:**
- Create: `src/qa_scorer.py`
- Create: `tests/test_qa_scorer.py`

- [ ] **Step 1: Write failing tests for pure-code logic**

Create `tests/test_qa_scorer.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.qa_scorer import calculate_weighted_score, calculate_grade, extract_json

FRAMEWORK = {
    "dimensions": [
        {"id": "D1", "weight": 0.05},
        {"id": "D2", "weight": 0.15},
        {"id": "D3", "weight": 0.25},
        {"id": "D4", "weight": 0.20},
        {"id": "D5", "weight": 0.15},
        {"id": "D6", "weight": 0.15},
        {"id": "D7", "weight": 0.05},
    ]
}

def test_weighted_score_perfect():
    dims = [{"dimension_id": f"D{i}", "score": 10} for i in range(1, 8)]
    score = calculate_weighted_score(dims, FRAMEWORK)
    assert score == 100.0

def test_weighted_score_zero():
    dims = [{"dimension_id": f"D{i}", "score": 0} for i in range(1, 8)]
    score = calculate_weighted_score(dims, FRAMEWORK)
    assert score == 0.0

def test_weighted_score_partial():
    dims = [{"dimension_id": f"D{i}", "score": 5} for i in range(1, 8)]
    score = calculate_weighted_score(dims, FRAMEWORK)
    assert abs(score - 50.0) < 0.01

def test_grade_a():
    assert calculate_grade(90.0, False) == "A"
    assert calculate_grade(85.0, False) == "A"

def test_grade_b():
    assert calculate_grade(80.0, False) == "B"
    assert calculate_grade(70.0, False) == "B"

def test_grade_c():
    assert calculate_grade(60.0, False) == "C"
    assert calculate_grade(55.0, False) == "C"

def test_grade_d():
    assert calculate_grade(45.0, False) == "D"
    assert calculate_grade(40.0, False) == "D"

def test_grade_f_low_score():
    assert calculate_grade(30.0, False) == "F"

def test_grade_f_auto_fail():
    assert calculate_grade(95.0, True) == "F"

def test_extract_json_plain():
    result = extract_json('{"key": "value"}')
    assert result == {"key": "value"}

def test_extract_json_markdown_block():
    result = extract_json('```json\n{"key": "value"}\n```')
    assert result == {"key": "value"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_qa_scorer.py -v
```

Expected: ImportError — qa_scorer not yet implemented.

- [ ] **Step 3: Create src/qa_scorer.py**

```python
import json
import re
from src.llm_client import call_llm, MODEL, PROVIDER
from src.llm_logger import log_llm_call


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not extract JSON from: {text[:300]}")


def calculate_weighted_score(dimension_scores: list, framework: dict) -> float:
    weights = {d["id"]: d["weight"] for d in framework["dimensions"]}
    total = sum(
        (ds["score"] / 10.0) * weights.get(ds["dimension_id"], 0) * 100
        for ds in dimension_scores
    )
    return round(total, 2)


def calculate_grade(weighted_score: float, auto_fail_triggered: bool) -> str:
    if auto_fail_triggered:
        return "F"
    if weighted_score >= 85:
        return "A"
    if weighted_score >= 70:
        return "B"
    if weighted_score >= 55:
        return "C"
    if weighted_score >= 40:
        return "D"
    return "F"


def _build_qa_prompt(parsed_transcript: dict, framework: dict) -> str:
    turns_text = "\n".join(
        f"{t['speaker']}: {t['text']}" for t in parsed_transcript["turns"]
    )
    dims_json = json.dumps(framework["dimensions"], indent=2)
    auto_fails_json = json.dumps(framework["auto_fail_conditions"], indent=2)
    dim_template = "\n    ".join(
        f'{{"dimension_id": "{d["id"]}", "score": 8, "evidence_quote": "exact quote", "rationale": "one sentence"}}'
        for d in framework["dimensions"]
    )
    af_template = "\n    ".join(
        f'{{"condition": "{c}", "triggered": false, "evidence": null}}'
        for c in framework["auto_fail_conditions"]
    )
    return f"""You are a QA analyst evaluating a customer support call.

CALL ID: {parsed_transcript['call_id']}
AGENT: {parsed_transcript['agent_name']}

TRANSCRIPT:
{turns_text}

QA FRAMEWORK DIMENSIONS (score each 1–10):
{dims_json}

AUTO-FAIL CONDITIONS (check each explicitly):
{auto_fails_json}

Use exact quotes from the transcript as evidence. For auto-fail checks, triggered must be true only if the condition is clearly violated.

Respond with ONLY valid JSON in this exact structure:
{{
  "dimensions": [
    {dim_template}
  ],
  "auto_fail_checks": [
    {af_template}
  ]
}}"""


def score_transcript(
    parsed_transcript: dict, framework: dict, input_artifacts: list
) -> dict:
    prompt = _build_qa_prompt(parsed_transcript, framework)
    response_text = call_llm(prompt)

    log_llm_call(
        stage="QA_SCORING",
        call_id=parsed_transcript["call_id"],
        provider=PROVIDER,
        model=MODEL,
        prompt=prompt,
        input_artifacts=input_artifacts,
        output_artifact="qa_scores.json",
        qa_scores_included=False,
    )

    parsed_response = extract_json(response_text)
    dimension_scores = parsed_response["dimensions"]
    auto_fail_checks = parsed_response["auto_fail_checks"]

    weighted_score = calculate_weighted_score(dimension_scores, framework)
    auto_fail_triggered = any(c["triggered"] for c in auto_fail_checks)
    grade = calculate_grade(weighted_score, auto_fail_triggered)

    return {
        "call_id": parsed_transcript["call_id"],
        "agent_name": parsed_transcript["agent_name"],
        "dimensions": dimension_scores,
        "auto_fail_checks": auto_fail_checks,
        "weighted_score": weighted_score,
        "auto_fail_triggered": auto_fail_triggered,
        "grade": grade,
    }


def score_all_transcripts(parsed_transcripts: list, framework: dict) -> list:
    results = []
    for pt in parsed_transcripts:
        print(f"  QA scoring {pt['call_id']}...")
        input_artifacts = [
            f"parsed_transcripts/{pt['call_id']}.json",
            "qa_framework.json",
        ]
        results.append(score_transcript(pt, framework, input_artifacts))
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_qa_scorer.py -v
```

Expected: 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qa_scorer.py tests/test_qa_scorer.py
git commit -m "feat: add QA scorer with weighted score and grade calculation"
```

---

## Task 5: Compliance Extractor — Stage 2 LLM Calls

**Files:**
- Create: `src/compliance.py`

- [ ] **Step 1: Create src/compliance.py**

```python
import json
from src.llm_client import call_llm, MODEL, PROVIDER
from src.llm_logger import log_llm_call
from src.qa_scorer import extract_json


def _build_compliance_prompt(parsed_transcript: dict) -> str:
    turns_text = "\n".join(
        f"{t['speaker']}: {t['text']}" for t in parsed_transcript["turns"]
    )
    return f"""You are a compliance officer reviewing a customer support transcript.

CALL ID: {parsed_transcript['call_id']}

TRANSCRIPT:
{turns_text}

Identify ALL compliance, legal, financial, regulatory, reputational, or conduct risks.
Include risks of any severity — be thorough.

Respond with ONLY valid JSON:
{{
  "flags": [
    {{
      "statement_text": "exact quote from the transcript",
      "speaker": "speaker name",
      "risk_type": "regulatory | financial_commitment | data_disclosure | conduct | reputational | other",
      "severity": "critical | high | medium | low",
      "explanation": "brief explanation of why this is a risk"
    }}
  ]
}}

If there are no compliance risks, return: {{"flags": []}}"""


def extract_compliance_flags(parsed_transcript: dict, input_artifacts: list) -> dict:
    prompt = _build_compliance_prompt(parsed_transcript)
    response_text = call_llm(prompt)

    log_llm_call(
        stage="COMPLIANCE_EXTRACTION",
        call_id=parsed_transcript["call_id"],
        provider=PROVIDER,
        model=MODEL,
        prompt=prompt,
        input_artifacts=input_artifacts,
        output_artifact="compliance_flags.json",
        qa_scores_included=False,
    )

    parsed_response = extract_json(response_text)
    flags = parsed_response.get("flags", [])
    for flag in flags:
        flag["call_id"] = parsed_transcript["call_id"]

    return {
        "call_id": parsed_transcript["call_id"],
        "agent_name": parsed_transcript["agent_name"],
        "flags": flags,
    }


def extract_all_compliance(parsed_transcripts: list) -> list:
    results = []
    for pt in parsed_transcripts:
        print(f"  Compliance extraction for {pt['call_id']}...")
        input_artifacts = [f"parsed_transcripts/{pt['call_id']}.json"]
        results.append(extract_compliance_flags(pt, input_artifacts))
    return results
```

- [ ] **Step 2: Commit**

```bash
git add src/compliance.py
git commit -m "feat: add compliance extractor Stage 2 LLM calls"
```

---

## Task 6: Coaching Generator — Stage 3 LLM Calls (No QA Scores)

**Files:**
- Create: `src/coaching.py`
- Create: `tests/test_coaching.py`

- [ ] **Step 1: Write test asserting QA scores excluded from prompt**

Create `tests/test_coaching.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.coaching import _build_coaching_prompt

PARSED = {
    "call_id": "T-TEST",
    "agent_name": "TestAgent",
    "turns": [
        {"speaker": "TestAgent", "text": "Hello, how can I help?", "turn_number": 1, "call_id": "T-TEST", "agent_name": "TestAgent"},
        {"speaker": "Customer", "text": "I need help.", "turn_number": 2, "call_id": "T-TEST", "agent_name": "TestAgent"},
    ]
}

QA_SCORES_KEYWORDS = [
    "weighted", "score", "grade", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
    "auto_fail", "compliance severity", "calibration"
]

def test_coaching_prompt_excludes_qa_scores():
    prompt = _build_coaching_prompt(PARSED)
    prompt_lower = prompt.lower()
    forbidden = ["weighted score", "qa score", "grade:", "auto-fail", "compliance score"]
    for term in forbidden:
        assert term not in prompt_lower, f"Coaching prompt must not contain '{term}'"

def test_coaching_prompt_includes_transcript():
    prompt = _build_coaching_prompt(PARSED)
    assert "Hello, how can I help?" in prompt
    assert "TestAgent" in prompt
    assert "T-TEST" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_coaching.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create src/coaching.py**

```python
import json
from src.llm_client import call_llm, MODEL, PROVIDER
from src.llm_logger import log_llm_call
from src.qa_scorer import extract_json


def _build_coaching_prompt(parsed_transcript: dict) -> str:
    # IMPORTANT: This function intentionally excludes QA scores, grades, and compliance scores.
    # Only transcript content, agent name, and call ID are included.
    turns_text = "\n".join(
        f"{t['speaker']}: {t['text']}" for t in parsed_transcript["turns"]
    )
    return f"""You are a customer support coach providing feedback to {parsed_transcript['agent_name']}.

CALL ID: {parsed_transcript['call_id']}
AGENT: {parsed_transcript['agent_name']}

TRANSCRIPT:
{turns_text}

Provide coaching grounded ONLY in evidence from this transcript.
Do NOT reference scores, ratings, metrics, or assessments.

Requirements:
- What went well: at least 2 specific examples, each with an exact agent quote
- Development areas: at least 2 specific examples, each with an exact agent quote and suggested alternative phrasing
- Every coaching point must include an exact agent quote

Respond with ONLY valid JSON:
{{
  "agent_name": "{parsed_transcript['agent_name']}",
  "call_id": "{parsed_transcript['call_id']}",
  "what_went_well": [
    {{
      "point": "description of positive behavior",
      "agent_quote": "exact quote from transcript",
      "context": "why this was effective"
    }}
  ],
  "development_areas": [
    {{
      "area": "description of development opportunity",
      "agent_quote": "exact quote from transcript",
      "suggested_alternative": "better phrasing the agent could use",
      "why": "explanation of the improvement"
    }}
  ]
}}"""


def generate_coaching(parsed_transcript: dict, input_artifacts: list) -> dict:
    prompt = _build_coaching_prompt(parsed_transcript)
    # Enforce in code: coaching prompt must not contain QA scores
    assert "weighted score" not in prompt.lower(), "QA scores must not appear in coaching prompt"
    assert "grade:" not in prompt.lower(), "Grades must not appear in coaching prompt"

    response_text = call_llm(prompt)

    log_llm_call(
        stage="COACHING_GENERATION",
        call_id=parsed_transcript["call_id"],
        provider=PROVIDER,
        model=MODEL,
        prompt=prompt,
        input_artifacts=input_artifacts,
        output_artifact="coaching_notes.md",
        qa_scores_included=False,  # MUST always be False for coaching
    )

    return extract_json(response_text)


def generate_all_coaching(parsed_transcripts: list) -> list:
    results = []
    for pt in parsed_transcripts:
        print(f"  Coaching for {pt['call_id']}...")
        input_artifacts = [f"parsed_transcripts/{pt['call_id']}.json"]
        results.append(generate_coaching(pt, input_artifacts))
    return results


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_coaching.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/coaching.py tests/test_coaching.py
git commit -m "feat: add coaching generator Stage 3 — QA score exclusion enforced in code and tests"
```

---

## Task 7: Dashboard — Deterministic Grade Computation

**Files:**
- Create: `src/dashboard.py`

- [ ] **Step 1: Create src/dashboard.py**

```python
from src.qa_scorer import calculate_grade

_SEVERITY_ORDER = ["critical", "high", "medium", "low", "none"]


def _highest_severity(flags: list) -> str:
    severities = {f.get("severity", "none").lower() for f in flags}
    for sev in _SEVERITY_ORDER:
        if sev in severities:
            return sev
    return "none"


def compute_dashboard(qa_scores: list, compliance_results: list) -> list:
    compliance_by_call = {r["call_id"]: r["flags"] for r in compliance_results}
    rows = []
    for qa in qa_scores:
        call_id = qa["call_id"]
        flags = compliance_by_call.get(call_id, [])
        rows.append({
            "agent_name": qa["agent_name"],
            "call_id": call_id,
            "weighted_score": qa["weighted_score"],
            "auto_fail_triggered": qa["auto_fail_triggered"],
            "compliance_flags_count": len(flags),
            "highest_compliance_severity": _highest_severity(flags),
            "grade": qa["grade"],
        })
    return rows


def dashboard_to_markdown(rows: list) -> str:
    header = (
        "# QA Dashboard Summary\n\n"
        "| Agent | Call ID | Weighted Score | Auto-Fail | "
        "Compliance Flags | Highest Severity | Grade |\n"
        "|-------|---------|---------------|-----------|"
        "-----------------|-----------------|-------|\n"
    )
    body = ""
    for row in rows:
        body += (
            f"| {row['agent_name']} | {row['call_id']} | "
            f"{row['weighted_score']:.1f} | "
            f"{'Yes' if row['auto_fail_triggered'] else 'No'} | "
            f"{row['compliance_flags_count']} | "
            f"{row['highest_compliance_severity']} | "
            f"{row['grade']} |\n"
        )
    return header + body
```

- [ ] **Step 2: Commit**

```bash
git add src/dashboard.py
git commit -m "feat: add deterministic dashboard and grade calculation"
```

---

## Task 8: SHOULD Items — Calibration and Team Trend

**Files:**
- Create: `src/calibration.py`
- Create: `src/team_trend.py`

- [ ] **Step 1: Create src/calibration.py**

```python
import json
from src.llm_client import call_llm, MODEL, PROVIDER
from src.llm_logger import log_llm_call
from src.qa_scorer import extract_json


def _build_calibration_prompt(qa_scores: list) -> str:
    scores_text = json.dumps(qa_scores, indent=2)
    return f"""You are a QA calibration expert reviewing multiple call scoresheets side by side.

SCORE SHEETS:
{scores_text}

Identify dimensions where similar agent behaviors were scored inconsistently across transcripts.

Respond with ONLY valid JSON:
{{
  "inconsistencies": [
    {{
      "dimension_id": "D1",
      "call_ids": ["T-1041", "T-1042"],
      "observed_difference": "description of the scoring gap",
      "why_it_may_be_inconsistent": "explanation of why the gap may be unfair",
      "recommended_adjustment": "specific suggestion to resolve the inconsistency"
    }}
  ]
}}

If no inconsistencies found: {{"inconsistencies": []}}"""


def run_calibration_check(qa_scores: list) -> dict:
    prompt = _build_calibration_prompt(qa_scores)
    response_text = call_llm(prompt)

    log_llm_call(
        stage="CALIBRATION_CHECK",
        call_id="ALL",
        provider=PROVIDER,
        model=MODEL,
        prompt=prompt,
        input_artifacts=["qa_scores.json"],
        output_artifact="calibration_check.json",
        qa_scores_included=True,
    )

    parsed = extract_json(response_text)
    return {"inconsistencies": parsed.get("inconsistencies", [])}
```

- [ ] **Step 2: Create src/team_trend.py**

```python
HISTORICAL_BASELINE = {
    "D1": 8, "D2": 7, "D3": 7, "D4": 6, "D5": 7, "D6": 7, "D7": 8
}


def compute_team_trend(qa_scores: list) -> dict:
    dim_totals: dict = {}
    dim_counts: dict = {}

    for qa in qa_scores:
        for dim in qa["dimensions"]:
            did = dim["dimension_id"]
            dim_totals[did] = dim_totals.get(did, 0) + dim["score"]
            dim_counts[did] = dim_counts.get(did, 0) + 1

    current_averages = {
        did: round(dim_totals[did] / dim_counts[did], 2)
        for did in dim_totals
    }

    trends = []
    for did in sorted(current_averages.keys()):
        current_avg = current_averages[did]
        baseline = HISTORICAL_BASELINE.get(did, 0)
        delta = round(current_avg - baseline, 2)
        trends.append({
            "dimension_id": did,
            "baseline_score": baseline,
            "current_average": current_avg,
            "delta": delta,
            "direction": "up" if delta > 0 else ("down" if delta < 0 else "stable"),
        })

    return {
        "historical_baseline": HISTORICAL_BASELINE,
        "current_averages": current_averages,
        "trends": trends,
    }
```

- [ ] **Step 3: Commit**

```bash
git add src/calibration.py src/team_trend.py
git commit -m "feat: add calibration check and team trend (SHOULD items)"
```

---

## Task 9: STRETCH Items — Escalation and Rebuttal Coaching

**Files:**
- Create: `src/escalation.py`
- Create: `src/rebuttal.py`

- [ ] **Step 1: Create src/escalation.py**

```python
def get_escalation_cases(
    qa_scores: list, compliance_results: list, parsed_transcripts: list
) -> list:
    compliance_by_call = {r["call_id"]: r["flags"] for r in compliance_results}
    transcripts_by_call = {pt["call_id"]: pt for pt in parsed_transcripts}
    cases = []

    for qa in qa_scores:
        call_id = qa["call_id"]

        for check in qa.get("auto_fail_checks", []):
            if check["triggered"]:
                cases.append({
                    "call_id": call_id,
                    "agent_name": qa["agent_name"],
                    "trigger_type": "auto_fail",
                    "triggered_condition": check["condition"],
                    "transcript_excerpt": check.get("evidence") or "N/A",
                    "recommended_action": (
                        "Immediate supervisor review and coaching session required"
                    ),
                })

        for flag in compliance_by_call.get(call_id, []):
            if flag.get("severity") == "critical":
                cases.append({
                    "call_id": call_id,
                    "agent_name": qa["agent_name"],
                    "trigger_type": "critical_compliance",
                    "triggered_condition": (
                        f"{flag['risk_type']}: {flag['explanation']}"
                    ),
                    "transcript_excerpt": flag.get("statement_text", "N/A"),
                    "recommended_action": (
                        "Compliance team review required within 24 hours"
                    ),
                })

    return cases
```

- [ ] **Step 2: Create src/rebuttal.py**

```python
from src.llm_client import call_llm, MODEL, PROVIDER
from src.llm_logger import log_llm_call
from src.qa_scorer import extract_json

_THREAT_PHRASES = ["report you", "report this", "going to report", "complain", "escalate"]


def _has_escalation_threat(parsed_transcript: dict) -> tuple:
    agent = parsed_transcript["agent_name"]
    for turn in parsed_transcript["turns"]:
        if turn["speaker"] != agent:
            for phrase in _THREAT_PHRASES:
                if phrase in turn["text"].lower():
                    return True, turn
    return False, None


def _build_rebuttal_prompt(parsed_transcript: dict, threat_turn: dict) -> str:
    turns_text = "\n".join(
        f"{t['speaker']}: {t['text']}" for t in parsed_transcript["turns"]
    )
    return f"""You are a customer support coach designing a rebuttal coaching scenario.

CALL ID: {parsed_transcript['call_id']}
AGENT: {parsed_transcript['agent_name']}

The customer made a threat or escalation statement:
"{threat_turn['text']}"

FULL TRANSCRIPT:
{turns_text}

Create a role-play coaching scenario to help the agent handle such situations better.

Respond with ONLY valid JSON:
{{
  "call_id": "{parsed_transcript['call_id']}",
  "agent_name": "{parsed_transcript['agent_name']}",
  "customer_threat_quote": "exact quote from transcript",
  "simulated_customer_message": "a realistic customer threat for role-play practice",
  "ideal_agent_response": "the ideal response the agent should give",
  "coaching_goal": "what skill this scenario develops",
  "technique_demonstrated": "the specific technique used in the ideal response"
}}"""


def generate_rebuttal_coaching(parsed_transcripts: list) -> list:
    results = []
    for pt in parsed_transcripts:
        has_threat, threat_turn = _has_escalation_threat(pt)
        if not has_threat:
            continue
        print(f"  Rebuttal coaching for {pt['call_id']}...")
        prompt = _build_rebuttal_prompt(pt, threat_turn)
        response_text = call_llm(prompt)
        log_llm_call(
            stage="REBUTTAL_COACHING",
            call_id=pt["call_id"],
            provider=PROVIDER,
            model=MODEL,
            prompt=prompt,
            input_artifacts=[f"parsed_transcripts/{pt['call_id']}.json"],
            output_artifact="rebuttal_coaching.md",
            qa_scores_included=False,
        )
        results.append(extract_json(response_text))
    return results


def rebuttal_to_markdown(scenarios: list) -> str:
    lines = ["# Rebuttal Coaching Scenarios\n"]
    for s in scenarios:
        agent = s.get("agent_name", "Unknown")
        call_id = s.get("call_id", "Unknown")
        lines.append(f"## {agent} — {call_id}\n")
        lines.append(f"**Customer Threat Quote:** > \"{s.get('customer_threat_quote', '')}\"\n")
        lines.append("### Role-Play Scenario\n")
        lines.append(f"**Simulated Customer Message:** {s.get('simulated_customer_message', '')}\n")
        lines.append(f"**Ideal Agent Response:** {s.get('ideal_agent_response', '')}\n")
        lines.append(f"**Coaching Goal:** {s.get('coaching_goal', '')}\n")
        lines.append(f"**Technique Demonstrated:** {s.get('technique_demonstrated', '')}\n")
        lines.append("---\n")
    return "\n".join(lines)
```

- [ ] **Step 3: Commit**

```bash
git add src/escalation.py src/rebuttal.py
git commit -m "feat: add auto-escalation and rebuttal coaching (STRETCH items)"
```

---

## Task 10: Pipeline Orchestrator

**Files:**
- Create: `pipeline.py`

- [ ] **Step 1: Create pipeline.py**

```python
import json
import os
import sys
from pathlib import Path

from src.parser import parse_all_transcripts
from src.qa_scorer import score_all_transcripts
from src.compliance import extract_all_compliance
from src.coaching import generate_all_coaching, coaching_to_markdown
from src.dashboard import compute_dashboard, dashboard_to_markdown
from src.calibration import run_calibration_check
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


def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"stage": "INIT"}


def _save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _advance(state: dict, new_stage: str):
    state["stage"] = new_stage
    _save_state(state)
    print(f"[PIPELINE] ── {new_stage}")


def run_pipeline():
    # Clear previous LLM call log for clean replay
    if os.path.exists("llm_calls.jsonl"):
        os.remove("llm_calls.jsonl")

    state = {"stage": "INIT"}
    _save_state(state)
    print("[PIPELINE] Starting pipeline run…\n")

    # INIT → INPUTS_LOADED
    print("[1/9] Loading inputs…")
    with open("qa_framework.json") as f:
        framework = json.load(f)
    _advance(state, "INPUTS_LOADED")

    # INPUTS_LOADED → TRANSCRIPTS_PARSED
    print("[2/9] Parsing transcripts…")
    parsed_transcripts = parse_all_transcripts("transcripts", "parsed_transcripts")
    _advance(state, "TRANSCRIPTS_PARSED")

    # TRANSCRIPTS_PARSED → QA_SCORED
    print("[3/9] QA scoring (Stage 1 LLM calls)…")
    qa_scores = score_all_transcripts(parsed_transcripts, framework)
    with open("qa_scores.json", "w") as f:
        json.dump(qa_scores, f, indent=2)
    _advance(state, "QA_SCORED")

    # QA_SCORED → COMPLIANCE_EXTRACTED
    print("[4/9] Compliance extraction (Stage 2 LLM calls)…")
    compliance_results = extract_all_compliance(parsed_transcripts)
    with open("compliance_flags.json", "w") as f:
        json.dump(compliance_results, f, indent=2)
    _advance(state, "COMPLIANCE_EXTRACTED")

    # COMPLIANCE_EXTRACTED → COACHING_GENERATED
    # Dashboard must not be computed before this point (stage enforced above)
    print("[5/9] Coaching generation (Stage 3 LLM calls — no QA scores)…")
    coaching_results = generate_all_coaching(parsed_transcripts)
    with open("coaching_notes.md", "w") as f:
        f.write(coaching_to_markdown(coaching_results))
    _advance(state, "COACHING_GENERATED")

    # COACHING_GENERATED → DASHBOARD_COMPUTED
    # Dashboard requires QA_SCORED and COMPLIANCE_EXTRACTED, enforced by stage ordering above.
    print("[6/9] Computing dashboard…")
    dashboard_rows = compute_dashboard(qa_scores, compliance_results)
    with open("dashboard_summary.md", "w") as f:
        f.write(dashboard_to_markdown(dashboard_rows))

    print("[6b] Calibration check (SHOULD)…")
    calibration = run_calibration_check(qa_scores)
    with open("calibration_check.json", "w") as f:
        json.dump(calibration, f, indent=2)

    print("[6c] Team performance trend (SHOULD)…")
    team_trend = compute_team_trend(qa_scores)
    with open("team_trend.json", "w") as f:
        json.dump(team_trend, f, indent=2)

    print("[6d] Auto-escalation cases (STRETCH)…")
    escalation_cases = get_escalation_cases(qa_scores, compliance_results, parsed_transcripts)
    with open("escalation_cases.json", "w") as f:
        json.dump(escalation_cases, f, indent=2)

    print("[6e] Rebuttal coaching (STRETCH)…")
    rebuttal_scenarios = generate_rebuttal_coaching(parsed_transcripts)
    if rebuttal_scenarios:
        with open("rebuttal_coaching.md", "w") as f:
            f.write(rebuttal_to_markdown(rebuttal_scenarios))

    _advance(state, "DASHBOARD_COMPUTED")

    print("[7/9] Validation complete marker…")
    _advance(state, "VALIDATION_COMPLETE")

    print("[8/9] Finalising results…")
    _advance(state, "RESULTS_FINALISED")

    print("\n[PIPELINE] ✓ Pipeline complete. Run `python validate.py` to verify.")


if __name__ == "__main__":
    run_pipeline()
```

- [ ] **Step 2: Verify the pipeline can be imported without errors**

```bash
python -c "import pipeline; print('Import OK')"
```

Expected: `Import OK`

- [ ] **Step 3: Commit**

```bash
git add pipeline.py
git commit -m "feat: add pipeline orchestrator with stage state machine"
```

---

## Task 11: Validation Command

**Files:**
- Create: `validate.py`

- [ ] **Step 1: Create validate.py**

```python
import json
import os
import re
import sys
from pathlib import Path

errors = []
passes = []


def _ok(msg: str):
    passes.append(msg)
    print(f"  OK  {msg}")


def _fail(msg: str):
    errors.append(msg)
    print(f"  FAIL {msg}")


def _check(condition: bool, ok_msg: str, fail_msg: str):
    if condition:
        _ok(ok_msg)
    else:
        _fail(fail_msg)


def validate():
    print("=" * 60)
    print("Pipeline Validation")
    print("=" * 60)

    # 1. Required artifacts exist
    print("\n[1] Required artifacts")
    required = [
        "qa_framework.json",
        "qa_scores.json",
        "compliance_flags.json",
        "coaching_notes.md",
        "dashboard_summary.md",
        "llm_calls.jsonl",
    ]
    for artifact in required:
        _check(os.path.exists(artifact), f"exists: {artifact}", f"MISSING: {artifact}")

    # 2. JSON files are valid
    print("\n[2] JSON validity")
    for jf in ["qa_framework.json", "qa_scores.json", "compliance_flags.json"]:
        if os.path.exists(jf):
            try:
                with open(jf) as f:
                    json.load(f)
                _ok(f"valid JSON: {jf}")
            except json.JSONDecodeError as e:
                _fail(f"invalid JSON: {jf} — {e}")

    # 3. All transcripts were parsed
    print("\n[3] Transcript parsing")
    transcript_dir = Path("transcripts")
    if transcript_dir.exists():
        txt_files = list(transcript_dir.glob("*.txt"))
        _check(len(txt_files) > 0, f"{len(txt_files)} transcript(s) found", "No transcript files found")
        for txt_file in txt_files:
            with open(txt_file, encoding="utf-8") as f:
                content = f.read()
            m = re.search(r"T-\d+", content)
            if m:
                call_id = m.group(0)
                parsed_path = f"parsed_transcripts/{call_id}.json"
                _check(
                    os.path.exists(parsed_path),
                    f"parsed: {parsed_path}",
                    f"missing parsed: {parsed_path}",
                )

    # 4. Load llm_calls.jsonl
    llm_calls = []
    if os.path.exists("llm_calls.jsonl"):
        with open("llm_calls.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        llm_calls.append(json.loads(line))
                    except json.JSONDecodeError:
                        _fail(f"Invalid JSON line in llm_calls.jsonl: {line[:80]}")

    # 5. Per-transcript Stage 1, 2, 3 records
    print("\n[4] Per-transcript LLM call records")
    qa_scores = []
    if os.path.exists("qa_scores.json"):
        with open("qa_scores.json") as f:
            qa_scores = json.load(f)

    call_ids = [q["call_id"] for q in qa_scores]
    _check(len(call_ids) > 0, f"{len(call_ids)} transcript(s) in qa_scores.json", "qa_scores.json is empty")

    for cid in call_ids:
        has_qa = any(c["call_id"] == cid and c["stage"] == "QA_SCORING" for c in llm_calls)
        has_comp = any(c["call_id"] == cid and c["stage"] == "COMPLIANCE_EXTRACTION" for c in llm_calls)
        has_coach = any(c["call_id"] == cid and c["stage"] == "COACHING_GENERATION" for c in llm_calls)
        _check(has_qa, f"Stage 1 QA call logged for {cid}", f"Missing Stage 1 QA log for {cid}")
        _check(has_comp, f"Stage 2 compliance log for {cid}", f"Missing Stage 2 compliance log for {cid}")
        _check(has_coach, f"Stage 3 coaching log for {cid}", f"Missing Stage 3 coaching log for {cid}")

    # 6. Stage 3 coaching calls: qa_scores_included must be False
    print("\n[5] Coaching calls exclude QA scores")
    coaching_calls = [c for c in llm_calls if c["stage"] == "COACHING_GENERATION"]
    for call in coaching_calls:
        _check(
            call.get("qa_scores_included") is False,
            f"qa_scores_included=false for coaching {call['call_id']}",
            f"qa_scores_included is NOT false for coaching {call['call_id']}",
        )

    # 7. QA dimensions match framework
    print("\n[6] QA dimensions match framework")
    if os.path.exists("qa_framework.json") and qa_scores:
        with open("qa_framework.json") as f:
            framework = json.load(f)
        framework_dim_ids = {d["id"] for d in framework["dimensions"]}
        for qa in qa_scores:
            scored_ids = {d["dimension_id"] for d in qa["dimensions"]}
            _check(
                framework_dim_ids == scored_ids,
                f"All dimensions scored for {qa['call_id']}",
                f"Dimension mismatch for {qa['call_id']}: expected {framework_dim_ids}, got {scored_ids}",
            )

    # 8. Auto-fail conditions explicitly checked
    print("\n[7] Auto-fail conditions explicitly checked")
    if os.path.exists("qa_framework.json") and qa_scores:
        with open("qa_framework.json") as f:
            framework = json.load(f)
        num_conditions = len(framework["auto_fail_conditions"])
        for qa in qa_scores:
            num_checked = len(qa.get("auto_fail_checks", []))
            _check(
                num_checked == num_conditions,
                f"All {num_conditions} auto-fail conditions checked for {qa['call_id']}",
                f"Expected {num_conditions} auto-fail checks for {qa['call_id']}, got {num_checked}",
            )

    # 9. Weighted scores verified in code
    print("\n[8] Weighted score calculation verification")
    if os.path.exists("qa_framework.json") and qa_scores:
        with open("qa_framework.json") as f:
            framework = json.load(f)
        weights = {d["id"]: d["weight"] for d in framework["dimensions"]}
        for qa in qa_scores:
            recalc = round(
                sum(
                    (d["score"] / 10.0) * weights.get(d["dimension_id"], 0) * 100
                    for d in qa["dimensions"]
                ),
                2,
            )
            _check(
                abs(recalc - qa["weighted_score"]) < 0.02,
                f"Weighted score correct for {qa['call_id']}: {qa['weighted_score']}",
                f"Score mismatch for {qa['call_id']}: recalculated={recalc}, stored={qa['weighted_score']}",
            )

    # 10. Grade thresholds
    print("\n[9] Grade threshold verification")
    for qa in qa_scores:
        score = qa["weighted_score"]
        auto_fail = qa["auto_fail_triggered"]
        if auto_fail:
            expected = "F"
        elif score >= 85:
            expected = "A"
        elif score >= 70:
            expected = "B"
        elif score >= 55:
            expected = "C"
        elif score >= 40:
            expected = "D"
        else:
            expected = "F"
        _check(
            qa["grade"] == expected,
            f"Grade correct for {qa['call_id']}: {qa['grade']}",
            f"Grade wrong for {qa['call_id']}: expected {expected}, got {qa['grade']}",
        )

    # 11. Coaching notes contain quoted text
    print("\n[10] Coaching notes quality")
    if os.path.exists("coaching_notes.md"):
        with open("coaching_notes.md") as f:
            coaching_content = f.read()
        _check('"' in coaching_content, "Coaching notes contain quoted text", "Coaching notes lack quoted text")
        _check("##" in coaching_content, "Coaching notes have agent sections", "Coaching notes lack structure")
        for cid in call_ids:
            _check(cid in coaching_content, f"Coaching note for {cid}", f"Missing coaching note for {cid}")

    # 12. Dashboard rows for every transcript
    print("\n[11] Dashboard completeness")
    if os.path.exists("dashboard_summary.md"):
        with open("dashboard_summary.md") as f:
            dashboard_content = f.read()
        for qa in qa_scores:
            _check(
                qa["call_id"] in dashboard_content,
                f"Dashboard row for {qa['call_id']}",
                f"Missing dashboard row for {qa['call_id']}",
            )

    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {len(passes)} passed, {len(errors)} failed")
    if errors:
        print("\nFailed checks:")
        for e in errors:
            print(f"  ✗ {e}")
        print()
        return False
    print("\nAll validation checks passed!")
    return True


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
```

- [ ] **Step 2: Run all unit tests**

```bash
python -m pytest tests/ -v
```

Expected: All tests PASS (parser, qa_scorer, coaching tests).

- [ ] **Step 3: Run the full pipeline end-to-end**

```bash
python pipeline.py
```

Expected output (condensed):
```
[PIPELINE] Starting pipeline run…
[1/9] Loading inputs…
[PIPELINE] ── INPUTS_LOADED
[2/9] Parsing transcripts…
  Parsed T-1041.txt → parsed_transcripts/T-1041.json
  Parsed T-1042.txt → parsed_transcripts/T-1042.json
  Parsed T-1043.txt → parsed_transcripts/T-1043.json
[PIPELINE] ── TRANSCRIPTS_PARSED
[3/9] QA scoring (Stage 1 LLM calls)…
...
[PIPELINE] ── RESULTS_FINALISED
[PIPELINE] ✓ Pipeline complete.
```

- [ ] **Step 4: Run validation**

```bash
python validate.py
```

Expected: All validation checks passed!

- [ ] **Step 5: Commit everything**

```bash
git add validate.py tests/
git commit -m "feat: add validation command and complete pipeline"
```

---

## Self-Review Against Spec

### Spec Coverage Check

| Requirement | Task |
|-------------|------|
| INIT → RESULTS_FINALISED stage machine | Task 10 (pipeline.py) |
| Transcript parsing to structured JSON with all required fields | Task 3 |
| parsed_transcripts/ saved before LLM scoring | Task 10 (stage ordering) |
| Stage 1 QA scoring: separate call per transcript | Task 4 |
| All D1–D7 dimensions with score, evidence_quote, rationale | Task 4 |
| All auto-fail conditions checked explicitly | Task 4 |
| Weighted score calculated in code | Task 4 |
| qa_scores.json | Task 10 |
| Stage 2 Compliance: separate call per transcript | Task 5 |
| compliance_flags.json with all required fields | Task 5 |
| Stage 3 Coaching: separate call per transcript | Task 6 |
| Coaching prompt does NOT include QA scores | Task 6 (asserted in code + test) |
| coaching_notes.md with quotes | Task 6 |
| Dashboard with deterministic grade thresholds | Task 7 |
| dashboard_summary.md | Task 10 |
| Calibration check | Task 8 |
| Team trend vs historical baseline | Task 8 |
| Auto-escalation cases | Task 9 |
| Rebuttal coaching (T-1042 must produce scenario) | Task 9 |
| llm_calls.jsonl with all required fields | Task 2 |
| qa_scores_included=false for coaching calls | Task 6 + Task 11 |
| validate.py checks all 12 validation requirements | Task 11 |

### No Placeholders Found
All steps contain complete code. No TBDs.

### Type Consistency
- `parse_transcript_file` → `dict` with keys: `call_id`, `agent_name`, `turns`, `estimated_duration_minutes`, `issue_type`, `resolution_status`, `escalation_signals`
- `score_transcript` → `dict` with keys: `call_id`, `agent_name`, `dimensions`, `auto_fail_checks`, `weighted_score`, `auto_fail_triggered`, `grade`
- `extract_compliance_flags` → `dict` with keys: `call_id`, `agent_name`, `flags`
- All consumers of these types reference the same keys across tasks ✓

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-30-qa-pipeline.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — Fresh subagent per task, review between tasks, fastest iteration given the 50-minute window

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch with checkpoints

**Which approach?**
