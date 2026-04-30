# AI Quality Assurance Pipeline

A replayable, multi-stage pipeline that evaluates customer support call transcripts against a structured QA framework, scores agent performance, extracts compliance risks, and generates personalised coaching notes — all using LLMs as intelligent reviewers, with deterministic code handling every calculation and grade.

---

## What It Does (Plain English)

Customer support teams have QA analysts who listen to calls and score agents. This pipeline replaces that manual review with a structured, auditable AI process:

1. It reads raw call transcripts and a QA scoring framework from disk
2. It runs each transcript through three separate AI reviewers in sequence
3. It calculates scores and grades in code (no LLM involvement)
4. It produces a full audit trail of every AI call made

The key design principle: **the AI reads and interprets, Python calculates and decides.**

---

## Architecture

```
transcripts/*.txt          qa_framework.json
        │                         │
        ▼                         ▼
┌─────────────────────────────────────────┐
│  STAGE 0: Deterministic Parser          │  (no LLM)
│  Regex-based turn extraction            │
│  → parsed_transcripts/{call_id}.json   │
└─────────────────────────────────────────┘
        │
        ▼ (per transcript, sequential)
┌─────────────────────────────────────────┐
│  STAGE 1: QA Evaluator Agent            │  (1 LLM call per transcript)
│  Scores D1–D7, checks auto-fail rules   │
│  → qa_scores.json                       │
└─────────────────────────────────────────┘
        │
        ▼ (per transcript, sequential)
┌─────────────────────────────────────────┐
│  STAGE 2: Compliance Analyst Agent      │  (1 LLM call per transcript)
│  Flags regulatory/conduct/legal risks   │
│  → compliance_flags.json               │
└─────────────────────────────────────────┘
        │
        ▼ (per transcript, sequential, NO QA scores passed in)
┌─────────────────────────────────────────┐
│  STAGE 3: Coach Agent                   │  (1 LLM call per transcript)
│  Writes personalised feedback from      │
│  transcript only — never sees scores    │
│  → coaching_notes.md                   │
└─────────────────────────────────────────┘
        │
        ▼ (deterministic code, no LLM)
┌─────────────────────────────────────────┐
│  STAGE 4: Dashboard + Optional Extras   │
│  Weighted scores, grades, trend,        │
│  escalation, calibration, rebuttal      │
│  → dashboard_summary.md + others       │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  VALIDATION                             │
│  python validate.py                     │
│  Checks every constraint automatically  │
└─────────────────────────────────────────┘
```

---

## Pipeline Stages

The pipeline enforces a strict linear state machine. Each stage must complete before the next begins. The current stage is written to `pipeline_state.json` at every transition.

| # | Stage Name | What Happens |
|---|-----------|--------------|
| 0 | `INIT` | Pipeline starts, previous log cleared |
| 1 | `INPUTS_LOADED` | `qa_framework.json` loaded into memory |
| 2 | `TRANSCRIPTS_PARSED` | All `.txt` files parsed to structured JSON |
| 3 | `QA_SCORED` | One LLM call per transcript for QA scoring |
| 4 | `COMPLIANCE_EXTRACTED` | One LLM call per transcript for compliance risks |
| 5 | `COACHING_GENERATED` | One LLM call per transcript for coaching notes |
| 6 | `DASHBOARD_COMPUTED` | Scores aggregated, grades assigned, extras run |
| 7 | `VALIDATION_COMPLETE` | State saved |
| 8 | `RESULTS_FINALISED` | Pipeline done |

The dashboard cannot run before stages 3 and 4 complete — this is enforced by code order, not configuration.

---

## The Three AI Agents

The pipeline uses the **OpenAI Agents SDK** to define three independent agents. Each is a separate entity with its own instructions, model settings, and output schema. They do not hand off to each other — they are called sequentially by the pipeline orchestrator.

### Agent 1 — QA Evaluator

```python
qa_evaluator = Agent(
    name="QA Evaluator",
    output_type=QAEvaluation,   # Pydantic schema enforced at API level
    model_settings=ModelSettings(temperature=0.1),
)
```

**What it does:** Reads the transcript and the QA framework and scores each dimension (D1–D7) on a 1–10 scale, providing an exact quote from the transcript as evidence and a one-sentence rationale. It also explicitly checks every auto-fail condition.

**What it does NOT do:** Calculate the weighted score or assign a grade. Those happen in Python after the LLM responds.

**Output type:** `QAEvaluation` — a Pydantic model with `dimensions: List[DimensionScore]` and `auto_fail_checks: List[AutoFailCheck]`. The SDK enforces this schema against the LLM's response before it reaches the application.

**Prompt safeguards:**
- Explicitly lists the valid dimension IDs (e.g., `D1, D2, D3, D4, D5, D6, D7`) to prevent the LLM hallucinating extra dimensions
- Provides detailed auto-fail guidance explaining what does and does not constitute a violation (e.g., requesting an email address before discussing account details counts as completed identity verification)

---

### Agent 2 — Compliance Analyst

```python
compliance_analyst = Agent(
    name="Compliance Analyst",
    output_type=ComplianceReport,
    model_settings=ModelSettings(temperature=0.1),
)
```

**What it does:** Reviews the transcript for legal, regulatory, financial, reputational, and conduct risks. Each flag includes the exact statement, speaker, risk type, severity, and explanation.

**What it does NOT do:** See the QA scores or evaluate agent performance in general. It is focused exclusively on risk.

**Why separate from Stage 1:** Combining QA scoring and compliance extraction in one call would mix two different analytical frames and make the output harder to validate, audit, or replace independently.

---

### Agent 3 — Coach

```python
coach = Agent(
    name="Coach",
    output_type=CoachingNotes,
    model_settings=ModelSettings(temperature=0.2),
)
```

**What it does:** Writes personalised coaching feedback for the agent — what went well (minimum 2 examples) and development areas (minimum 2 examples with suggested alternative phrasing). Every point must include an exact agent quote from the transcript.

**Critical constraint:** The coaching prompt is built by `build_coaching_prompt()`, which takes only the parsed transcript, agent name, and call ID. It never receives QA scores, weighted totals, grades, or compliance severities. This is enforced with `assert` statements before the LLM call fires.

**Why coaching is isolated from scores:** Coaching grounded in scores becomes generic ("you scored 6/10 on empathy"). Coaching grounded only in transcript evidence is specific and actionable ("when the customer said X, you responded with Y — consider saying Z instead").

---

## How Scoring Works (Deterministic Code)

The LLM outputs raw dimension scores (integers 1–10). The weighted total and grade are computed entirely in Python:

```python
# Weighted score: each dimension score scaled to its framework weight
total = sum(
    (score / 10.0) * weight * 100
    for each dimension
)

# Grade thresholds
A  >= 85
B  >= 70
C  >= 55
D  >= 40
F  <  40
F  if any auto-fail condition triggered
```

This means the LLM cannot influence the grade directly — it can only influence it through the scores it assigns and the auto-fail conditions it flags.

---

## Structured Output (Schema Enforcement)

All three agents use `output_type=` — the OpenAI Agents SDK's structured output feature. This instructs the API to validate and constrain the model's response to match the Pydantic schema before the application receives it.

This replaces fragile approaches like regex JSON extraction for the main pipeline stages. The Pydantic models are:

| Agent | Schema | Key Fields |
|-------|--------|-----------|
| QA Evaluator | `QAEvaluation` | `dimensions[]`, `auto_fail_checks[]` |
| Compliance Analyst | `ComplianceReport` | `flags[]` with `risk_type`, `severity` |
| Coach | `CoachingNotes` | `what_went_well[]`, `development_areas[]` |

Post-processing in `run_qa_scoring` also filters out any dimension entries whose ID does not appear in the framework (defence against LLM hallucinating extra dimensions despite the prompt constraint).

---

## Transcript Parsing (No LLM)

`src/parser.py` uses pure regex to deterministically extract structure from each `.txt` file before any LLM call happens:

- **Header:** `--- T-XXXX (Agent: Name) ---` → call ID and agent name
- **Turns:** `Speaker: text` → structured turn records with turn numbers
- **Metadata:** issue type (keyword detection), resolution status (keyword detection), escalation signals (customer turn keyword scan), estimated duration (turn count ÷ 3)

This runs first, outputs to `parsed_transcripts/`, and the LLM stages consume those JSON files — never the raw `.txt` files directly.

---

## Optional Stages

These run after the core pipeline if the main stages succeed:

| Stage | What It Does | Output |
|-------|-------------|--------|
| **Team Trend** | Compares current dimension averages against a hardcoded historical baseline in code | `team_trend.json` |
| **Escalation Cases** | Flags any call with a triggered auto-fail or critical compliance finding | `escalation_cases.json` |
| **Calibration Check** | One additional LLM call reviewing all score sheets together to spot inconsistent scoring | `calibration_check.json` |
| **Rebuttal Coaching** | For calls where a customer threatens to report or complain, generates a role-play scenario | `rebuttal_coaching.md` |

---

## LLM Call Audit Log

Every LLM call appends one JSON record to `llm_calls.jsonl`:

```json
{
  "stage": "COACHING_GENERATION",
  "call_id": "T-1041",
  "timestamp": "2025-04-30T10:23:45+00:00",
  "provider": "openrouter",
  "model": "openai/gpt-4o-mini",
  "prompt_hash": "a3f8c2d1e9b04712",
  "input_artifacts": ["parsed_transcripts/T-1041.json"],
  "output_artifact": "coaching_notes.md",
  "qa_scores_included": false
}
```

`qa_scores_included` is always `false` for Stage 3 coaching calls. The `prompt_hash` (SHA-256, first 16 chars) lets you verify a prompt was not modified between runs without storing the full prompt text.

---

## Project Structure

```
.
├── pipeline.py               # Orchestrator — runs all stages in order
├── validate.py               # Validation command
├── qa_framework.json         # QA dimensions and auto-fail conditions (input)
├── transcripts/              # Raw .txt transcript files (input)
├── parsed_transcripts/       # Structured JSON per transcript (generated)
├── qa_scores.json            # QA dimension scores + auto-fail results (generated)
├── compliance_flags.json     # Compliance risk flags per call (generated)
├── coaching_notes.md         # Agent coaching feedback (generated)
├── dashboard_summary.md      # Scores, grades, compliance summary (generated)
├── team_trend.json           # Dimension trend vs historical baseline (generated)
├── escalation_cases.json     # Auto-escalation triggers (generated)
├── calibration_check.json    # Cross-transcript scoring consistency (generated)
├── rebuttal_coaching.md      # Role-play scenarios for threat handling (generated)
├── llm_calls.jsonl           # Full audit log of every LLM call (generated)
├── pipeline_state.json       # Current pipeline stage (generated)
└── src/
    ├── agents_module.py      # Agent definitions, prompts, LLM call functions
    ├── parser.py             # Deterministic transcript parser
    ├── dashboard.py          # Weighted score + grade calculation
    ├── llm_logger.py         # LLM call audit logger
    ├── team_trend.py         # Historical baseline comparison
    ├── escalation.py         # Escalation case generator
    ├── calibration.py        # Cross-transcript calibration check
    └── rebuttal.py           # Rebuttal coaching scenario generator
```

---

## Setup and Running

### Prerequisites

```bash
pip install -r requirements.txt
```

Requires `OPENROUTER_API_KEY` in a `.env` file:

```
OPENROUTER_API_KEY=your_key_here
```

### Run the pipeline

```bash
python pipeline.py
```

This regenerates all output artifacts from scratch. The previous `llm_calls.jsonl` is cleared at the start of each run.

### Validate outputs

```bash
python validate.py
```

Checks all constraints automatically and exits with code `0` (pass) or `1` (fail). Validation covers:

- All required artifacts exist
- All JSON files are valid
- Every transcript has a corresponding parsed JSON
- Every transcript has separate Stage 1, 2, and 3 LLM call records
- All Stage 3 coaching records show `qa_scores_included: false`
- QA dimension IDs match the framework exactly
- All auto-fail conditions are checked for every transcript
- Weighted scores match the formula recalculated from raw dimension scores
- Grades match the deterministic threshold logic
- Coaching notes contain quoted text and cover every call ID
- Dashboard has a row for every processed transcript

---

## Replacing the Framework or Transcripts

The pipeline has no hardcoded dependency on specific call IDs, agent names, or transcript wording. To test with different fixtures:

1. Replace `qa_framework.json` with any valid file using the same `dimensions` + `auto_fail_conditions` structure
2. Replace or add `.txt` files in `transcripts/` using the header format `--- T-XXXX (Agent: Name) ---`
3. Run `python pipeline.py` — all stages regenerate from the new inputs

The parser, scoring, grading, and validation all derive their structure from the framework file at runtime.

---

## Model and Provider

The pipeline uses **OpenRouter** as the API gateway and **GPT-4o-mini** as the default model. These are configured at the top of `src/agents_module.py`:

```python
MODEL = "openai/gpt-4o-mini"
PROVIDER = "openrouter"
```

Any model supported by OpenRouter can be substituted here. The Agents SDK is configured to use OpenRouter's OpenAI-compatible endpoint via `set_default_openai_client()`.
