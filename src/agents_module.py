import os
import json
import re
from typing import List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, ModelSettings, set_default_openai_client, set_default_openai_api, set_tracing_disabled
from pydantic import BaseModel

load_dotenv()

MODEL = "openai/gpt-4o-mini"
PROVIDER = "openrouter"

_async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

set_default_openai_client(_async_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


# ── Pydantic output types (SDK enforces schema via output_type=) ───────────

class DimensionScore(BaseModel):
    dimension_id: str
    score: int
    evidence_quote: str
    rationale: str

class AutoFailCheck(BaseModel):
    condition: str
    triggered: bool
    evidence: Optional[str] = None

class QAEvaluation(BaseModel):
    dimensions: List[DimensionScore]
    auto_fail_checks: List[AutoFailCheck]

class ComplianceFlag(BaseModel):
    statement_text: str
    speaker: str
    risk_type: str
    severity: str
    explanation: str

class ComplianceReport(BaseModel):
    flags: List[ComplianceFlag]

class WhatWentWellItem(BaseModel):
    point: str
    agent_quote: str
    context: str

class DevelopmentAreaItem(BaseModel):
    area: str
    agent_quote: str
    suggested_alternative: str
    why: str

class CoachingNotes(BaseModel):
    agent_name: str
    call_id: str
    what_went_well: List[WhatWentWellItem]
    development_areas: List[DevelopmentAreaItem]


class CalibrationInconsistency(BaseModel):
    dimension_id: str
    call_ids: List[str]
    observed_difference: str
    why_it_may_be_inconsistent: str
    recommended_adjustment: str

class CalibrationReport(BaseModel):
    inconsistencies: List[CalibrationInconsistency]


class RebuttalScenario(BaseModel):
    call_id: str
    agent_name: str
    customer_threat_quote: str
    simulated_customer_message: str
    ideal_agent_response: str
    coaching_goal: str
    technique_demonstrated: str


# ── Fallback: only used if output_type cannot be applied ──
def extract_json(text: str) -> dict:
    text = text.strip()
    # strip // comments and trailing commas (common LLM quirks)
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
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


def _run_with_retry(agent, prompt: str, max_attempts: int = 3):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return Runner.run_sync(agent, prompt)
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                print(f"    Retry {attempt}/{max_attempts - 1} after error: {e}")
    raise last_error


# ── Agent 1: QA Evaluator — output_type=QAEvaluation ──────────────────────

qa_evaluator = Agent(
    name="QA Evaluator",
    model=MODEL,
    model_settings=ModelSettings(temperature=0.1),
    output_type=QAEvaluation,
    instructions=(
        "You are a professional QA analyst for a customer support team. "
        "Evaluate support call transcripts against the QA framework provided. "
        "Score each dimension 1-10 using exact quotes from the transcript as evidence. "
        "Check every auto-fail condition explicitly — set triggered=true only if clearly violated."
    ),
)


def build_qa_prompt(parsed_transcript: dict, framework: dict) -> str:
    turns_text = "\n".join(f"{t['speaker']}: {t['text']}" for t in parsed_transcript["turns"])
    dims_json = json.dumps(framework["dimensions"], indent=2)
    auto_fails_json = json.dumps(framework["auto_fail_conditions"], indent=2)
    return (
        f"Evaluate this customer support call.\n\n"
        f"CALL ID: {parsed_transcript['call_id']}\n"
        f"AGENT: {parsed_transcript['agent_name']}\n\n"
        f"TRANSCRIPT:\n{turns_text}\n\n"
        f"QA DIMENSIONS (score each 1-10):\n{dims_json}\n\n"
        f"IMPORTANT: Score EXACTLY these {len(framework['dimensions'])} dimension IDs: "
        f"{', '.join(d['id'] for d in framework['dimensions'])}. "
        f"Do not add, rename, or omit any dimension.\n\n"
        f"AUTO-FAIL CONDITIONS — set triggered=true ONLY if the condition is clearly and "
        f"unambiguously violated with direct evidence from the transcript:\n{auto_fails_json}\n\n"
        "AUTO-FAIL GUIDANCE:\n"
        "- 'Promises a specific outcome they cannot guarantee': triggered ONLY if the agent "
        "makes a definite promise (e.g. 'I guarantee', 'it WILL happen'). "
        "Hedged estimates like 'typically within 24 hours' or explicitly saying "
        "'I can't guarantee' are NOT triggers — they are good expectation management.\n"
        "- 'Discloses account details without identity verification': triggered ONLY if the "
        "agent reveals sensitive account data without first requesting ANY identifying information "
        "from the customer. Asking for and receiving the customer's email address, account number, "
        "or any other personal identifier constitutes completing identity verification. "
        "If the agent asked 'what is your email?' and the customer provided it before the agent "
        "discussed account details, this condition is NOT triggered.\n"
        "- 'Uses dismissive or confrontational language': triggered if the agent is dismissive, "
        "deflecting, or shows indifference to the customer's distress. Examples that SHOULD trigger: "
        "'That's just how it works', 'There's not much I can do', 'That's your right' (as a brush-off "
        "when a customer threatens to complain), offering no empathy or alternatives when a customer is "
        "clearly upset. Examples that should NOT trigger: empathetic language, professional boundary-setting "
        "with explanation, or appropriately managing expectations.\n"
        "- 'Provides incorrect regulatory information': triggered ONLY if the agent states "
        "a regulation incorrectly as fact. Uncertainty or hedging is not a trigger.\n\n"
        f"Evaluate all {len(framework['dimensions'])} dimensions and check all "
        f"{len(framework['auto_fail_conditions'])} auto-fail conditions."
    )


def run_qa_scoring(parsed_transcript: dict, framework: dict) -> dict:
    from src.llm_logger import log_llm_call
    from src.dashboard import calculate_weighted_score, calculate_grade

    prompt = build_qa_prompt(parsed_transcript, framework)
    result = _run_with_retry(qa_evaluator, prompt)

    log_llm_call(
        stage="QA_SCORING",
        call_id=parsed_transcript["call_id"],
        provider=PROVIDER,
        model=MODEL,
        prompt=prompt,
        input_artifacts=[
            f"parsed_transcripts/{parsed_transcript['call_id']}.json",
            "qa_framework.json",
        ],
        output_artifact="qa_scores.json",
        qa_scores_included=False,
    )

    # result.final_output is a typed QAEvaluation — no JSON parsing needed
    qa_output: QAEvaluation = result.final_output

    # Filter to only framework dimension IDs and deduplicate — guards against LLM hallucination
    framework_ids = [d["id"] for d in framework["dimensions"]]
    seen = set()
    dimension_scores = []
    for d in qa_output.dimensions:
        if d.dimension_id in framework_ids and d.dimension_id not in seen:
            dimension_scores.append(d.model_dump())
            seen.add(d.dimension_id)

    auto_fail_checks = [a.model_dump() for a in qa_output.auto_fail_checks]

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


# ── Agent 2: Compliance Analyst — output_type=ComplianceReport ────────────

compliance_analyst = Agent(
    name="Compliance Analyst",
    model=MODEL,
    model_settings=ModelSettings(temperature=0.1),
    output_type=ComplianceReport,
    instructions=(
        "You are a compliance officer reviewing customer support calls. "
        "Identify ALL legal, regulatory, financial, reputational, and conduct risks. "
        "Be thorough — include risks at any severity level (critical/high/medium/low). "
        "If there are no risks, return an empty flags list."
    ),
)


def build_compliance_prompt(parsed_transcript: dict) -> str:
    turns_text = "\n".join(f"{t['speaker']}: {t['text']}" for t in parsed_transcript["turns"])
    return (
        f"Review this transcript for compliance risks.\n\n"
        f"CALL ID: {parsed_transcript['call_id']}\n\n"
        f"TRANSCRIPT:\n{turns_text}\n\n"
        "Identify ALL compliance, legal, financial, regulatory, reputational, or conduct risks. "
        "For each flag provide: exact statement text, speaker name, risk_type "
        "(regulatory|financial_commitment|data_disclosure|conduct|reputational|other), "
        "severity (critical|high|medium|low), and a brief explanation."
    )


def run_compliance_extraction(parsed_transcript: dict) -> dict:
    from src.llm_logger import log_llm_call

    prompt = build_compliance_prompt(parsed_transcript)
    result = _run_with_retry(compliance_analyst, prompt)

    log_llm_call(
        stage="COMPLIANCE_EXTRACTION",
        call_id=parsed_transcript["call_id"],
        provider=PROVIDER,
        model=MODEL,
        prompt=prompt,
        input_artifacts=[f"parsed_transcripts/{parsed_transcript['call_id']}.json"],
        output_artifact="compliance_flags.json",
        qa_scores_included=False,
    )

    compliance_output: ComplianceReport = result.final_output
    flags = [f.model_dump() for f in compliance_output.flags]
    for flag in flags:
        flag["call_id"] = parsed_transcript["call_id"]

    return {
        "call_id": parsed_transcript["call_id"],
        "agent_name": parsed_transcript["agent_name"],
        "flags": flags,
    }


# ── Agent 3: Coach — output_type=CoachingNotes, NO QA scores ever ─────────

coach = Agent(
    name="Coach",
    model=MODEL,
    model_settings=ModelSettings(temperature=0.2),
    output_type=CoachingNotes,
    instructions=(
        "You are a customer support performance coach writing personalised feedback. "
        "Write coaching notes based ONLY on what you observe in the transcript. "
        "NEVER reference scores, ratings, grades, or metrics of any kind. "
        "Every coaching point must include an exact agent quote from the transcript. "
        "Provide at least 2 what_went_well examples and at least 2 development_areas."
    ),
)


def build_coaching_prompt(parsed_transcript: dict) -> str:
    # This function intentionally excludes all QA scores, grades, and compliance data.
    turns_text = "\n".join(f"{t['speaker']}: {t['text']}" for t in parsed_transcript["turns"])
    return (
        f"Write coaching notes for {parsed_transcript['agent_name']}.\n\n"
        f"CALL ID: {parsed_transcript['call_id']}\n"
        f"AGENT: {parsed_transcript['agent_name']}\n\n"
        f"TRANSCRIPT:\n{turns_text}\n\n"
        "Base ALL feedback only on what you observe in this transcript. "
        "Every coaching point must include an exact agent quote. "
        "Provide at least 2 what_went_well examples and at least 2 development_areas."
    )


def run_coaching(parsed_transcript: dict) -> dict:
    from src.llm_logger import log_llm_call

    prompt = build_coaching_prompt(parsed_transcript)

    # Enforce: coaching prompt must not contain QA scores
    assert "weighted score" not in prompt.lower(), "QA scores must not appear in coaching prompt"
    assert "grade:" not in prompt.lower(), "Grades must not appear in coaching prompt"

    result = _run_with_retry(coach, prompt)

    log_llm_call(
        stage="COACHING_GENERATION",
        call_id=parsed_transcript["call_id"],
        provider=PROVIDER,
        model=MODEL,
        prompt=prompt,
        input_artifacts=[f"parsed_transcripts/{parsed_transcript['call_id']}.json"],
        output_artifact="coaching_notes.md",
        qa_scores_included=False,  # MUST always be False
    )

    coaching_output: CoachingNotes = result.final_output
    return coaching_output.model_dump()


# ── Agent 4: Calibration Reviewer — output_type=CalibrationReport ─────────

calibration_agent = Agent(
    name="Calibration Reviewer",
    model=MODEL,
    model_settings=ModelSettings(temperature=0.1),
    output_type=CalibrationReport,
    instructions=(
        "You are a QA calibration expert reviewing multiple scoresheets side by side. "
        "Identify dimensions where similar agent behaviours were scored inconsistently across calls. "
        "If no inconsistencies exist, return an empty inconsistencies list."
    ),
)


# ── Agent 5: Rebuttal Coach — output_type=RebuttalScenario ────────────────

rebuttal_agent = Agent(
    name="Rebuttal Coach",
    model=MODEL,
    model_settings=ModelSettings(temperature=0.2),
    output_type=RebuttalScenario,
    instructions=(
        "You are a customer support coach designing role-play scenarios for handling customer threats or complaints. "
        "Create a practical coaching scenario based on the transcript provided."
    ),
)
