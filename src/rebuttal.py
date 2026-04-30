from src.llm_logger import log_llm_call

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
    return (
        f"Design a rebuttal coaching scenario for {parsed_transcript['agent_name']}.\n\n"
        f"CALL ID: {parsed_transcript['call_id']}\n"
        f"AGENT: {parsed_transcript['agent_name']}\n\n"
        f"The customer made this threatening statement: \"{threat_turn['text']}\"\n\n"
        f"FULL TRANSCRIPT:\n{turns_text}\n\n"
        "Fill in all fields:\n"
        f"- customer_threat_quote: the exact quote above from the transcript\n"
        f"- simulated_customer_message: a realistic variation of that threat for role-play practice\n"
        f"- ideal_agent_response: the best possible response the agent could give\n"
        f"- coaching_goal: what communication skill this scenario develops\n"
        f"- technique_demonstrated: the specific de-escalation or empathy technique used"
    )


def generate_rebuttal_coaching(parsed_transcripts: list) -> list:
    from src.agents_module import rebuttal_agent, _run_with_retry, MODEL, PROVIDER
    from src.agents_module import RebuttalScenario

    results = []
    for pt in parsed_transcripts:
        has_threat, threat_turn = _has_escalation_threat(pt)
        if not has_threat:
            continue
        print(f"  Rebuttal coaching for {pt['call_id']}...")
        prompt = _build_rebuttal_prompt(pt, threat_turn)
        result = _run_with_retry(rebuttal_agent, prompt)
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
        rebuttal_output: RebuttalScenario = result.final_output
        results.append(rebuttal_output.model_dump())
    return results


def rebuttal_to_markdown(scenarios: list) -> str:
    lines = ["# Rebuttal Coaching Scenarios\n"]
    for s in scenarios:
        agent_name = s.get("agent_name", "Unknown")
        call_id = s.get("call_id", "Unknown")
        lines.append(f"## {agent_name} — {call_id}\n")
        lines.append(f"**Customer Threat Quote:** > \"{s.get('customer_threat_quote', '')}\"\n")
        lines.append("### Role-Play Scenario\n")
        lines.append(f"**Simulated Customer Message:** {s.get('simulated_customer_message', '')}\n")
        lines.append(f"**Ideal Agent Response:** {s.get('ideal_agent_response', '')}\n")
        lines.append(f"**Coaching Goal:** {s.get('coaching_goal', '')}\n")
        lines.append(f"**Technique Demonstrated:** {s.get('technique_demonstrated', '')}\n")
        lines.append("---\n")
    return "\n".join(lines)
