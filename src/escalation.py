def get_escalation_cases(
    qa_scores: list, compliance_results: list, parsed_transcripts: list
) -> list:
    compliance_by_call = {r["call_id"]: r["flags"] for r in compliance_results}
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
                    "recommended_action": "Immediate supervisor review and coaching session required",
                })

        for flag in compliance_by_call.get(call_id, []):
            if flag.get("severity") == "critical":
                cases.append({
                    "call_id": call_id,
                    "agent_name": qa["agent_name"],
                    "trigger_type": "critical_compliance",
                    "triggered_condition": f"{flag['risk_type']}: {flag['explanation']}",
                    "transcript_excerpt": flag.get("statement_text", "N/A"),
                    "recommended_action": "Compliance team review required within 24 hours",
                })

    return cases
