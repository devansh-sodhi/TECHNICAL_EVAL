_SEVERITY_ORDER = ["critical", "high", "medium", "low", "none"]


def _highest_severity(flags: list) -> str:
    severities = {f.get("severity", "none").lower() for f in flags}
    for sev in _SEVERITY_ORDER:
        if sev in severities:
            return sev
    return "none"


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


def calculate_weighted_score(dimension_scores: list, framework: dict) -> float:
    weights = {d["id"]: d["weight"] for d in framework["dimensions"]}
    total = sum(
        (ds["score"] / 10.0) * weights.get(ds["dimension_id"], 0) * 100
        for ds in dimension_scores
    )
    return round(total, 2)


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
