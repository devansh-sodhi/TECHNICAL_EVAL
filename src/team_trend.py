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
