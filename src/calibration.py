import json
from src.llm_logger import log_llm_call


def _build_calibration_prompt(qa_scores: list) -> str:
    scores_text = json.dumps(qa_scores, indent=2)
    return f"""You are a QA calibration expert reviewing multiple call scoresheets side by side.

SCORE SHEETS:
{scores_text}

Identify dimensions where similar agent behaviors were scored inconsistently across transcripts.
For each inconsistency provide: dimension_id, call_ids (list), observed_difference, why_it_may_be_inconsistent, recommended_adjustment.
If no inconsistencies found, return an empty inconsistencies list."""


def run_calibration_check(qa_scores: list) -> dict:
    from src.agents_module import calibration_agent, _run_with_retry, MODEL, PROVIDER
    from src.agents_module import CalibrationReport

    prompt = _build_calibration_prompt(qa_scores)
    result = _run_with_retry(calibration_agent, prompt)

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

    calibration_output: CalibrationReport = result.final_output
    return {"inconsistencies": [i.model_dump() for i in calibration_output.inconsistencies]}
