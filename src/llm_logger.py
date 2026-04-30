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
