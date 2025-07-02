import os
import json
import fcntl
from typing import Dict, Any, Optional
from datetime import datetime
from utils.logger_config import setup_logger


logger = setup_logger(name=__name__, log_file="logs/app.log")

JOB_TOTAL_STEPS = 18

STEP_DESCRIPTIONS = {
    1: "Initializing dubbing system components...",
    2: "Downloading video from S3 storage...",
    3: "Extracting audio from video...",
    4: "Transcribing extracted audio...",
    5: "Structuring transcription...",
    6: "Cleaning transcription...",
    7: "Optimizing transcription...",
    8: "Adjusting transcription segments timing...",
    9: "Translating transcription segments...",
    10: "Transcription review required...",
    11: "Generating TTS segments...",
    12: "Auto-correcting segment durations...",
    13: "Processing background audio...",
    14: "Creating all audio files...",
    15: "Processing video with new audio...",
    16: "Creating final video with stereo audio...",
    17: "Uploading results to S3 storage...",
    18: "Finalizing dubbing job..."
}

JOB_STATUS_PROCESSING = "processing"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"


def get_or_create_job_status(job_id: str) -> Dict[str, Any]:
    status_file = f"jobs/{job_id}/status.json"

    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in status file: {status_file}")

    status = {
        "status": JOB_STATUS_PROCESSING,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "step": 1,
        "total_steps": JOB_TOTAL_STEPS,
        "step_description": STEP_DESCRIPTIONS[1],
        "progress_percentage": round((1 / max(JOB_TOTAL_STEPS, 1)) * 100),
        "error_message": None,
        "result_urls": None,
        "processing_steps": list(STEP_DESCRIPTIONS.values())
    }

    os.makedirs(os.path.dirname(status_file), exist_ok=True)

    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)

    return status


def update_job_status(job_id: str,
                      step: Optional[int] = None,
                      status: Optional[str] = None,
                      error_message: Optional[str] = None,
                      completed_at: Optional[str] = None,
                      result_urls: Optional[Dict[str, str]] = None,
                      **other_updates):
    status_file = f"jobs/{job_id}/status.json"

    os.makedirs(os.path.dirname(status_file), exist_ok=True)

    try:
        if os.path.exists(status_file):
            with open(status_file, 'r+') as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    current_status = json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        else:
            current_status = get_or_create_job_status(job_id)

        updates = {}

        if step is not None:
            updates["step"] = step
            updates["total_steps"] = JOB_TOTAL_STEPS

            if step in STEP_DESCRIPTIONS:
                updates["step_description"] = STEP_DESCRIPTIONS[step]

            updates["progress_percentage"] = round((step / JOB_TOTAL_STEPS) * 100)

            if step == JOB_TOTAL_STEPS and status is None:
                updates["status"] = JOB_STATUS_COMPLETED
                updates["completed_at"] = datetime.now().isoformat()

        if status is not None:
            updates["status"] = status

            if status == JOB_STATUS_COMPLETED and completed_at is None:
                updates["completed_at"] = datetime.now().isoformat()

            if status == JOB_STATUS_FAILED and "step_description" not in updates:
                updates["step_description"] = "Failed" if error_message is None else f"Failed: {error_message}"

        if error_message is not None:
            updates["error_message"] = error_message
        if completed_at is not None:
            updates["completed_at"] = completed_at
        if result_urls is not None:
            updates["result_urls"] = result_urls

        updates.update(other_updates)

        current_status.update(updates)

        with open(status_file, 'w') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(current_status, f, indent=2)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        return True

    except Exception as e:
        logger.error(f"Error updating job status for {job_id}: {str(e)}")
        return False


def finalize_job(job_id: str):
    try:
        update_job_status(
            job_id=job_id,
            status=JOB_STATUS_COMPLETED,
            completed_at=datetime.now().isoformat(),
            progress_percentage=100
        )
        logger.info(f"Job: {job_id} finalized successfully")
        return True
    except Exception as e:
        logger.error(f"Error during job finalization: {e}")
        return False
