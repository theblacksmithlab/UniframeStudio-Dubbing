import os
import json
import fcntl
import shutil
from typing import Dict, Any, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JOB_TOTAL_STEPS = 16

STEP_DESCRIPTIONS = {
    1: "Initializing job...",
    2: "Downloading video from S3...",
    3: "Extracting audio from video...",
    4: "Transcribing extracted audio...",
    5: "Structuring transcription...",
    6: "Cleaning transcription...",
    7: "Optimizing transcription...",
    8: "Adjusting transcription segments timing...",
    9: "Translating transcription segments...",
    10: "Generating TTS segments...",
    11: "Auto-correcting segment durations...",
    12: "Creating final audio files...",
    13: "Processing video with new audio...",
    14: "Creating final videos with stereo audio...",
    15: "Uploading results to S3...",
    16: "Finalizing..."
}

JOB_STATUS_PROCESSING = "processing"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"

def get_or_create_job_status(job_id: str) -> Dict[str, Any]:
    """Get job status from file or create new one"""
    status_file = f"jobs/{job_id}/status.json"

    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            return json.load(f)
    else:
        os.makedirs(f"jobs/{job_id}", exist_ok=True)
        status = {
            "status": JOB_STATUS_PROCESSING,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "step": 1,
            "total_steps": JOB_TOTAL_STEPS,
            "description": STEP_DESCRIPTIONS[1],
            "progress_percentage": round((1 / JOB_TOTAL_STEPS) * 100),
            "error_message": None,
            "result_urls": None
        }
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
                updates["description"] = STEP_DESCRIPTIONS[step]

            updates["progress_percentage"] = round((step / JOB_TOTAL_STEPS) * 100)

            if step == JOB_TOTAL_STEPS and status is None:
                updates["status"] = JOB_STATUS_COMPLETED
                updates["completed_at"] = datetime.now().isoformat()

        if status is not None:
            updates["status"] = status

            if status == JOB_STATUS_COMPLETED and completed_at is None:
                updates["completed_at"] = datetime.now().isoformat()

            if status == JOB_STATUS_FAILED and "description" not in updates:
                updates["description"] = "Failed" if error_message is None else f"Failed: {error_message}"

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
        print(f"Error updating job status for {job_id}: {str(e)}")
        return False


async def finalize_job(job_id: str):
    try:
        status_file = f"jobs/{job_id}/status.json"
        job_result_dir = f"jobs/{job_id}/job_result"

        if os.path.exists(status_file) and os.path.exists(job_result_dir):
            shutil.copy(status_file, os.path.join(job_result_dir, "status.json"))

            os.remove(status_file)

            logger.info(f"Status file moved to job_result for job {job_id}")
    except Exception as e:
        logger.error(f"Error moving status file: {e}")
