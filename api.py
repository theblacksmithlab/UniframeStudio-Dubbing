import sys
from datetime import datetime
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn
import os
import json
from modules.job_status import get_or_create_job_status
from dotenv import load_dotenv
from utils.logger_config import setup_logger


logger = setup_logger(name=__name__, log_file='logs/app.log')

app = FastAPI(title="Uniframe Studio Video Processing API", version="1.0.0")


class ProcessVideoRequest(BaseModel):
    job_id: str
    video_url: str
    target_language: str
    tts_provider: str
    tts_voice: str
    source_language: Optional[str] = None
    user_is_premium: bool
    api_keys: Dict[str, str]


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    description: Optional[str] = None
    progress_percentage: Optional[int] = None
    error_message: Optional[str] = None


class JobResult(BaseModel):
    job_id: str
    status: str
    result_urls: Optional[Dict[str, str]] = None
    error_message: Optional[str] = None


@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    try:
        status_data = get_or_create_job_status(job_id)

        return JobStatus(
            job_id=job_id,
            status=status_data["status"],
            created_at=status_data["created_at"],
            completed_at=status_data.get("completed_at"),
            step=status_data.get("step"),
            total_steps=status_data.get("total_steps"),
            description=status_data.get("description"),
            progress_percentage=status_data.get("progress_percentage"),
            error_message=status_data.get("error_message")
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/job/{job_id}/result", response_model=JobResult)
async def get_job_result(job_id: str):
    try:
        status_data = get_or_create_job_status(job_id)

        if status_data["status"] == "processing":
            raise HTTPException(status_code=202, detail="Job still processing")

        return JobResult(
            job_id=job_id,
            status=status_data["status"],
            result_urls=status_data.get("result_urls"),
            error_message=status_data.get("error_message")
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.post("/process-video", response_model=JobStatus)
async def start_video_processing(request: ProcessVideoRequest):
    if not request.job_id:
        raise HTTPException(status_code=400, detail="Job_id is required")

    if not request.video_url:
        raise HTTPException(status_code=400, detail="Video_url is required")

    if not request.target_language:
        raise HTTPException(status_code=400, detail="Target_language is required")

    if not request.tts_provider:
        raise HTTPException(status_code=400, detail="TTS provider is required")
    elif request.tts_provider not in ["openai", "elevenlabs"]:
        raise HTTPException(status_code=400, detail="TTS provider must be either 'openai' or 'elevenlabs'")

    if not request.tts_voice:
        raise HTTPException(status_code=400, detail="TTS voice is required")

    if not request.api_keys:
        raise HTTPException(status_code=400, detail="API-keys is required")

    if "openai" not in request.api_keys or not request.api_keys.get("openai"):
        raise HTTPException(status_code=400, detail="OpenAI API key is required for all operations")

    if request.tts_provider == "elevenlabs" and (
            "elevenlabs" not in request.api_keys or not request.api_keys.get("elevenlabs")):
        raise HTTPException(status_code=400, detail="ElevenLabs API key is required for ElevenLabs TTS provider")

    os.makedirs("jobs", exist_ok=True)
    os.makedirs(f"jobs/{request.job_id}", exist_ok=True)

    job_status = get_or_create_job_status(request.job_id)

    job_params_path = f"jobs/{request.job_id}/job_params.json"
    with open(job_params_path, 'w') as f:
        json.dump(request.model_dump(), f)

    with open(f"jobs/{request.job_id}/pending", 'w') as f:
        f.write(datetime.now().isoformat())

    subprocess.Popen([
        sys.executable,
        "worker.py",
        "--job_id",
        request.job_id
    ])

    return JobStatus(
        job_id=request.job_id,
        status=job_status["status"],
        created_at=job_status["created_at"],
        completed_at=job_status.get("completed_at"),
        step=job_status["step"],
        total_steps=job_status["total_steps"],
        description=job_status["description"],
        progress_percentage=job_status["progress_percentage"],
        error_message=job_status.get("error_message")
    )

# async def process_video_job(request: ProcessVideoRequest):
#     job_id = request.job_id
#
#     try:
#         update_job_status(job_id=job_id, step=2)
#
#         # await download_video_from_s3(request.video_url, job_id)
#
#         fake_download_video_from_s3(request.video_url, job_id)
#
#         result = process_job(
#             job_id=job_id,
#             source_language=request.source_language,
#             target_language=request.target_language,
#             tts_provider=request.tts_provider,
#             tts_voice=request.tts_voice,
#             elevenlabs_api_key=request.api_keys.get("elevenlabs"),
#             openai_api_key=request.api_keys.get("openai"),
#             is_premium=request.user_is_premium
#         )
#
#         if result["status"] == "success":
#             update_job_status(job_id=job_id, step=15)
#
#             result_urls = await upload_results_to_s3(job_id, request.user_is_premium)
#
#             update_job_status(
#                 job_id=job_id,
#                 step=16,
#                 result_urls=result_urls
#             )
#
#             await finalize_job(job_id)
#         else:
#             step = result.get("step")
#
#             update_job_status(
#                 job_id=job_id,
#                 status=JOB_STATUS_FAILED,
#                 step=step,
#                 error_message=result.get("message", "Unknown error")
#             )
#
#     except Exception as e:
#         logger.error(f"Error processing job {job_id}: {str(e)}")
#         update_job_status(
#             job_id=job_id,
#             status=JOB_STATUS_FAILED,
#             error_message=str(e)
#         )

if __name__ == "__main__":
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)