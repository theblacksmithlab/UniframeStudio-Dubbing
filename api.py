import sys
from datetime import datetime
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn
import os
import json
from modules.job_status import get_or_create_job_status
from dotenv import load_dotenv
from utils.logger_config import setup_logger, get_job_logger

logger = setup_logger(name=__name__, log_file="logs/app.log")


app = FastAPI(title="Uniframe Studio Video Processing API", version="1.0.0")


class ProcessVideoRequest(BaseModel):
    job_id: str
    video_url: str
    target_language: str
    tts_provider: str
    tts_voice: str
    source_language: Optional[str] = None
    transcription_keywords: Optional[str] = None
    enable_vad: bool = True


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    step_description: Optional[str] = None
    progress_percentage: Optional[int] = None
    error_message: Optional[str] = None
    processing_steps: Optional[List[str]] = None
    review_required_url: Optional[str] = None


class JobResult(BaseModel):
    job_id: str
    status: str
    result_urls: Optional[Dict[str, str]] = None
    error_message: Optional[str] = None


@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    job_logger = get_job_logger(logger, job_id)

    job_logger.info(f"Got job status request")

    try:
        status_data = get_or_create_job_status(job_id)

        return JobStatus(
            job_id=job_id,
            status=status_data["status"],
            created_at=status_data["created_at"],
            completed_at=status_data.get("completed_at"),
            step=status_data.get("step"),
            total_steps=status_data.get("total_steps"),
            step_description=status_data.get("step_description"),
            progress_percentage=status_data.get("progress_percentage"),
            error_message=status_data.get("error_message"),
            processing_steps=status_data.get("processing_steps"),
            review_required_url=status_data.get("review_required_url"),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/job/{job_id}/result", response_model=JobResult)
async def get_job_result(job_id: str):
    job_logger = get_job_logger(logger, job_id)

    job_logger.info(f"Got job result request")

    try:
        status_data = get_or_create_job_status(job_id)

        if status_data["status"] == "processing":
            raise HTTPException(status_code=202, detail="Job still processing")

        return JobResult(
            job_id=job_id,
            status=status_data["status"],
            result_urls=status_data.get("result_urls"),
            error_message=status_data.get("error_message"),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.post("/process-video", response_model=JobStatus)
async def start_video_processing(request: ProcessVideoRequest):
    load_dotenv()

    job_id = request.job_id
    job_logger = get_job_logger(logger, job_id)

    job_logger.info(f"Got new dubbing job request")

    if not job_id:
        raise HTTPException(status_code=400, detail="Job_id is required")

    if not request.video_url:
        raise HTTPException(status_code=400, detail="Video_url is required")

    if not request.target_language:
        raise HTTPException(status_code=400, detail="Target_language is required")

    if not request.tts_provider:
        raise HTTPException(status_code=400, detail="TTS provider is required")
    elif request.tts_provider not in ["openai", "elevenlabs"]:
        raise HTTPException(
            status_code=400,
            detail="TTS provider must be either 'openai' or 'elevenlabs'",
        )

    if not request.tts_voice:
        raise HTTPException(status_code=400, detail="TTS voice is required")

    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured in internal service's environment"
        )

    if request.tts_provider == "elevenlabs":
        elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
        if not elevenlabs_key:
            raise HTTPException(
                status_code=500,
                detail="ElevenLabs API key not configured in internal service's environment"
            )

    os.makedirs("jobs", exist_ok=True)
    os.makedirs(f"jobs/{job_id}", exist_ok=True)

    job_status = get_or_create_job_status(job_id)

    job_params_path = f"jobs/{job_id}/job_params.json"
    with open(job_params_path, "w") as f:
        json.dump(request.model_dump(), f)

    with open(f"jobs/{job_id}/pending", "w") as f:
        f.write(datetime.now().isoformat())

    subprocess.Popen([sys.executable, "worker.py", "--job_id", job_id])

    return JobStatus(
        job_id=job_id,
        status=job_status["status"],
        created_at=job_status["created_at"],
        completed_at=job_status.get("completed_at"),
        step=job_status["step"],
        total_steps=job_status["total_steps"],
        step_description=job_status["step_description"],
        progress_percentage=job_status["progress_percentage"],
        error_message=job_status.get("error_message"),
        processing_steps=job_status.get("processing_steps"),
    )


@app.get("/health")
async def health_check():
    logger.info("Got health check request...")

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "uniframe-dubbing-service"
    }

if __name__ == "__main__":
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)
