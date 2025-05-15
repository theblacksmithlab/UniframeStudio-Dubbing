from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import boto3
import os
import json
import shutil
from urllib.parse import urlparse
from datetime import datetime
import logging
from processing_pipeline import process_job

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Processing API", version="1.0.0")


# Pydantic models
class ProcessVideoRequest(BaseModel):
    job_id: str
    video_url: str
    target_language: str
    tts_provider: str  # "openai" or "elevenlabs"
    tts_voice: str
    source_language: Optional[str] = None
    user_is_premium: bool = False
    api_keys: Dict[str, str]

class JobStatus(BaseModel):
    job_id: str
    status: str  # "processing", "completed", "failed"
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


def get_or_create_job_status(job_id: str) -> Dict[str, Any]:
    """Get job status from file or create new one"""
    status_file = f"jobs/{job_id}/status.json"

    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            return json.load(f)
    else:
        os.makedirs(f"jobs/{job_id}", exist_ok=True)
        status = {
            "status": "processing",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "step": 0,
            "total_steps": 10,
            "description": "Initializing...",
            "progress_percentage": 0,
            "error_message": None,
            "result_urls": None
        }
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        return status


def update_job_status(job_id: str, updates: Dict[str, Any]):
    """Update job status file with new data"""
    status_file = f"jobs/{job_id}/status.json"

    # Get current status
    status = get_or_create_job_status(job_id)

    # Update with new data
    status.update(updates)

    # Save back to file
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


@app.post("/process-video", response_model=Dict[str, str])
async def start_video_processing(
        request: ProcessVideoRequest,
        background_tasks: BackgroundTasks
):
    """Start video processing job"""

    # Initialize job status
    get_or_create_job_status(request.job_id)

    # Start processing in background
    background_tasks.add_task(process_video_job, request)

    return {"job_id": request.job_id, "status": "processing"}


@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status"""
    status_file = f"jobs/{job_id}/status.json"

    if not os.path.exists(status_file):
        raise HTTPException(status_code=404, detail="Job not found")

    with open(status_file, 'r') as f:
        status_data = json.load(f)

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


@app.get("/job/{job_id}/result", response_model=JobResult)
async def get_job_result(job_id: str):
    """Get job result with S3 URLs"""
    status_file = f"jobs/{job_id}/status.json"

    if not os.path.exists(status_file):
        raise HTTPException(status_code=404, detail="Job not found")

    with open(status_file, 'r') as f:
        status_data = json.load(f)

    if status_data["status"] == "processing":
        raise HTTPException(status_code=202, detail="Job still processing")

    return JobResult(
        job_id=job_id,
        status=status_data["status"],
        result_urls=status_data.get("result_urls"),
        error_message=status_data.get("error_message")
    )


async def process_video_job(request: ProcessVideoRequest):
    """Background task to process video"""
    job_id = request.job_id

    try:
        # Update status: Downloading
        update_job_status(job_id, {
            "description": "Downloading video from S3...",
            "progress_percentage": 0
        })

        # Download video from S3
        video_path = await download_video_from_s3(request.video_url, job_id)

        # Update status: Processing
        update_job_status(job_id, {
            "description": "Starting video processing...",
            "progress_percentage": 5
        })

        # Run the actual video processing
        result = process_job(
            job_id=job_id,
            source_language=request.source_language,
            target_language=request.target_language,
            tts_provider=request.tts_provider,
            tts_voice=request.tts_voice,
            elevenlabs_api_key=request.api_keys.get("elevenlabs"),
            openai_api_key=request.api_keys.get("openai"),
            is_premium=request.user_is_premium
        )

        if result["status"] == "success":
            # Upload results to S3
            update_job_status(job_id, {
                "description": "Uploading results to S3...",
                "progress_percentage": 95
            })
            result_urls = await upload_results_to_s3(job_id)

            # Mark as completed
            update_job_status(job_id, {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "result_urls": result_urls,
                "description": "Completed successfully",
                "progress_percentage": 100
            })
        else:
            # Mark as failed
            update_job_status(job_id, {
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "error_message": result.get("message", "Unknown error"),
                "description": "Failed",
                "progress_percentage": 0
            })

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        update_job_status(job_id, {
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error_message": str(e),
            "description": "Failed with exception",
            "progress_percentage": 0
        })


async def download_video_from_s3(video_url: str, job_id: str) -> str:
    """Download video from S3 to local job directory"""
    # Parse S3 URL (s3://bucket/path/to/video.mp4)
    parsed = urlparse(video_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')

    # Create local directory
    video_input_dir = f"jobs/{job_id}/video_input"
    os.makedirs(video_input_dir, exist_ok=True)

    # Extract filename from key
    filename = os.path.basename(key)
    local_path = os.path.join(video_input_dir, filename)

    # Download from S3
    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded video from {video_url} to {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download video from S3: {e}")
        raise


async def upload_results_to_s3(job_id: str) -> Dict[str, str]:
    """Upload processed results to S3"""
    # Read pipeline result to get file paths
    job_result_dir = f"jobs/{job_id}/job_result"
    pipeline_result_path = os.path.join(job_result_dir, "pipeline_result.json")

    with open(pipeline_result_path, 'r') as f:
        pipeline_result = json.load(f)

    # Extract S3 bucket from environment or config
    s3_bucket = os.getenv('S3_BUCKET', 'your-default-bucket')
    s3_client = boto3.client('s3')

    result_urls = {}

    # Files to upload
    files_to_upload = {
        'video': pipeline_result['output_files']['final_video'],
        'audio': pipeline_result['output_files']['final_audio'],
        'transcription': pipeline_result['output_files']['translated']
    }

    # Add stereo audio if exists
    if pipeline_result['output_files'].get('final_audio_stereo'):
        files_to_upload['audio_stereo'] = pipeline_result['output_files']['final_audio_stereo']

    for file_type, local_path in files_to_upload.items():
        if os.path.exists(local_path):
            # S3 key: jobs/{job_id}/results/{filename}
            filename = os.path.basename(local_path)
            s3_key = f"jobs/{job_id}/results/{filename}"

            try:
                s3_client.upload_file(local_path, s3_bucket, s3_key)
                result_urls[file_type] = f"s3://{s3_bucket}/{s3_key}"
                logger.info(f"Uploaded {file_type} to {result_urls[file_type]}")
            except Exception as e:
                logger.error(f"Failed to upload {file_type}: {e}")
                raise

    return result_urls


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)