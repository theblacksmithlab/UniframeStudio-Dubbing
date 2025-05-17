from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn
import boto3
import os
import json
from urllib.parse import urlparse
import logging
from processing_pipeline import process_job
from modules.job_status import get_or_create_job_status, update_job_status, JOB_STATUS_FAILED, finalize_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Processing API", version="1.0.0")


# Pydantic models
class ProcessVideoRequest(BaseModel):
    job_id: str
    video_url: str
    target_language: str
    tts_provider: str
    tts_voice: str
    source_language: Optional[str] = None
    user_is_premium: bool = False
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
    """Get job status"""
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
    """Get job result with S3 URLs"""
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
async def start_video_processing(
        request: ProcessVideoRequest,
        background_tasks: BackgroundTasks
):
    job_status = get_or_create_job_status(request.job_id)

    background_tasks.add_task(process_video_job, request)

    return JobStatus(
        job_id=request.job_id,
        status=job_status["status"],
        created_at=job_status["created_at"],
        step=job_status["step"],
        total_steps=job_status["total_steps"],
        description=job_status["description"],
        progress_percentage=job_status["progress_percentage"],
        completed_at=job_status.get("completed_at"),
        error_message=job_status.get("error_message")
    )

async def process_video_job(request: ProcessVideoRequest):
    job_id = request.job_id

    try:
        update_job_status(job_id=job_id, step=2)

        await download_video_from_s3(request.video_url, job_id)

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
            update_job_status(job_id=job_id, step=15)

            result_urls = await upload_results_to_s3(job_id, request.user_is_premium)

            update_job_status(
                job_id=job_id,
                step=16,
                result_urls=result_urls
            )

            await finalize_job(job_id)
        else:
            update_job_status(
                job_id=job_id,
                status=JOB_STATUS_FAILED,
                error_message=result.get("message", "Unknown error")
            )

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        update_job_status(
            job_id=job_id,
            status=JOB_STATUS_FAILED,
            error_message=str(e)
        )


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


async def upload_results_to_s3(job_id: str, is_premium: bool = False) -> Dict[str, str]:
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

    # Определяем файлы для загрузки в зависимости от премиум статуса
    files_to_upload = {}

    # Общие файлы для всех типов пользователей
    files_to_upload.update({
        'translated': pipeline_result['output_files']['translated'],
        'audio': pipeline_result['output_files']['final_audio'],
        'audio_stereo': pipeline_result['output_files']['final_audio_stereo'],
        'video_premium': pipeline_result['output_files']['final_video_premium'],
    })

    # Дополнительные файлы для обычных пользователей
    if not is_premium and 'final_video' in pipeline_result['output_files']:
        files_to_upload.update({
            'video': pipeline_result['output_files']['final_video'],
            'audio_with_ads': pipeline_result['output_files'].get('final_audio_with_ads', ''),
            'audio_stereo_with_ads': pipeline_result['output_files'].get('final_audio_stereo_with_ads', '')
        })

    # Опционально добавляем видео без звука для отладки
    if 'final_video_mute_premium' in pipeline_result['output_files']:
        files_to_upload['video_mute_premium'] = pipeline_result['output_files']['final_video_mute_premium']

    if not is_premium and 'final_video_mute' in pipeline_result['output_files']:
        files_to_upload['video_mute'] = pipeline_result['output_files']['final_video_mute']

    # Загружаем файлы в S3
    for file_type, local_path in files_to_upload.items():
        if local_path and os.path.exists(local_path):
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