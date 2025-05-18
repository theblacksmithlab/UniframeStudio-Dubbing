import argparse
import json
import os
import sys
import shutil
import boto3
from botocore.exceptions import ClientError
from processing_pipeline import process_job
from modules.job_status import update_job_status, finalize_job, JOB_STATUS_FAILED
from dotenv import load_dotenv
from utils.logger_config import setup_logger


logger = setup_logger(name=__name__, log_file='logs/app.log')


def download_video_from_s3(video_url: str, job_id: str) -> str:
    if not video_url.startswith('s3://'):
        raise ValueError(f"Invalid S3 URL format: {video_url}")

    parts = video_url[5:].split('/', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URL: {video_url}")

    bucket = parts[0]
    key = parts[1]

    video_input_dir = f"jobs/{job_id}/video_input"
    os.makedirs(video_input_dir, exist_ok=True)

    filename = os.path.basename(key)
    local_path = os.path.join(video_input_dir, filename)

    s3_client = boto3.client('s3')
    try:
        logger.info(f"Downloading video-file from {video_url} to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Successfully downloaded video-file: {local_path}")
        return local_path
    except ClientError as e:
        logger.error(f"Error downloading video-file from S3: {e}")
        raise


def upload_results_to_s3(job_id: str, is_premium: bool = False) -> dict:
    job_result_dir = f"jobs/{job_id}/job_result"
    pipeline_result_path = os.path.join(job_result_dir, "pipeline_result.json")

    if not os.path.exists(pipeline_result_path):
        logger.error(f"Pipeline result file not found: {pipeline_result_path}")
        raise FileNotFoundError(f"Pipeline result not found for job {job_id}")

    with open(pipeline_result_path, 'r') as f:
        pipeline_result = json.load(f)

    s3_bucket = os.getenv('S3_BUCKET')
    if not s3_bucket:
        logger.error("S3_BUCKET environment variable not set")
        raise EnvironmentError("S3_BUCKET environment variable must be set")

    s3_client = boto3.client('s3')
    result_urls = {}

    files_to_upload = {}

    files_to_upload.update({
        'translated': pipeline_result['output_files']['translated'],
        'audio': pipeline_result['output_files']['final_audio'],
        'audio_stereo': pipeline_result['output_files']['final_audio_stereo'],
        'video_premium': pipeline_result['output_files']['final_video_premium'],
    })

    if not is_premium:
        files_to_upload.update({
            'video': pipeline_result['output_files'].get('final_video', ''),
            'audio_with_ads': pipeline_result['output_files'].get('final_audio_with_ads', ''),
            'audio_stereo_with_ads': pipeline_result['output_files'].get('final_audio_stereo_with_ads', '')
        })

    if 'final_video_mute_premium' in pipeline_result['output_files']:
        files_to_upload['video_mute_premium'] = pipeline_result['output_files']['final_video_mute_premium']

    if not is_premium and 'final_video_mute' in pipeline_result['output_files']:
        files_to_upload['video_mute'] = pipeline_result['output_files']['final_video_mute']

    for file_type, local_path in files_to_upload.items():
        if local_path and os.path.exists(local_path):
            filename = os.path.basename(local_path)
            s3_key = f"jobs/{job_id}/results/{filename}"

            try:
                logger.info(f"Uploading {file_type} to S3: {s3_key}")
                s3_client.upload_file(local_path, s3_bucket, s3_key)
                result_urls[file_type] = f"s3://{s3_bucket}/{s3_key}"
                logger.info(f"Successfully uploaded {file_type} to {result_urls[file_type]}")
            except ClientError as e:
                logger.error(f"Error uploading {file_type} to S3: {e}")
                continue

    return result_urls


def cleanup_job_files(job_id):
    job_base_dir = f"jobs/{job_id}"
    logger.info(f"Cleaning up files for job {job_id}, keeping only status.json")

    for item in os.listdir(job_base_dir):
        item_path = os.path.join(job_base_dir, item)

        if item == "status.json":
            continue

        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                logger.info(f"Removed directory: {item}")
            elif os.path.isfile(item_path):
                os.remove(item_path)
                logger.info(f"Removed file: {item}")
        except Exception as e:
            logger.error(f"Error removing item {item_path}: {e}")

    logger.info(f"Cleanup completed for job {job_id}, only status.json remains")
    return True


def main():
    parser = argparse.ArgumentParser(description="Worker for processing video jobs")
    parser.add_argument("--job_id", required=True, help="Job ID to process")
    args = parser.parse_args()

    job_id = args.job_id
    job_dir = f"jobs/{job_id}"

    logger.info(f"Starting worker for job: {job_id}")

    if not os.path.exists(job_dir):
        logger.error(f"Job directory not found: {job_dir}")
        return 1

    pending_file = os.path.join(job_dir, "pending")
    if not os.path.exists(pending_file):
        logger.error(f"Job is not pending: {job_id}")
        return 1

    os.remove(pending_file)

    params_file = os.path.join(job_dir, "job_params.json")
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to read job params file: {e}")
        update_job_status(
            job_id=job_id,
            status=JOB_STATUS_FAILED,
            error_message=f"Failed to read job parameters: {str(e)}"
        )
        return 1

    try:
        update_job_status(job_id=job_id, step=2)

        try:
            download_video_from_s3(params["video_url"], job_id)
            update_job_status(job_id=job_id, step=3)
            logger.info(f"Successfully downloaded video for job {job_id}")
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            update_job_status(
                job_id=job_id,
                status=JOB_STATUS_FAILED,
                error_message=f"Failed to download video: {str(e)}"
            )
            return 1

        logger.info(f"Starting video processing for job {job_id}")

        result = process_job(
            job_id=job_id,
            source_language=params.get("source_language"),
            target_language=params["target_language"],
            tts_provider=params["tts_provider"],
            tts_voice=params["tts_voice"],
            elevenlabs_api_key=params["api_keys"].get("elevenlabs"),
            openai_api_key=params["api_keys"].get("openai"),
            is_premium=params.get("user_is_premium", False)
        )

        if result["status"] == "success":
            logger.info(f"Processing completed successfully for job {job_id}")

            update_job_status(job_id=job_id, step=15)

            try:
                logger.info(f"Uploading results to S3 for job {job_id}")
                result_urls = upload_results_to_s3(job_id, params.get("user_is_premium", False))

                update_job_status(
                    job_id=job_id,
                    step=16,
                    result_urls=result_urls
                )

                finalize_job(job_id)

                logger.info(f"Cleaning up temporary files for job {job_id}")
                cleanup_job_files(job_id)

                logger.info(f"Job {job_id} completed successfully")
            except Exception as e:
                logger.error(f"Error uploading results: {e}")
                update_job_status(
                    job_id=job_id,
                    status=JOB_STATUS_FAILED,
                    error_message=f"Failed to upload results: {str(e)}"
                )
                return 1
        else:
            logger.error(f"Processing failed for job {job_id}: {result.get('message', 'Unknown error')}")
            update_job_status(
                job_id=job_id,
                status=JOB_STATUS_FAILED,
                step=result.get("step"),
                error_message=result.get("message", "Unknown error")
            )

    except Exception as e:
        logger.error(f"Unexpected error processing job {job_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        update_job_status(
            job_id=job_id,
            status=JOB_STATUS_FAILED,
            error_message=f"Unexpected error: {str(e)}"
        )
        return 1

    return 0


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())