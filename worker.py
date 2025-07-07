import argparse
import json
import os
import sys
import shutil
import boto3
from botocore.exceptions import ClientError
from dubbing_job_processing import process_job
from modules.job_status import update_job_status, finalize_job, JOB_STATUS_FAILED
from dotenv import load_dotenv
from utils.logger_config import setup_logger, get_job_logger

logger = setup_logger(name=__name__, log_file="logs/app.log")


def download_video_from_s3(video_url: str, job_id: str) -> str:
    job_logger = get_job_logger(logger, job_id)

    if not video_url.startswith("s3://"):
        raise ValueError(f"Invalid S3 URL format: {video_url}")

    parts = video_url[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URL: {video_url}")

    bucket = parts[0]
    key = parts[1]

    video_input_dir = f"jobs/{job_id}/video_input"
    os.makedirs(video_input_dir, exist_ok=True)

    filename = os.path.basename(key)
    local_path = os.path.join(video_input_dir, filename)

    s3_client = boto3.client("s3")
    try:
        s3_client.download_file(bucket, key, local_path)
        job_logger.info("Successfully downloaded video from cloud storage")
        return local_path
    except ClientError as e:
        job_logger.error(f"Error downloading video-file from cloud storage: {e}")
        raise


def upload_results_to_s3(job_id: str) -> dict:
    job_logger = get_job_logger(logger, job_id)

    job_result_dir = f"jobs/{job_id}/job_result"
    job_result_file_path = os.path.join(job_result_dir, "pipeline_result.json")

    if not os.path.exists(job_result_file_path):
        job_logger.error(f"Job result file not found: {job_result_file_path}")
        raise FileNotFoundError("Job result not found")

    with open(job_result_file_path, "r") as f:
        job_result = json.load(f)

    s3_bucket = os.getenv("S3_BUCKET")
    if not s3_bucket:
        job_logger.error("S3_BUCKET environment variable not set")
        raise EnvironmentError("S3_BUCKET environment variable must be set")

    s3_client = boto3.client("s3")
    result_urls = {}

    files_to_upload = {
        "translated": job_result["output_files"]["translated"],
        "audio": job_result["output_files"]["final_audio"],
        "audio_stereo": job_result["output_files"]["final_audio_stereo"],
        "final_video": job_result["output_files"]["final_video"],
        "final_video_tts_based": job_result["output_files"]["final_video_tts_based"]
    }

    bg_files = {
        "final_audio_with_bg": job_result["output_files"].get("final_audio_with_bg"),
        "final_audio_stereo_with_bg": job_result["output_files"].get("final_audio_stereo_with_bg"),
        "final_video_with_bg": job_result["output_files"].get("final_video_with_bg")
    }

    for key, path in bg_files.items():
        if path:
            files_to_upload[key] = path

    for file_type, local_path in files_to_upload.items():
        if local_path and os.path.exists(local_path):
            filename = os.path.basename(local_path)
            s3_key = f"jobs/{job_id}/results/{filename}"

            try:
                job_logger.info(f"Uploading {file_type} to S3: {s3_key}")
                s3_client.upload_file(local_path, s3_bucket, s3_key)
                result_urls[file_type] = f"s3://{s3_bucket}/{s3_key}"
                job_logger.info(f"Successfully uploaded {file_type} to {result_urls[file_type]}")
            except ClientError as e:
                job_logger.error(f"Error uploading {file_type} to S3: {e}")
                continue
        else:
            job_logger.warning(f"File {file_type} not found or path is None: {local_path}")

    return result_urls


def cleanup_job_files(job_id):
    job_logger = get_job_logger(logger, job_id)

    job_base_dir = f"jobs/{job_id}"

    for item in os.listdir(job_base_dir):
        item_path = os.path.join(job_base_dir, item)

        if item == "status.json":
            continue

        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                job_logger.info(f"Removed directory: {item}")
            elif os.path.isfile(item_path):
                os.remove(item_path)
                job_logger.info(f"Removed file: {item}")
        except Exception as e:
            job_logger.error(f"Error removing item {item_path}: {e}")

    job_logger.info("Job files cleaned up successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Worker for video processing job")
    parser.add_argument("--job_id", required=True, help="Job ID to process")
    args = parser.parse_args()

    load_dotenv()

    job_id = args.job_id
    job_dir = f"jobs/{job_id}"

    job_logger = get_job_logger(logger, job_id)

    job_logger.info("Launching Job Worker...")

    if not os.path.exists(job_dir):
        job_logger.error(f"Job directory not found: {job_dir}")
        return 1

    pending_file = os.path.join(job_dir, "pending")
    if not os.path.exists(pending_file):
        job_logger.error("Job's pending file not found")
        return 1

    os.remove(pending_file)

    params_file = os.path.join(job_dir, "job_params.json")
    try:
        with open(params_file, "r") as f:
            params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        job_logger.error(f"Failed to read job params file: {e}")
        update_job_status(
            job_id=job_id,
            status=JOB_STATUS_FAILED,
            error_message=f"Failed to read job parameters: {str(e)}",
        )
        return 1

    try:
        update_job_status(job_id=job_id, step=2)

        try:
            job_logger.info("Downloading original video-file from cloud storage...")
            download_video_from_s3(params["video_url"], job_id)
        except Exception as e:
            job_logger.error(f"Error downloading video: {e}")
            update_job_status(
                job_id=job_id,
                status=JOB_STATUS_FAILED,
                error_message=f"Failed to download video from cloud storage: {str(e)}",
            )
            return 1

        openai_api_key = os.getenv('OPENAI_API_KEY')
        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')

        if not openai_api_key:
            job_logger.error("OpenAI API key not found in environment variables")
            update_job_status(
                job_id=job_id,
                status=JOB_STATUS_FAILED,
                error_message="OpenAI API key not configured",
            )
            return 1

        if params["tts_provider"] == "elevenlabs" and not elevenlabs_api_key:
            job_logger.error("ElevenLabs API key not found in environment variables")
            update_job_status(
                job_id=job_id,
                status=JOB_STATUS_FAILED,
                error_message="ElevenLabs API key not configured",
            )
            return 1

        result = process_job(
            job_id=job_id,
            source_language=params.get("source_language"),
            target_language=params["target_language"],
            tts_provider=params["tts_provider"],
            tts_voice=params["tts_voice"],
            elevenlabs_api_key=elevenlabs_api_key,
            openai_api_key=openai_api_key,
            transcription_keywords=params.get("transcription_keywords"),
            enable_vad=params.get("enable_vad", True),
        )

        if result["status"] == "success":
            job_logger.info("Job processing completed successfully")

            update_job_status(job_id=job_id, step=18)

            try:
                job_logger.info("Uploading dubbing job results to cloud storage...")

                result_urls = upload_results_to_s3(job_id)

                update_job_status(job_id=job_id, step=19, result_urls=result_urls)

                job_logger.info("Finalizing job...")

                finalize_job(job_id)

                job_logger.info("Cleaning up temporary files...")

                cleanup_job_files(job_id)
            except Exception as e:
                if 'job_logger' in locals():
                    job_logger.error(f"Error uploading results: {e}")
                else:
                    logger.error(f"Error uploading results for job {job_id}: {e}")

                update_job_status(
                    job_id=job_id,
                    status=JOB_STATUS_FAILED,
                    error_message=f"Failed to upload results: {str(e)}",
                )
                return 1
        else:
            job_logger.error(
                f"Job processing failed: {result.get('message', 'Unknown error')}"
            )

            update_job_status(
                job_id=job_id,
                status=JOB_STATUS_FAILED,
                step=result.get("step"),
                error_message=result.get("message", "Unknown error"),
            )

    except Exception as e:
        job_logger.error(f"Unexpected error processing job: {str(e)}")
        import traceback

        job_logger.error(traceback.format_exc())

        update_job_status(
            job_id=job_id,
            status=JOB_STATUS_FAILED,
            error_message=f"Unexpected error: {str(e)}",
        )
        return 1

    return 0


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())
