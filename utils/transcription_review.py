import os
import time

import boto3
from botocore.exceptions import ClientError

from modules.job_status import update_job_status
from utils.logger_config import setup_logger, get_job_logger

logger = setup_logger(name=__name__, log_file="logs/app.log")


def upload_transcription_for_review(job_id: str, transcription_file_path: str) -> str:
    log = get_job_logger(logger, job_id)

    s3_bucket = os.getenv("S3_BUCKET")
    if not s3_bucket:
        log.error("S3_BUCKET environment variable not set")
        raise EnvironmentError("S3_BUCKET environment variable must be set")

    if not os.path.exists(transcription_file_path):
        log.error(f"Transcription file not found: {transcription_file_path}")
        raise FileNotFoundError(f"Transcription file not found: {transcription_file_path}")

    s3_client = boto3.client("s3")

    s3_key = f"jobs/{job_id}/review_required/transcription_original.json"

    try:
        log.info(f"Uploading transcription for review to S3: {s3_key}")
        s3_client.upload_file(transcription_file_path, s3_bucket, s3_key)
        s3_url = f"s3://{s3_bucket}/{s3_key}"
        log.info(f"Successfully uploaded transcription for review: {s3_url}")
        return s3_url
    except ClientError as e:
        log.error(f"Error uploading transcription to S3: {e}")
        raise


def check_s3_file_exists(s3_bucket: str, s3_key: str, job_id=None) -> bool:
    if job_id:
        log = get_job_logger(logger, job_id)
    else:
        log = logger

    s3_client = boto3.client("s3")
    try:
        s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            return False
        else:
            log.error(f"Error checking S3 file existence: {e}")
            return False


def download_corrected_transcription(job_id: str) -> str:
    job_logger = get_job_logger(logger, job_id)

    s3_bucket = os.getenv("S3_BUCKET")
    s3_key = f"jobs/{job_id}/review_required/transcription_corrected.json"

    local_dir = f"jobs/{job_id}/review"
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, "transcription_corrected.json")

    s3_client = boto3.client("s3")
    try:
        job_logger.info(f"Downloading corrected transcription from S3: {s3_key}")
        s3_client.download_file(s3_bucket, s3_key, local_path)
        job_logger.info(f"Successfully downloaded corrected transcription: {local_path}")
        return local_path
    except ClientError as e:
        job_logger.error(f"Error downloading corrected transcription: {e}")
        raise


def get_review_result(job_id: str, original_translated_transcription: str) -> str:
    log = get_job_logger(logger, job_id)

    log.info(f"{'=' * 50}")
    log.info(f"Starting review process for job {job_id}")
    log.info(f"Original transcription: {original_translated_transcription}")

    current_step = 11

    try:
        review_required_s3_url = upload_transcription_for_review(job_id, original_translated_transcription)
        log.info(f"Transcription uploaded for review: {review_required_s3_url}")

        update_job_status(
            job_id=job_id,
            step=current_step,
            review_required_url=review_required_s3_url
        )
        log.info(f"Step updated to review ({current_step}) for job {job_id}")

        s3_bucket = os.getenv("S3_BUCKET")
        corrected_s3_key = f"jobs/{job_id}/review_required/transcription_corrected.json"

        max_checks = 90
        check_interval = 10

        log.info(f"Waiting for review... (max {max_checks * check_interval // 60} minutes)")

        for check_num in range(1, max_checks + 1):
            time.sleep(check_interval)

            if check_s3_file_exists(s3_bucket, corrected_s3_key, job_id=job_id):
                log.info(f"Found corrected transcription after {check_num * check_interval} seconds!")

                corrected_transcription_path = download_corrected_transcription(job_id)

                log.info(f"Replacing original file with corrected transcription...")
                import shutil
                shutil.copy2(corrected_transcription_path, original_translated_transcription)

                log.info(f"Original transcription file updated with corrections")
                log.info(f"Continuing with updated file: {original_translated_transcription}")
                return original_translated_transcription

            if check_num % 6 == 0:
                minutes_elapsed = check_num * check_interval // 60
                log.info(f"Still waiting... {minutes_elapsed} minutes elapsed")

        log.info(f"Review timeout reached (15 minutes). Keeping original transcription.")

        log.info(f"Continuing with original transcription: {original_translated_transcription}")
        return original_translated_transcription

    except Exception as e:
        log.error(f"Error in review process: {e}")
        log.info(f"Falling back to original transcription due to error")

        return original_translated_transcription
