"""MinerU v4 API client: upload PDF → auto-extract → get images + markdown.

Uses Bearer token auth. Flow:
    1. POST /api/v4/file-urls/batch  → get signed upload URL + batch_id
    2. PUT file to signed URL
    3. System auto-submits extraction task
    4. GET /api/v4/extract-results/batch/{batch_id}  → poll for results

Test scenario:
    >>> from pathlib import Path
    >>> from .client import upload_and_extract
    >>> result = upload_and_extract(Path("tests/fixtures/ph01/gxae11701.pdf"))
    >>> assert result["state"] == "done"
"""

import os
import time
from pathlib import Path

import requests


BASE_URL = "https://mineru.net/api/v4"
DEFAULT_TIMEOUT = 30


def _get_token() -> str:
    key = os.environ.get("MINERU_API_KEY", "")
    if not key:
        key = os.environ.get("MINERU_API_TOKEN", "")
    if not key:
        raise RuntimeError("MINERU_API_KEY or MINERU_API_TOKEN not set")
    return key


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_token()}",
        "Content-Type": "application/json",
    }


def upload_and_extract(
    pdf_path: Path,
    is_ocr: bool = False,
    enable_formula: bool = True,
    enable_table: bool = True,
    language: str = "en",
    poll_interval: int = 5,
    max_wait: int = 300,
) -> dict:
    """Upload a local PDF to MinerU v4 and wait for extraction results.

    Args:
        pdf_path: Local PDF file path.
        is_ocr: Enable OCR for scanned documents.
        enable_formula: Enable LaTeX formula recognition.
        enable_table: Enable table extraction.
        language: Document language ('en', 'ch', etc.).
        poll_interval: Seconds between poll requests.
        max_wait: Maximum seconds to wait for completion.

    Returns:
        dict with keys: state, full_zip_url, etc. On success, state == "done".

    Raises:
        RuntimeError: Upload or extraction failed.
        TimeoutError: Extraction did not complete within max_wait.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Step 1: Get signed upload URL
    print(f"[MinerU] Requesting upload URL for {pdf_path.name} ...")
    upload_resp = requests.post(
        f"{BASE_URL}/file-urls/batch",
        headers=_headers(),
        json={
            "files": [
                {
                    "name": pdf_path.name,
                    "data_id": pdf_path.stem,
                }
            ],
            "is_ocr": is_ocr,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
            "language": language,
        },
        timeout=DEFAULT_TIMEOUT,
    )
    upload_resp.raise_for_status()
    upload_data = upload_resp.json()

    if upload_data.get("code") != 0:
        raise RuntimeError(f"Upload URL request failed: {upload_data}")

    batch_id = upload_data["data"]["batch_id"]
    file_urls = upload_data["data"]["file_urls"]
    if not file_urls:
        raise RuntimeError("No upload URLs returned")

    signed_url = file_urls[0]
    print(f"[MinerU] batch_id={batch_id}, uploading ...")

    # Step 2: PUT file to signed URL
    file_bytes = pdf_path.read_bytes()
    put_resp = requests.put(
        signed_url,
        data=file_bytes,
        timeout=DEFAULT_TIMEOUT * 2,
    )
    put_resp.raise_for_status()
    print(f"[MinerU] Upload complete ({len(file_bytes)} bytes). Waiting for extraction ...")

    # Step 3: Poll for results via batch endpoint
    elapsed = 0
    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval

        result_resp = requests.get(
            f"{BASE_URL}/extract-results/batch/{batch_id}",
            headers=_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
        result_resp.raise_for_status()
        result_data = result_resp.json()

        if result_data.get("code") != 0:
            raise RuntimeError(f"Result query failed: {result_data}")

        batch = result_data.get("data", {})
        results = batch.get("extract_result", [])
        if not results:
            state = "unknown"
        else:
            state = results[0].get("state", "unknown")
        print(f"[MinerU] state={state} ({elapsed}s)")

        if state == "done":
            return results[0] if results else batch
        if state in ("failed", "error"):
            err = results[0].get("err_msg", "unknown") if results else "unknown"
            raise RuntimeError(f"Extraction failed: {err}")

    raise TimeoutError(f"Extraction did not complete within {max_wait}s")
