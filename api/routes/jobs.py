"""
api/routes/jobs.py
------------------
Asenkron üretim işlerinin durumunu sorgulama.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import JobResponse
from src.jobs import get_job, queue_position

router = APIRouter(prefix="/jobs", tags=["Jobs"])

RETRY_AFTER_SECONDS = 15


def job_payload(job: dict) -> dict:
    """İş satırını API yanıtına çevirir."""
    return {
        **{k: job.get(k) for k in ("id", "kind", "status", "params", "result", "error",
                                   "created_at", "started_at", "finished_at")},
        "queue_position": queue_position(job["id"]) if job.get("status") == "queued" else None,
        "poll_url": f"/api/v1/jobs/{job['id']}",
    }


@router.get("/{job_id}", response_model=JobResponse, summary="Üretim işinin durumu")
def get_job_endpoint(job_id: str):
    """queued → running → completed | failed. Tamamlanınca `result` alanı dolar."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"İş bulunamadı: {job_id}")
    return job_payload(job)
