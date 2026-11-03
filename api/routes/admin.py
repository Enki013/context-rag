"""
api/routes/admin.py
-------------------
Pipeline gözlemlenebilirliği (yalnızca admin): üretim çalışmalarının özeti ve elenen sorular.
"""

from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Depends, Query

from api.security import require_admin
from src.metrics import list_rejected, summarize_runs

router = APIRouter(prefix="/admin", tags=["Admin"], dependencies=[Depends(require_admin)])


@router.get("/metrics", summary="Üretim pipeline metrikleri (son N çalışma)")
def metrics_endpoint(limit: Annotated[int, Query(ge=1, le=1000)] = 100):
    """Aşama başına eleme oranları, ortalama süreler ve en sık eleme sebepleri."""
    return summarize_runs(limit=limit)


@router.get("/rejected-questions", summary="Elenen sorular (sebep ve metin bağlamıyla)")
def rejected_endpoint(
    stage: Annotated[Optional[Literal["structural", "verify", "dedup"]], Query()] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
):
    """Değerlendirme setine eklenecek adayları incelemek için."""
    return {"items": list_rejected(stage=stage, limit=limit)}
