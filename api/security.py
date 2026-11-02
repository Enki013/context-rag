"""
api/security.py
---------------
Admin uçları için basit anahtar kontrolü.

Mobil istemciler bu anahtarı taşımaz; yalnızca backend/cron/admin araçları
`X-Admin-Key` header'ı ile gönderir.
"""

import secrets
from typing import Optional

from fastapi import Header, HTTPException, status

import config


def is_admin(x_admin_key: Optional[str]) -> bool:
    expected = config.ADMIN_API_KEY
    if not expected or not x_admin_key:
        return False
    return secrets.compare_digest(x_admin_key, expected)


def require_admin(x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")):
    """Admin anahtarı yoksa veya yanlışsa isteği reddeder."""
    if not config.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin uçları kapalı: sunucuda ADMIN_API_KEY ortam değişkeni tanımlı değil.",
        )
    if not is_admin(x_admin_key):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Geçersiz veya eksik X-Admin-Key.")
