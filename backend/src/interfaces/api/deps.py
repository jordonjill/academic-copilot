"""共享依赖：认证、安全。"""
import os
from fastapi import HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends

security = HTTPBearer()


def _expected_access_key() -> str:
    return os.getenv("ACCESS_KEY", "123")


def _expected_admin_access_key() -> str | None:
    key = os.getenv("ADMIN_ACCESS_KEY", "").strip()
    return key or None


async def verify_access_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    if credentials.credentials != _expected_access_key():
        raise HTTPException(
            status_code=401,
            detail="Invalid access key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


async def verify_admin_access_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    expected = _expected_admin_access_key()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_ACCESS_KEY is not configured",
        )
    if credentials.credentials != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid admin access key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
