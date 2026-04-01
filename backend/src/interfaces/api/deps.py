"""共享依赖：认证、安全。"""
import os
from fastapi import HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends

security = HTTPBearer()


def _expected_access_key() -> str:
    return os.getenv("ACCESS_KEY", "123")


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
