"""共享依赖：认证、安全。"""
import os
from fastapi import HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends

ACCESS_KEY: str = os.getenv("ACCESS_KEY", "123")
security = HTTPBearer()


async def verify_access_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    if credentials.credentials != ACCESS_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid access key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
