"""Academic Copilot — FastAPI 服务入口。"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import socket
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.interfaces.api.routes import admin, chat, health
from src.interfaces.api.service import reload_runtime_config, warn_timeout_misconfiguration_once
from src.infrastructure.tools.loader import initialize_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
_DEFAULT_EXECUTOR: ThreadPoolExecutor | None = None


def _tools_strict_startup() -> bool:
    return os.getenv("TOOLS_STRICT_STARTUP", "true").strip().lower() in {"1", "true", "yes", "on"}


def _chat_max_workers() -> int:
    raw = os.getenv("CHAT_MAX_WORKERS", "4").strip()
    try:
        workers = int(raw)
    except ValueError:
        workers = 4
    return max(1, workers)


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    global _DEFAULT_EXECUTOR
    logger.info("Academic Copilot starting — initializing tools...")
    loop = asyncio.get_running_loop()
    _DEFAULT_EXECUTOR = ThreadPoolExecutor(
        max_workers=_chat_max_workers(),
        thread_name_prefix="copilot-worker",
    )
    loop.set_default_executor(_DEFAULT_EXECUTOR)
    logger.info("Configured default chat executor max_workers=%d", _chat_max_workers())
    try:
        try:
            warn_timeout_misconfiguration_once()
            tools_report = await initialize_tools()
            runtime_report = reload_runtime_config()
            runtime_failed = runtime_report.get("failed", [])
            tools_failed = tools_report.get("failed", [])
            if runtime_failed:
                raise RuntimeError(f"Runtime config validation failed with {len(runtime_failed)} issue(s)")
            if tools_failed and _tools_strict_startup():
                raise RuntimeError(f"Tool initialization failed with {len(tools_failed)} issue(s)")
            logger.info(
                "Tools loaded: %d tools, %d servers",
                len(tools_report.get("loaded_tools", [])),
                len(tools_report.get("loaded_servers", [])),
            )
            if tools_failed:
                logger.warning("Tools loaded with failures: %d", len(tools_failed))
            logger.info(
                "Runtime config loaded: %d agents, %d workflows",
                len(runtime_report["loaded"]["agents"]),
                len(runtime_report["loaded"]["workflows"]),
            )
        except Exception as e:
            logger.error("Startup validation failed: %s", e)
            raise
        yield
    finally:
        if _DEFAULT_EXECUTOR is not None:
            _DEFAULT_EXECUTOR.shutdown(wait=False, cancel_futures=True)
            _DEFAULT_EXECUTOR = None
        logger.info("Academic Copilot shutdown.")


app = FastAPI(
    title="Academic Copilot",
    description="Multi-agent Academic Research Assistant",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def request_observability_middleware(request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    started = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "http.request.error",
            extra={
                "event": "http.request.error",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        raise
    duration_ms = round((perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "http.request.complete",
        extra={
            "event": "http.request.complete",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response

app.include_router(chat.router)
app.include_router(health.router)
app.include_router(admin.router)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.exception_handler(HTTPException)
async def http_exc(request, exc):
    return JSONResponse(status_code=exc.status_code,
                        content={"success": False, "message": exc.detail})


@app.exception_handler(Exception)
async def general_exc(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500,
                        content={"success": False, "message": "Internal server error."})


if __name__ == "__main__":
    import uvicorn
    ip = socket.gethostbyname(socket.gethostname())
    logger.info(f"Serving at http://{ip}:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
