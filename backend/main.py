"""Academic Copilot — FastAPI 服务入口。"""
import logging
import socket
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.interfaces.api.routes import admin, chat, health
from src.interfaces.api.service import reload_runtime_config
from src.infrastructure.tools.loader import initialize_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Academic Copilot starting — initializing tools...")
    try:
        tools_report = await initialize_tools()
        runtime_report = reload_runtime_config()
        failed = runtime_report.get("failed", [])
        if failed:
            raise RuntimeError(f"Runtime config validation failed with {len(failed)} issue(s)")
        logger.info(
            "Tools loaded: %d tools, %d servers",
            len(tools_report.get("loaded_tools", [])),
            len(tools_report.get("loaded_servers", [])),
        )
        logger.info(
            "Runtime config loaded: %d agents, %d workflows",
            len(runtime_report["loaded"]["agents"]),
            len(runtime_report["loaded"]["workflows"]),
        )
    except Exception as e:
        logger.error("Startup validation failed: %s", e)
        raise
    yield
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
