"""Academic Copilot — FastAPI 服务入口。"""
import logging
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import chat, research, health
from src.tools.mcp_loader import initialize_mcp_tools
from src.tools.registry import _init_role_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Academic Copilot starting — initializing MCP tools...")
    try:
        await initialize_mcp_tools()
        _init_role_tools()
    except Exception as e:
        logger.warning(f"MCP init failed (non-fatal): {e}")
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
app.include_router(research.router)
app.include_router(health.router)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        return HTMLResponse(open("static/index.html", encoding="utf-8").read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend not found.")


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
