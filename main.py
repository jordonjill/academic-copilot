import logging
import socket
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.research_agent_app import create_research_agent, ResearchAgentApp

ACCESS_KEY = "123"

security = HTTPBearer()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Proposal Agent Web Application",
    description="Intelligent Research Proposal Agent Web App",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    topic: str
    model_type: str = "ollama"

class ResearchResponse(BaseModel):
    success: bool
    message: str
    session_id: Optional[str] = None

async def verify_access_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if credentials.credentials != ACCESS_KEY:
        logger.warning(f"Invalid Access key attempt: {credentials.credentials[:8]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid Access key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return credentials.credentials

class SessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        self.research_agents: Dict[str, ResearchAgentApp] = {}
    
    def create_session(self, topic: str, model_type: str = "ollama") -> str:
        session_id = str(uuid.uuid4())
        try:
            agent = create_research_agent(model_type)
            self.research_agents[session_id] = agent
            self.active_sessions[session_id] = {
                "topic": topic,
                "model_type": model_type,
                "status": "created",
                "created_at": datetime.now().isoformat(),
                "websocket": None
            }
            logger.info(f"Created session {session_id} for topic: {topic}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create research session: {str(e)}")
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        return self.active_sessions.get(session_id)
    
    def get_agent(self, session_id: str) -> Optional[ResearchAgentApp]:
        return self.research_agents.get(session_id)
    
    def set_websocket(self, session_id: str, websocket: WebSocket):
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["websocket"] = websocket
            self.active_sessions[session_id]["status"] = "connected"
    
    def remove_session(self, session_id: str):
        self.active_sessions.pop(session_id, None)
        self.research_agents.pop(session_id, None)
        logger.info(f"Removed session {session_id}")

session_manager = SessionManager()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()

        return HTMLResponse(content=html_content)
    
    except FileNotFoundError:
        logger.error("Frontend HTML file not found")

        raise HTTPException(
            status_code=404,
            detail="Frontend page not found"
        )

@app.post("/research/start", response_model=ResearchResponse)
async def start_research(
    request: ResearchRequest,
    access_key: str = Depends(verify_access_key)
):
    """Start a new research session"""
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Research topic cannot be empty")
    
    if request.model_type not in ["ollama", "gemini", "openai"]:
        raise HTTPException(status_code=400, detail="Unsupported model type")
    
    try:
        session_id = session_manager.create_session(request.topic, request.model_type)
        return ResearchResponse(
            success=True,
            message="Research session created successfully, connect via WebSocket to start",
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Error starting research: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start research: {str(e)}")

@app.get("/research/status/{session_id}")
async def get_research_status(
    session_id: str,
    access_key: str = Depends(verify_access_key)
):
    """Get research session status"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    agent = session_manager.get_agent(session_id)
    current_state = agent.get_current_state() if agent else None
    
    return {
        "session_id": session_id,
        "topic": session["topic"],
        "model_type": session["model_type"],
        "status": session["status"],
        "created_at": session["created_at"],
        "current_state": current_state
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication"""
    query_params = dict(websocket.query_params)
    access_key = query_params.get("access_key")

    if not access_key or access_key != ACCESS_KEY:
        await websocket.close(code=1008, reason="Invalid or missing access key")
        logger.warning(f"WebSocket connection rejected: Invalid access key")
        return
    
    await websocket.accept()
    
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.send_json({
            "type": "error",
            "message": "Session not found",
            "timestamp": datetime.now().isoformat()
        })
        await websocket.close()
        return
    
    session_manager.set_websocket(session_id, websocket)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": f"Connected to session {session_id}",
            "topic": session["topic"],
            "timestamp": datetime.now().isoformat()
        })
        
        agent = session_manager.get_agent(session_id)
        if not agent:
            await websocket.send_json({
                "type": "error",
                "message": "Research agent not initialized",
                "timestamp": datetime.now().isoformat()
            })
            return
        
        async def websocket_send(message: Dict):
            try:
                await websocket.send_json(message)
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")
                raise e
        
        session["status"] = "running"
        
        try:
            await agent.run_research_async(
                initial_topic=session["topic"],
                websocket_send=websocket_send,
                recursion_limit=15
            )
            session["status"] = "completed"
        except Exception as e:
            logger.error(f"Research execution error: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Research execution failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
            session["status"] = "failed"
        
        while True:
            try:
                message = await websocket.receive_json()
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                elif message.get("type") == "disconnect":
                    break
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                break
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        session_manager.remove_session(session_id)

@app.get("/sessions")
async def list_sessions(api_key: str = Depends(verify_access_key)):
    """List all active sessions (for debugging)"""
    return {
        session_id: {
            "topic": data["topic"],
            "model_type": data["model_type"],
            "status": data["status"],
            "created_at": data["created_at"]
        }
        for session_id, data in session_manager.active_sessions.items()
    }

@app.get("/health")
async def health_check(model_type: str = "ollama"):
    """Health check endpoint - Tests LLM connectivity for specified model type"""
    if model_type not in ["ollama", "gemini", "openai"]:
        return {
            "healthy": False,
            "status": f"Unsupported model type: {model_type}",
            "timestamp": datetime.now().isoformat(),
            "active_sessions": len(session_manager.active_sessions),
            "model_type": model_type
        }
    
    try:
        test_agent = create_research_agent(model_type)
        health_result = test_agent.health_check()
        
        is_healthy = health_result["status"] == "healthy"
        
        return {
            "healthy": is_healthy,
            "status": health_result["message"],
            "timestamp": datetime.now().isoformat(),
            "active_sessions": len(session_manager.active_sessions),
            "model_type": model_type
        }
        
    except Exception as e:
        logger.error(f"Health check failed for {model_type}: {e}")
        return {
            "healthy": False,
            "status": f"Service unavailable: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "active_sessions": len(session_manager.active_sessions),
            "model_type": model_type
        }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    logger.info(f"Server will be accessible at:")
    logger.info(f"  - Network: http://{local_ip}:8000")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )