"""
SupportDesk RL — FastAPI Server
Exposes the environment over HTTP + WebSocket so OpenEnv clients can connect.
"""
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import json

from ..server.environment import SupportDeskEnvironment
from ..models import SupportAction, SupportObservation, SupportState

# Try to use openenv-core if available; fall back to plain FastAPI
try:
    from openenv.core.env_server import HTTPEnvServer

    _env_class = SupportDeskEnvironment
    _server = HTTPEnvServer(
        env=_env_class,
        action_cls=SupportAction,
        observation_cls=SupportObservation,
        max_concurrent_envs=4,
    )
    app = FastAPI(
        title="SupportDesk RL Environment",
        description="OpenEnv customer support operations environment for RL training.",
        version="0.1.0",
    )
    _server.register_routes(app)

except Exception:
    # -----------------------------------------------------------------------
    # Fallback: Plain FastAPI with manual WebSocket + REST endpoints
    # This ensures the environment runs even without full openenv-core install.
    # -----------------------------------------------------------------------
    app = FastAPI(
        title="SupportDesk RL Environment",
        description="OpenEnv customer support operations environment for RL training.",
        version="0.1.0",
    )

    # Shared environment pool (one per WebSocket session)
    _sessions: dict = {}

    @app.get("/health")
    async def health():
        return {"status": "ok", "environment": "supportdesk_env", "version": "0.1.0"}

    @app.post("/reset")
    async def reset_http(body: dict = None):
        env = SupportDeskEnvironment()
        task = (body or {}).get("task_name", "ticket_classification")
        obs = env.reset(task_name=task)
        session_id = "default"
        _sessions[session_id] = env
        return {
            "observation": obs.model_dump(),
            "reward": 0.0,
            "done": False,
        }

    @app.get("/reset")
    async def reset_get(task_name: str = "ticket_classification"):
        """GET /reset — used by the OpenEnv validator to confirm environment is live."""
        env = SupportDeskEnvironment()
        obs = env.reset(task_name=task_name)
        _sessions["default"] = env
        return {
            "observation": obs.model_dump(),
            "reward": 0.0,
            "done": False,
        }

    @app.post("/step")
    async def step_http(body: dict):
        session_id = body.get("session_id", "default")
        env = _sessions.get(session_id)
        if env is None:
            return JSONResponse(status_code=400, content={"error": "No active session. Call /reset first."})
        action = SupportAction(**body.get("action", body))
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": round(reward, 4),
            "done": done,
            "info": info,
        }

    @app.get("/state")
    async def state_http(session_id: str = "default"):
        env = _sessions.get(session_id)
        if env is None:
            return JSONResponse(status_code=400, content={"error": "No active session."})
        return env.state.model_dump()

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        """WebSocket endpoint — primary interface for openenv EnvClient."""
        await ws.accept()
        env = SupportDeskEnvironment()
        session_id = id(ws)
        _sessions[session_id] = env

        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                method = msg.get("method", "")

                if method == "reset":
                    task = msg.get("task_name", "ticket_classification")
                    obs = env.reset(task_name=task)
                    await ws.send_text(json.dumps({
                        "observation": obs.model_dump(),
                        "reward": 0.0,
                        "done": False,
                    }))

                elif method == "step":
                    action_data = msg.get("action", {})
                    action = SupportAction(**action_data)
                    obs, reward, done, info = env.step(action)
                    await ws.send_text(json.dumps({
                        "observation": obs.model_dump(),
                        "reward": round(reward, 4),
                        "done": done,
                        "info": info,
                    }))

                elif method == "state":
                    await ws.send_text(json.dumps(env.state.model_dump()))

                else:
                    await ws.send_text(json.dumps({"error": f"Unknown method: {method}"}))

        except WebSocketDisconnect:
            pass
        finally:
            _sessions.pop(session_id, None)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("supportdesk_env.server.app:app", host="0.0.0.0", port=port, reload=False)
