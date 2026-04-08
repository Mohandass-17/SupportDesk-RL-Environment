"""
SupportDesk RL — Environment Client
Connects to the running server via WebSocket (openenv-core) or HTTP.
"""
from typing import Optional
from .models import SupportAction, SupportObservation, SupportState

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult

    class SupportDeskEnv(EnvClient[SupportAction, SupportObservation, SupportState]):
        """
        WebSocket client for the SupportDesk RL environment.

        Usage (async):
            async with SupportDeskEnv(base_url="https://your-space.hf.space") as env:
                result = await env.reset(task_name="ticket_classification")
                result = await env.step(SupportAction(action_type="classify", value="billing_issue"))

        Usage (sync):
            with SupportDeskEnv(base_url="https://your-space.hf.space").sync() as env:
                result = env.reset(task_name="ticket_classification")
                result = env.step(SupportAction(action_type="classify", value="billing_issue"))
        """

        def _step_payload(self, action: SupportAction) -> dict:
            return action.model_dump(exclude_none=True)

        def _parse_result(self, payload: dict) -> StepResult:
            obs_data = payload.get("observation", {})
            obs = SupportObservation(**obs_data)
            return StepResult(
                observation=obs,
                reward=payload.get("reward", 0.0),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: dict) -> SupportState:
            return SupportState(**payload)

except ImportError:
    # Fallback stub when openenv-core is not installed
    class SupportDeskEnv:  # type: ignore[no-redef]
        """Lightweight HTTP client — works without openenv-core installed."""

        def __init__(self, base_url: str = "http://localhost:7860"):
            self.base_url = base_url.rstrip("/")
            self._session = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def sync(self):
            return self

        def reset(self, task_name: str = "ticket_classification"):
            import urllib.request, json
            url = f"{self.base_url}/reset?task_name={task_name}"
            with urllib.request.urlopen(url) as resp:
                data = json.loads(resp.read())
            obs = SupportObservation(**data["observation"])
            from types import SimpleNamespace
            return SimpleNamespace(observation=obs, reward=0.0, done=False)

        def step(self, action: SupportAction):
            import urllib.request, json
            payload = json.dumps({"action": action.model_dump(exclude_none=True)}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/step",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
            obs = SupportObservation(**data["observation"])
            from types import SimpleNamespace
            return SimpleNamespace(
                observation=obs,
                reward=data.get("reward", 0.0),
                done=data.get("done", False),
            )
