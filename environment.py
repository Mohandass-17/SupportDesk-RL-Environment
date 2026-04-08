"""
SupportDeskEnv — OpenEnv-compatible RL environment.

Implements the standard OpenEnv interface:
  reset()  → initial observation
  step()   → (observation, reward, done, info)
  state()  → current raw state dict
"""

import json
import random
from pathlib import Path
from typing import Optional

from env.models import Observation, Ticket
from env.rewards import compute_reward


KNOWLEDGE_BASE = [
    "Password resets are handled via the 'Forgot Password' link on the login page.",
    "Billing disputes and duplicate charges must be escalated to the Finance team.",
    "All security incidents (hacked accounts, unauthorized access) require immediate escalation.",
    "How-to and general usage questions can be resolved directly with a helpful response.",
    "Server errors (5xx) affecting purchases or core workflows must be escalated to Engineering.",
    "Service outages with business impact must be escalated to the Incident Response team.",
    "Account and subscription changes can be self-served; provide step-by-step instructions.",
    "Technical bugs should be classified and routed to the QA/Engineering queue.",
]


class SupportDeskEnv:

    VALID_ACTIONS = {"classify_ticket", "respond_ticket", "escalate_ticket"}

    def __init__(self, seed: Optional[int] = None):
        data_path = Path(__file__).parent.parent / "datasets" / "tickets.json"
        with open(data_path) as f:
            raw = json.load(f)
        self.dataset = [Ticket(**t) for t in raw]

        self.rng = random.Random(seed)
        self._ticket_queue: list[Ticket] = []
        self.current_ticket: Optional[Ticket] = None
        self.step_count = 0
        self.episode = 0
        self.done = False
        self.episode_rewards: list[float] = []

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Start a new episode. Shuffles ticket queue each episode."""
        self._ticket_queue = self.rng.sample(self.dataset, len(self.dataset))
        self.current_ticket = self._ticket_queue.pop(0)
        self.step_count = 0
        self.episode += 1
        self.done = False
        self.episode_rewards = []
        return self.state()

    def step(self, action: str, response: Optional[str] = None) -> tuple:
        """
        Process one agent action.

        Args:
            action:   One of 'classify_ticket', 'respond_ticket', 'escalate_ticket'.
            response: Optional free-text response (used for respond_ticket scoring).

        Returns:
            (observation, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        action = action.strip().lower()
        if action not in self.VALID_ACTIONS:
            # Invalid action → small penalty, no progress
            reward = 0.0
            info = {"error": f"Invalid action '{action}'. Must be one of {self.VALID_ACTIONS}."}
            return self.state(), reward, False, info

        correct_action = self.current_ticket.correct_action
        reward = compute_reward(action, correct_action, self.step_count, response)
        self.episode_rewards.append(reward)
        self.step_count += 1

        info = {
            "correct_action": correct_action,
            "was_correct": action == correct_action,
            "ticket_id": self.current_ticket.id,
        }

        # Advance to next ticket on correct action, or end episode if queue empty
        if action == correct_action:
            if self._ticket_queue:
                self.current_ticket = self._ticket_queue.pop(0)
            else:
                self.done = True

        # Safety: cap episode length
        if self.step_count >= 20:
            self.done = True

        return self.state(), reward, self.done, info

    def state(self) -> dict:
        """Return the current environment state as a plain dict."""
        obs = Observation(
            ticket=self.current_ticket,
            step=self.step_count,
            episode=self.episode,
            knowledge_base=KNOWLEDGE_BASE,
        )
        return obs.model_dump()
