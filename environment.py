import json
import random
from pathlib import Path
from models import Observation, Ticket
from rewards import compute_reward

KNOWLEDGE_BASE = [
    "Password resets are handled via the Forgot Password link on the login page.",
    "Billing disputes and duplicate charges must be escalated to the Finance team.",
    "All security incidents require immediate escalation.",
    "How-to questions can be resolved directly with a helpful response.",
    "Server errors affecting purchases must be escalated to Engineering.",
    "Service outages with business impact must be escalated to Incident Response.",
    "Account and subscription changes can be self-served.",
    "Technical bugs should be classified and routed to QA/Engineering queue.",
]

class SupportDeskEnv:

    VALID_ACTIONS = {"classify_ticket", "respond_ticket", "escalate_ticket"}

    def __init__(self, seed=None):
        data_path = Path(__file__).parent / "tickets.json"
        with open(data_path) as f:
            raw = json.load(f)
        self.dataset = [Ticket(**t) for t in raw]
        self.rng = random.Random(seed)
        self._ticket_queue = []
        self.current_ticket = None
        self.step_count = 0
        self.episode = 0
        self.done = False

    def reset(self):
        self._ticket_queue = self.rng.sample(self.dataset, len(self.dataset))
        self.current_ticket = self._ticket_queue.pop(0)
        self.step_count = 0
        self.episode += 1
        self.done = False
        return self.state()

    def step(self, action, response=None):
        if self.done:
            self.reset()

        action = action.strip().lower()
        if action not in self.VALID_ACTIONS:
            info = {"error": "Invalid action", "was_correct": False,
                    "ticket_id": self.current_ticket.id,
                    "correct_action": self.current_ticket.correct_action}
            return self.state(), 0.0, False, info

        correct_action = self.current_ticket.correct_action
        reward = compute_reward(action, correct_action, self.step_count, response)
        self.step_count += 1

        info = {
            "correct_action": correct_action,
            "was_correct": action == correct_action,
            "ticket_id": self.current_ticket.id,
        }

        if action == correct_action:
            if self._ticket_queue:
                self.current_ticket = self._ticket_queue.pop(0)
            else:
                self.done = True

        if self.step_count >= 20:
            self.done = True

        return self.state(), reward, self.done, info

    def state(self):
        obs = Observation(
            ticket=self.current_ticket,
            step=self.step_count,
            episode=self.episode,
            knowledge_base=KNOWLEDGE_BASE,
        )
        return obs.model_dump()
