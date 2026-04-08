"""
SupportDesk RL Environment — Core Logic
Three tasks of increasing difficulty simulating a real-world customer support operation.
"""
import uuid
from typing import Optional, Tuple, Dict, Any

from ..models import SupportAction, SupportObservation, SupportState

# ---------------------------------------------------------------------------
# TASK 1 — TICKET CLASSIFICATION (Easy)
# Agent must classify incoming support tickets into the correct category.
# ---------------------------------------------------------------------------
CLASSIFICATION_TICKETS = [
    {
        "id": "T001",
        "content": "I cannot log in to my account. I reset my password but it still says invalid credentials.",
        "expected": "authentication_issue",
        "related": ["account_issue"],
    },
    {
        "id": "T002",
        "content": "My credit card was charged twice for the same order this month. Please refund.",
        "expected": "billing_issue",
        "related": ["payment_issue"],
    },
    {
        "id": "T003",
        "content": "The iOS app keeps crashing every time I try to upload a profile photo on iPhone 15.",
        "expected": "technical_issue",
        "related": ["bug_report"],
    },
    {
        "id": "T004",
        "content": "It would be great if you could add a dark mode to the dashboard.",
        "expected": "feature_request",
        "related": [],
    },
    {
        "id": "T005",
        "content": "What are your support hours and how do I reach a human agent?",
        "expected": "general_inquiry",
        "related": [],
    },
]
CLASSIFICATION_VALID_VALUES = [
    "authentication_issue",
    "billing_issue",
    "technical_issue",
    "feature_request",
    "general_inquiry",
]

# ---------------------------------------------------------------------------
# TASK 2 — TICKET RESOLUTION (Medium)
# Agent must respond to a ticket using the knowledge base.
# ---------------------------------------------------------------------------
KB_ARTICLES = [
    "Password Reset: Go to Settings > Security > Reset Password. Enter your registered email. Check inbox for link (valid 15 minutes). If not received, check spam folder.",
    "Billing Disputes: Navigate to Account > Billing > Transaction History. Select the disputed charge and click 'Dispute'. Our team responds within 2 business days.",
    "App Crashes: Uninstall and reinstall the app. Ensure you are on the latest version (v4.2+). If crashing persists after reinstall, clear device cache and retry.",
    "Data Export: Go to Settings > Privacy > Export My Data. Select date range and format (CSV or JSON). Download link sent to email within 1 hour.",
    "Account Deletion: Go to Settings > Account > Delete Account. Note: deletion is permanent and all data is removed after 30-day grace period.",
]

RESOLUTION_TICKETS = [
    {
        "id": "T010",
        "content": "I forgot my password and the reset link expired. I cannot get into my account.",
        "context": KB_ARTICLES[0],
        "keywords": ["reset", "password", "settings", "security", "email", "link"],
        "expected_topic": "password_reset",
    },
    {
        "id": "T011",
        "content": "I see a charge of $49.99 I don't recognize. How do I dispute this?",
        "context": KB_ARTICLES[1],
        "keywords": ["billing", "dispute", "transaction", "account", "charge"],
        "expected_topic": "billing_dispute",
    },
    {
        "id": "T012",
        "content": "I want to download all my data before I cancel my subscription.",
        "context": KB_ARTICLES[3],
        "keywords": ["export", "data", "settings", "privacy", "download", "email"],
        "expected_topic": "data_export",
    },
]

# ---------------------------------------------------------------------------
# TASK 3 — INCIDENT ESCALATION (Hard)
# Agent must assess incidents, decide escalation, and assign priority.
# Requires multi-dimensional reasoning (severity + business impact + priority).
# ---------------------------------------------------------------------------
ESCALATION_INCIDENTS = [
    {
        "id": "INC001",
        "content": (
            "Payment service is returning 503 errors for 100% of checkout requests. "
            "Affecting all users globally. Revenue impact: ~$15,000/minute. "
            "On-call engineer has been paged but no response in 10 minutes."
        ),
        "expected_action": "escalate",
        "expected_priority": "P0",
        "notes": "Total service outage with high revenue impact requires immediate P0 escalation.",
    },
    {
        "id": "INC002",
        "content": (
            "Dark mode toggle is slightly misaligned on iPad screens with iOS 17. "
            "Reported by 3 users. No functional impact. UI only."
        ),
        "expected_action": "no_escalate",
        "expected_priority": "P3",
        "notes": "Minor cosmetic UI bug with no functional impact. Log as P3 enhancement.",
    },
    {
        "id": "INC003",
        "content": (
            "User data export feature is broken — exports generate empty files. "
            "Affecting ~30% of export requests. GDPR compliance risk. "
            "Legal team notified. Started 2 hours ago after deployment v4.1.2."
        ),
        "expected_action": "escalate",
        "expected_priority": "P1",
        "notes": "GDPR compliance risk + 30% failure rate requires P1 escalation.",
    },
    {
        "id": "INC004",
        "content": (
            "Search autocomplete is 200ms slower than usual after today's update. "
            "Performance degrades slightly but all functionality works. "
            "Affects about 15% of search queries."
        ),
        "expected_action": "no_escalate",
        "expected_priority": "P2",
        "notes": "Performance regression without outage. Log as P2 to fix in next sprint.",
    },
]
ESCALATION_VALID_ACTIONS = ["escalate", "no_escalate"]
ESCALATION_VALID_PRIORITIES = ["P0", "P1", "P2", "P3"]

# ---------------------------------------------------------------------------
# TASK CONFIG
# ---------------------------------------------------------------------------
TASKS = {
    "ticket_classification": {
        "difficulty": "easy",
        "description": "Classify incoming support tickets into the correct category.",
        "items": CLASSIFICATION_TICKETS,
        "valid_actions": ["classify"],
        "valid_values": CLASSIFICATION_VALID_VALUES,
    },
    "ticket_resolution": {
        "difficulty": "medium",
        "description": "Respond to support tickets using the knowledge base.",
        "items": RESOLUTION_TICKETS,
        "valid_actions": ["respond"],
        "valid_values": None,
    },
    "incident_escalation": {
        "difficulty": "hard",
        "description": "Assess incidents and make escalation and priority decisions.",
        "items": ESCALATION_INCIDENTS,
        "valid_actions": ESCALATION_VALID_ACTIONS,
        "valid_values": ESCALATION_VALID_PRIORITIES,
    },
}


# ---------------------------------------------------------------------------
# GRADERS
# ---------------------------------------------------------------------------

def grade_classification(action: SupportAction, item: Dict) -> Tuple[float, str]:
    """Grade a ticket classification action. Returns (score 0.0-1.0, feedback)."""
    submitted = action.value.strip().lower().replace(" ", "_")
    expected = item["expected"].lower()
    related = [r.lower() for r in item.get("related", [])]

    if submitted == expected:
        return 1.0, f"✅ Correct! '{submitted}' is the right category."
    if submitted in related:
        return 0.7, f"⚠️ Partially correct. '{submitted}' is related but '{expected}' is more precise."
    # Partial credit for correct top-level domain
    domain_map = {
        "authentication_issue": "auth",
        "account_issue": "auth",
        "billing_issue": "billing",
        "payment_issue": "billing",
        "technical_issue": "tech",
        "bug_report": "tech",
    }
    if domain_map.get(submitted) and domain_map.get(submitted) == domain_map.get(expected):
        return 0.4, f"⚠️ Same domain but wrong category. Expected '{expected}', got '{submitted}'."
    return 0.0, f"❌ Wrong. Expected '{expected}', got '{submitted}'."


def grade_resolution(action: SupportAction, item: Dict) -> Tuple[float, str]:
    """Grade a KB-based response. Score based on keyword coverage."""
    response = action.value.lower()
    keywords = [k.lower() for k in item["keywords"]]
    matched = sum(1 for kw in keywords if kw in response)
    ratio = matched / len(keywords) if keywords else 0.0

    if ratio >= 0.8:
        score = 1.0
        feedback = f"✅ Excellent response! Covered {matched}/{len(keywords)} key points."
    elif ratio >= 0.5:
        score = 0.6
        feedback = f"⚠️ Adequate response. Covered {matched}/{len(keywords)} key points."
    elif ratio >= 0.25:
        score = 0.3
        feedback = f"⚠️ Incomplete response. Only covered {matched}/{len(keywords)} key points."
    else:
        score = 0.0
        feedback = f"❌ Response did not address the issue. Covered {matched}/{len(keywords)} key points."
    return score, feedback


def grade_escalation(action: SupportAction, item: Dict) -> Tuple[float, str]:
    """Grade escalation decision + priority assignment."""
    submitted_action = action.action_type.strip().lower().replace(" ", "_")
    submitted_priority = (action.priority or action.value or "").strip().upper()

    expected_action = item["expected_action"]
    expected_priority = item["expected_priority"]

    action_correct = submitted_action in (expected_action, expected_action.replace("_", ""))
    priority_correct = submitted_priority == expected_priority

    if action_correct and priority_correct:
        return 1.0, f"✅ Perfect! Correct decision ({submitted_action}) and priority ({submitted_priority})."
    elif action_correct and not priority_correct:
        return 0.6, f"⚠️ Correct decision but wrong priority. Expected {expected_priority}, got {submitted_priority}."
    elif not action_correct and priority_correct:
        return 0.3, f"⚠️ Wrong decision but right priority. Expected {expected_action}, got {submitted_action}."
    else:
        # Extra penalty for wrong escalation on critical incidents
        penalty = -0.3 if expected_action == "escalate" and submitted_action == "no_escalate" else 0.0
        score = max(0.0, 0.0 + penalty)
        return score, f"❌ Wrong decision and priority. Expected {expected_action}/{expected_priority}."


# ---------------------------------------------------------------------------
# ENVIRONMENT CLASS
# ---------------------------------------------------------------------------

class SupportDeskEnvironment:
    """
    SupportDesk RL Environment — simulates real customer support operations.

    Three tasks:
      - ticket_classification (Easy): classify tickets into categories
      - ticket_resolution (Medium): respond using knowledge base
      - incident_escalation (Hard): make escalation and priority decisions
    """

    def __init__(self):
        self._state: Optional[SupportState] = None
        self._task_config: Optional[Dict] = None
        self._items: Optional[list] = None

    def reset(self, task_name: str = "ticket_classification") -> SupportObservation:
        """Start a new episode for the given task."""
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Valid: {list(TASKS.keys())}")

        config = TASKS[task_name]
        self._task_config = config
        self._items = config["items"]

        self._state = SupportState(
            task_id=task_name,
            episode_id=str(uuid.uuid4()),
            current_index=0,
            total_items=len(self._items),
            step_count=0,
            total_reward=0.0,
            done=False,
        )

        return self._make_observation(
            message=f"Episode started. Task: {task_name} ({config['difficulty']}). {len(self._items)} items to process."
        )

    def step(self, action: SupportAction) -> Tuple[SupportObservation, float, bool, Dict]:
        """Execute an action and return (observation, reward, done, info)."""
        if self._state is None or self._state.done:
            raise RuntimeError("Call reset() before step(), or episode is already done.")

        idx = self._state.current_index
        item = self._items[idx]
        task_id = self._state.task_id

        # --- Grade the action ---
        if task_id == "ticket_classification":
            reward, feedback = grade_classification(action, item)
        elif task_id == "ticket_resolution":
            reward, feedback = grade_resolution(action, item)
        elif task_id == "incident_escalation":
            reward, feedback = grade_escalation(action, item)
        else:
            reward, feedback = 0.0, "Unknown task."

        # --- Advance state ---
        self._state.step_count += 1
        self._state.total_reward += reward
        self._state.current_index += 1

        done = self._state.current_index >= self._state.total_items
        self._state.done = done

        obs = self._make_observation(reward=reward, message=feedback, done=done)
        return obs, reward, done, {"feedback": feedback, "item_id": item["id"]}

    @property
    def state(self) -> SupportState:
        if self._state is None:
            return SupportState(task_id="none", episode_id="none")
        return self._state

    def _make_observation(
        self,
        reward: float = 0.0,
        message: str = "",
        done: bool = False,
    ) -> SupportObservation:
        """Build observation for the current state."""
        if self._state is None or self._state.done:
            return SupportObservation(
                task_id="none",
                item_id="done",
                content="Episode complete.",
                done=True,
                reward=reward,
                message=message,
            )

        idx = self._state.current_index
        if idx >= len(self._items):
            return SupportObservation(
                task_id=self._state.task_id,
                item_id="done",
                content="All items processed. Episode complete.",
                done=True,
                reward=reward,
                message=message,
            )

        item = self._items[idx]
        config = self._task_config

        return SupportObservation(
            task_id=self._state.task_id,
            item_id=item["id"],
            content=item["content"],
            context=item.get("context"),
            valid_actions=config.get("valid_actions"),
            valid_values=config.get("valid_values"),
            step=self._state.step_count,
            message=message,
            done=done,
            reward=reward,
        )
