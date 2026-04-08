#!/usr/bin/env python3
"""
SupportDesk RL — Baseline Inference Script
Runs an LLM agent through all 3 tasks and logs results in OpenEnv format.

Required env vars:
  HF_TOKEN    — Hugging Face API token (required)
  API_BASE_URL — LLM endpoint (default: https://api.openai.com/v1)
  MODEL_NAME   — Model identifier (default: gpt-4o-mini)
"""
import os
import sys
import json
import re

# ---------------------------------------------------------------------------
# Environment variable configuration (required by OpenEnv spec)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError(
        "HF_TOKEN environment variable is required. "
        "Set it to your Hugging Face API token before running."
    )

# ---------------------------------------------------------------------------
# Import OpenAI client (required by OpenEnv spec)
# ---------------------------------------------------------------------------
from openai import OpenAI

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Import environment directly (avoids needing a running server for eval)
# ---------------------------------------------------------------------------
try:
    from supportdesk_env.server.environment import SupportDeskEnvironment
    from supportdesk_env.models import SupportAction
except ImportError:
    # Running from repo root without the package installed
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from supportdesk_env.server.environment import SupportDeskEnvironment
    from supportdesk_env.models import SupportAction


# ---------------------------------------------------------------------------
# Task-specific prompt builders
# ---------------------------------------------------------------------------

def build_classification_prompt(obs) -> str:
    return (
        "You are a customer support operations agent. "
        "Classify the following support ticket into exactly ONE category.\n\n"
        f"Valid categories: {', '.join(obs.valid_values or [])}\n\n"
        f"Ticket: {obs.content}\n\n"
        "Respond with ONLY the category name (e.g. billing_issue). No explanation."
    )


def build_resolution_prompt(obs) -> str:
    kb = f"\nKnowledge Base:\n{obs.context}\n" if obs.context else ""
    return (
        "You are a customer support specialist. "
        "Write a helpful, concise response to the customer's question "
        "using the knowledge base provided.\n"
        f"{kb}\n"
        f"Customer question: {obs.content}\n\n"
        "Your response:"
    )


def build_escalation_prompt(obs) -> str:
    return (
        "You are a senior support operations manager. "
        "Read the incident report below and decide:\n"
        "1. Should it be ESCALATED or NOT?\n"
        "2. What priority should it be assigned? (P0=Critical outage, P1=Major, P2=Minor, P3=Enhancement)\n\n"
        f"Incident: {obs.content}\n\n"
        "Respond in EXACTLY this format (no extra text):\n"
        "DECISION: escalate|no_escalate\n"
        "PRIORITY: P0|P1|P2|P3"
    )


# ---------------------------------------------------------------------------
# Action parsers — convert raw LLM text to SupportAction
# ---------------------------------------------------------------------------

def parse_classification_action(text: str, valid_values: list) -> SupportAction:
    text_clean = text.strip().lower().replace(" ", "_").replace("-", "_")
    # Try exact match first
    for v in valid_values:
        if v.lower() == text_clean:
            return SupportAction(action_type="classify", value=v)
    # Try substring match
    for v in valid_values:
        if v.lower() in text_clean or text_clean in v.lower():
            return SupportAction(action_type="classify", value=v)
    # Default fallback
    return SupportAction(action_type="classify", value=text_clean[:64])


def parse_resolution_action(text: str) -> SupportAction:
    return SupportAction(action_type="respond", value=text.strip()[:512])


def parse_escalation_action(text: str) -> SupportAction:
    text_up = text.upper()
    # Extract decision
    decision = "escalate"
    if "NO_ESCALATE" in text_up or "NO ESCALATE" in text_up or "NOT ESCALATE" in text_up:
        decision = "no_escalate"
    elif "ESCALATE" in text_up:
        decision = "escalate"

    # Extract priority
    priority = "P2"  # default
    for p in ["P0", "P1", "P2", "P3"]:
        if p in text_up:
            priority = p
            break

    return SupportAction(action_type=decision, value=decision, priority=priority)


PROMPT_BUILDERS = {
    "ticket_classification": build_classification_prompt,
    "ticket_resolution": build_resolution_prompt,
    "incident_escalation": build_escalation_prompt,
}

ACTION_PARSERS = {
    "ticket_classification": lambda text, obs: parse_classification_action(
        text, obs.valid_values or []
    ),
    "ticket_resolution": lambda text, obs: parse_resolution_action(text),
    "incident_escalation": lambda text, obs: parse_escalation_action(text),
}

TASK_BENCHMARK = "supportdesk_ops"
MAX_STEPS_PER_TASK = 20  # safety ceiling


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(task_id: str, env: SupportDeskEnvironment) -> None:
    """Run one complete task episode and print OpenEnv-format logs."""
    obs = env.reset(task_name=task_id)

    # [START] line
    print(f"[START] task={task_id} env={TASK_BENCHMARK} model={MODEL_NAME}")
    sys.stdout.flush()

    step = 0
    rewards = []
    done = False
    last_error = None

    while not done and step < MAX_STEPS_PER_TASK:
        # Build prompt for current observation
        prompt_fn = PROMPT_BUILDERS[task_id]
        prompt = prompt_fn(obs)

        # Call LLM
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert AI operations agent. "
                            "Follow instructions precisely and respond in the exact format requested."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            raw_text = response.choices[0].message.content.strip()
            last_error = None
        except Exception as exc:
            raw_text = ""
            last_error = str(exc)[:120].replace("\n", " ")

        # Parse action
        parse_fn = ACTION_PARSERS[task_id]
        action = parse_fn(raw_text, obs)

        # Step environment
        try:
            obs, reward, done, info = env.step(action)
            reward = round(float(reward), 2)
        except Exception as exc:
            reward = 0.0
            done = True
            last_error = str(exc)[:120].replace("\n", " ")

        step += 1
        rewards.append(reward)

        # Sanitise action string for single-line log
        action_str = action.value.replace("\n", " ").replace("\r", "")[:80]

        # [STEP] line
        error_field = last_error if last_error else "null"
        print(
            f"[STEP] step={step} action={action_str} "
            f"reward={reward:.2f} done={str(done).lower()} error={error_field}"
        )
        sys.stdout.flush()

    success = done and step <= MAX_STEPS_PER_TASK
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # [END] line
    print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_inference():
    env = SupportDeskEnvironment()
    tasks = ["ticket_classification", "ticket_resolution", "incident_escalation"]
    for task_id in tasks:
        print(f"\n{'='*60}")
        print(f"Running task: {task_id}")
        print(f"{'='*60}")
        try:
            run_task(task_id, env)
        except Exception as exc:
            # Always emit [END] even on exception
            print(f"[END] success=false steps=0 rewards=")
            print(f"ERROR: {exc}", file=sys.stderr)
        sys.stdout.flush()


if __name__ == "__main__":
    run_inference()
