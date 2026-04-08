"""
inference.py — SupportDesk-OpenEnv agent runner.

Required env vars:
  HF_TOKEN      Hugging Face token (used as OpenAI-compatible API key)
  API_BASE_URL  (optional) defaults to https://api.openai.com/v1
  MODEL_NAME    (optional) defaults to gpt-4.1-mini
  SEED          (optional) integer seed for reproducibility
"""

import os
import re
import json
import sys
from openai import OpenAI
from env.environment import SupportDeskEnv

# ── Config ─────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")
SEED         = int(os.getenv("SEED", "42"))
MAX_EPISODES = int(os.getenv("MAX_EPISODES", "1"))

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is required.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
env    = SupportDeskEnv(seed=SEED)

# ── Prompt helpers ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI support desk agent. 
Your job is to decide the correct action for each customer support ticket.

VALID ACTIONS (reply with EXACTLY one):
  classify_ticket  — ticket needs routing/labelling before it can be handled
  respond_ticket   — you can resolve this directly with a helpful answer
  escalate_ticket  — ticket involves billing fraud, security, outages, or high-severity issues

Rules:
- Reply with ONLY the action keyword on the first line.
- Optionally add a brief explanation on subsequent lines.
- Never invent new action names."""


def build_prompt(obs: dict) -> str:
    ticket  = obs["ticket"]
    kb      = obs["knowledge_base"]
    step    = obs["step"]
    episode = obs["episode"]

    kb_text = "\n".join(f"  • {entry}" for entry in kb)

    return (
        f"Episode {episode} | Step {step}\n\n"
        f"TICKET [{ticket['id']}] (Priority: {ticket['priority'].upper()})\n"
        f"{ticket['text']}\n\n"
        f"KNOWLEDGE BASE:\n{kb_text}\n\n"
        f"What is the correct action? Reply with one of: "
        f"classify_ticket | respond_ticket | escalate_ticket"
    )


def extract_action(raw: str) -> tuple[str, str]:
    """
    Extract the action keyword from the model's raw response.
    Returns (action, response_text).
    """
    valid = {"classify_ticket", "respond_ticket", "escalate_ticket"}
    lines = raw.strip().splitlines()

    # Check first line for exact match
    first = lines[0].strip().lower() if lines else ""
    if first in valid:
        response_text = "\n".join(lines[1:]).strip()
        return first, response_text

    # Fallback: scan full text for any valid keyword
    for keyword in valid:
        if keyword in raw.lower():
            return keyword, raw

    # Last resort: return raw (will be caught as invalid by env)
    return raw.strip().lower(), ""


# ── Episode runner ─────────────────────────────────────────────────────────

def run_episode(episode_num: int) -> dict:
    obs  = env.reset()
    done = False
    step = 0
    rewards = []
    correct = 0
    total   = 0

    print(f"[START] episode={episode_num} task=supportdesk model={MODEL_NAME} seed={SEED}")

    while not done and step < 20:
        prompt = build_prompt(obs)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,   # deterministic for eval
                max_tokens=200,
            )
            raw = completion.choices[0].message.content
        except Exception as exc:
            print(f"[ERROR] API call failed: {exc}", file=sys.stderr)
            break

        action, response_text = extract_action(raw)

        obs, reward, done, info = env.step(action, response=response_text)
        rewards.append(reward)
        step += 1
        total += 1
        if info.get("was_correct"):
            correct += 1

        error_str = info.get("error", "null")
        print(
            f"[STEP] step={step} ticket={info.get('ticket_id', '?')} "
            f"action={action} correct={info.get('correct_action', '?')} "
            f"reward={reward:.4f} done={str(done).lower()} error={error_str}"
        )

    success      = done and (correct == total)
    reward_list  = ",".join(f"{r:.4f}" for r in rewards)
    total_reward = sum(rewards)

    print(
        f"[END] episode={episode_num} success={str(success).lower()} "
        f"steps={step} correct={correct}/{total} "
        f"total_reward={total_reward:.4f} rewards=[{reward_list}]"
    )

    return {
        "episode": episode_num,
        "success": success,
        "steps": step,
        "correct": correct,
        "total": total,
        "total_reward": total_reward,
        "rewards": rewards,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def run():
    all_stats = []
    for ep in range(1, MAX_EPISODES + 1):
        stats = run_episode(ep)
        all_stats.append(stats)

    # Aggregate summary
    n          = len(all_stats)
    avg_reward = sum(s["total_reward"] for s in all_stats) / n
    win_rate   = sum(1 for s in all_stats if s["success"]) / n

    print(
        f"\n[SUMMARY] episodes={n} avg_total_reward={avg_reward:.4f} "
        f"win_rate={win_rate:.2%}"
    )


if __name__ == "__main__":
    run()
