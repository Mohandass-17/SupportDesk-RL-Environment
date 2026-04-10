---
# SupportDesk RL Environment
tags:
  - openenv
  - reinforcement-learning
  - rl-environment
  - customer-support
  - agent-evaluation
---

# 🎧 SupportDesk RL — OpenEnv Customer Support Operations Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **An OpenEnv reinforcement learning environment that simulates real-world customer support operations.**
> Train AI agents to triage tickets, resolve queries, and make critical escalation decisions — tasks performed daily by support teams at companies like Meta and Hugging Face.

---

## 🌍 Motivation

Modern AI companies handle thousands of support tickets, incidents, and user queries daily. Human agents must:

- **Classify** incoming tickets into the right category quickly
- **Resolve** queries accurately using knowledge base articles
- **Escalate** critical incidents before they become outages

These workflows are increasingly being augmented by AI. **SupportDesk RL** provides a clean, deterministic, and reproducible environment to train and evaluate agents on these real operational tasks — making it directly useful for post-training LLMs to be genuinely helpful in production workflows.

---

## 🧠 Environment Overview

```
LLM Agent
    │
    ▼
inference.py
    │
    ▼
SupportDesk RL Environment  (reset / step / state)
    │
    ├── Task 1: Ticket Classification  (Easy)
    ├── Task 2: Ticket Resolution      (Medium)
    └── Task 3: Incident Escalation    (Hard)
    │
    ▼
Programmatic Graders → Reward (0.0 – 1.0)
```

---

## 🎮 Action Space

| Field | Type | Description |
|---|---|---|
| `action_type` | `str` | `"classify"` \| `"respond"` \| `"escalate"` \| `"no_escalate"` |
| `value` | `str` | The classification label, response text, or escalation decision |
| `priority` | `str` (optional) | `"P0"` \| `"P1"` \| `"P2"` \| `"P3"` — used in escalation task |
| `rationale` | `str` (optional) | Optional explanation from the agent |

---

## 👁️ Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Which task is running |
| `item_id` | `str` | Current ticket / incident ID |
| `content` | `str` | The main text the agent must act on |
| `context` | `str` (optional) | Knowledge base article (for resolution task) |
| `valid_actions` | `List[str]` | Allowed action types |
| `valid_values` | `List[str]` (optional) | Allowed values (classification labels / priorities) |
| `step` | `int` | Current step in episode |
| `message` | `str` | Feedback from the last action |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Reward from the last action |

---

## 🎯 Tasks

### Task 1 — Ticket Classification `(Easy)`

The agent receives 5 incoming customer support tickets and must classify each into the correct category.

**Valid categories:** `authentication_issue`, `billing_issue`, `technical_issue`, `feature_request`, `general_inquiry`

**Grader:**
| Outcome | Score |
|---|---|
| Exact category match | 1.0 |
| Related category (same domain) | 0.4 – 0.7 |
| Wrong category | 0.0 |

**Example:**
```
Ticket: "I cannot log in to my account after the password reset."
Correct action: classify → authentication_issue → score: 1.0
```

---

### Task 2 — Ticket Resolution `(Medium)`

The agent receives 3 customer tickets plus a relevant knowledge base article and must write a helpful response.

**Grader:** Keyword coverage — how many key resolution terms appear in the response.

| Coverage | Score |
|---|---|
| ≥ 80% keywords | 1.0 |
| ≥ 50% keywords | 0.6 |
| ≥ 25% keywords | 0.3 |
| < 25% keywords | 0.0 |

**Example:**
```
Ticket: "I forgot my password and the reset link expired."
KB: "Go to Settings > Security > Reset Password. Enter your email..."
Good response: "Please go to Settings, then Security, and click Reset Password..."
Score: 1.0 (covers: reset, password, settings, security, email, link)
```

---

### Task 3 — Incident Escalation `(Hard)`

The agent receives 4 production incident reports and must:
1. Decide whether to **escalate** or **not escalate**
2. Assign the correct **priority** (P0 = Critical outage → P3 = Enhancement)

**Grader:**
| Outcome | Score |
|---|---|
| Correct decision + correct priority | 1.0 |
| Correct decision, wrong priority | 0.6 |
| Wrong decision, correct priority | 0.3 |
| Wrong escalation on critical incident | 0.0 |

**Example:**
```
Incident: "Payment service returning 503 for 100% of requests. $15K/min revenue loss."
Correct: escalate → P0 → score: 1.0
Wrong:   no_escalate → P2 → score: 0.0 (critical incident penalty)
```

---

## 📊 Reward Design

Rewards are dense and trajectory-based (not just end-of-episode):

- Every step returns a score **0.0 – 1.0** based on grader outcome
- Partial credit is given for near-correct decisions (not binary pass/fail)
- Critical failures (e.g. not escalating a P0 incident) receive extra penalty
- Reward accumulates across all items in a task episode

This design provides clear learning signal for RL training and is fair for LLM evaluation.

---

## 🏗️ Project Structure

```
.
├── inference.py              ← Baseline inference script (REQUIRED: root directory)
├── openenv.yaml              ← OpenEnv manifest
├── Dockerfile                ← Container definition (HF Spaces compatible)
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
│
└── supportdesk_env/
    ├── __init__.py
    ├── models.py             ← Pydantic Action, Observation, State
    ├── client.py             ← EnvClient (openenv-core compatible)
    └── server/
        ├── __init__.py
        ├── environment.py    ← Core logic, tasks, graders, reward engine
        └── app.py            ← FastAPI server (HTTP + WebSocket)
```

---

## 🚀 Setup & Usage

### Local Development

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/supportdesk-openenv
cd supportdesk-openenv

# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn supportdesk_env.server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Run the Baseline Inference Script

```bash
export HF_TOKEN=your_hf_api_token
export API_BASE_URL=https://api.openai.com/v1  # or HF Inference API
export MODEL_NAME=gpt-4o-mini

python inference.py
```

**Expected output format:**
```
[START] task=ticket_classification env=supportdesk_ops model=gpt-4o-mini
[STEP] step=1 action=authentication_issue reward=1.00 done=false error=null
[STEP] step=2 action=billing_issue reward=1.00 done=false error=null
...
[END] success=true steps=5 rewards=1.00,1.00,1.00,0.70,1.00
```

### Docker

```bash
# Build the container
docker build -t supportdesk-openenv .

# Run it
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  supportdesk-openenv
```

### OpenEnv Validation

```bash
pip install openenv-core
openenv validate
```

Expected:
```
✅ All 3/3 checks passed! Your submission is ready to submit.
```

---

## 🤗 Hugging Face Deployment

This environment is deployed as a Docker Space on Hugging Face:

1. Create a new Space → type: **Docker**
2. Add the tag: `openenv`
3. Push this repository to the Space
4. Set Space secrets: `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`
5. Wait for status: **Running**

The validator will ping `/reset` — if it returns HTTP 200, the Space is live.

---

## 📈 Baseline Scores

Results using `gpt-4o-mini` on the default task configuration:

| Task | Difficulty | Avg Score | Notes |
|---|---|---|---|
| `ticket_classification` | Easy | ~0.88 | Occasional synonym mismatch |
| `ticket_resolution` | Medium | ~0.72 | Misses minor keywords |
| `incident_escalation` | Hard | ~0.65 | Priority assignment is challenging |

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | ✅ Yes | — | Hugging Face / LLM API token |
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM endpoint |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model identifier |

---

## 🏆 Why SupportDesk RL?

- **Real-world utility** — Support automation is a genuine unsolved problem at scale
- **Deterministic graders** — Same input always produces same score (reproducible)
- **Difficulty progression** — Easy classification → Medium KB QA → Hard multi-factor decision
- **Dense rewards** — Partial credit at every step, not sparse end-of-episode signal
- **Lightweight** — Runs entirely in-process; no external APIs or databases needed
- **Novel domain** — First support operations environment in the OpenEnv ecosystem

---

## 📄 License

MIT License. See [LICENSE](LICENSE).
