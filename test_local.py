#!/usr/bin/env python3
"""
Quick local smoke test — verifies the environment works WITHOUT needing HF_TOKEN.
Run this before deploying: python test_local.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from supportdesk_env.server.environment import SupportDeskEnvironment
from supportdesk_env.models import SupportAction

PASS = "✅ PASS"
FAIL = "❌ FAIL"

def test_task(task_id: str, actions: list, env: SupportDeskEnvironment):
    print(f"\n--- Testing: {task_id} ---")
    obs = env.reset(task_name=task_id)
    assert obs.task_id == task_id, f"Expected task_id={task_id}"
    assert not obs.done, "Should not be done after reset"
    print(f"  reset() {PASS} — first item: {obs.item_id}")

    total_reward = 0.0
    for i, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"  step {i+1}: reward={reward:.2f} done={done} — {info.get('feedback','')[:60]}")
        assert 0.0 <= reward <= 1.0, f"Reward out of range: {reward}"

    state = env.state
    assert state.task_id == task_id
    print(f"  state() {PASS} — steps={state.step_count} total_reward={total_reward:.2f}")
    print(f"  {PASS} {task_id} complete")
    return total_reward


def main():
    env = SupportDeskEnvironment()
    errors = []

    # --- Task 1: Classification ---
    try:
        classification_actions = [
            SupportAction(action_type="classify", value="authentication_issue"),
            SupportAction(action_type="classify", value="billing_issue"),
            SupportAction(action_type="classify", value="technical_issue"),
            SupportAction(action_type="classify", value="feature_request"),
            SupportAction(action_type="classify", value="general_inquiry"),
        ]
        r = test_task("ticket_classification", classification_actions, env)
        assert r >= 4.0, f"Expected total reward >=4.0, got {r}"
    except Exception as e:
        errors.append(f"ticket_classification: {e}")

    # --- Task 2: Resolution ---
    try:
        resolution_actions = [
            SupportAction(action_type="respond", value="Go to Settings > Security > Reset Password and enter your registered email to receive a reset link."),
            SupportAction(action_type="respond", value="To dispute a billing charge, go to Account > Billing > Transaction History and click Dispute."),
            SupportAction(action_type="respond", value="You can export your data via Settings > Privacy > Export My Data. A download link will be emailed to you."),
        ]
        test_task("ticket_resolution", resolution_actions, env)
    except Exception as e:
        errors.append(f"ticket_resolution: {e}")

    # --- Task 3: Escalation ---
    try:
        escalation_actions = [
            SupportAction(action_type="escalate", value="escalate", priority="P0"),
            SupportAction(action_type="no_escalate", value="no_escalate", priority="P3"),
            SupportAction(action_type="escalate", value="escalate", priority="P1"),
            SupportAction(action_type="no_escalate", value="no_escalate", priority="P2"),
        ]
        r = test_task("incident_escalation", escalation_actions, env)
        assert r >= 3.0, f"Expected total reward >=3.0 with perfect actions, got {r}"
    except Exception as e:
        errors.append(f"incident_escalation: {e}")

    # --- Summary ---
    print("\n" + "="*50)
    if errors:
        print(f"❌ {len(errors)} test(s) FAILED:")
        for e in errors:
            print(f"   {e}")
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED — Environment is ready to submit!")
        print("="*50)


if __name__ == "__main__":
    main()
