def compute_reward(action_type, correct_action, step, response=None):
    reward = 0.0
    if action_type == correct_action:
        reward += 0.60
    if action_type == "respond_ticket" and response and len(response.strip()) > 10:
        reward += 0.20
    if action_type == correct_action and step <= 3:
        reward += 0.10
    if step > 10:
        reward -= 0.10
    if step > 15:
        reward -= 0.20
    return round(max(0.0, min(reward, 1.0)), 4)
