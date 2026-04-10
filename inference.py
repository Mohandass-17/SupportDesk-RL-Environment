import os
import sys
import time

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN is required.")
    sys.exit(1)

from openai import OpenAI
from environment import SupportDeskEnv

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM = "You are a support desk classifier. Reply with ONLY one word: classify_ticket, respond_ticket, or escalate_ticket"

def get_action(text):
    valid = ["escalate_ticket", "classify_ticket", "respond_ticket"]
    text = text.strip().lower()
    for v in valid:
        if v in text:
            return v
    return "respond_ticket"

def run_episode(episode_num):
    env = SupportDeskEnv(seed=episode_num)
    obs = env.reset()
    done = False
    step = 0
    rewards = []
    correct = 0

    print("[START] episode=" + str(episode_num) + " model=" + MODEL_NAME)

    while not done and step < 20:
        ticket = obs["ticket"]
        prompt = (
            "Ticket: " + ticket["text"] + "\n"
            "Priority: " + ticket["priority"] + "\n"
            "Category: " + ticket["category"] + "\n\n"
            "classify_ticket = bug or technical issue needing routing\n"
            "respond_ticket = simple how-to or account question\n"
            "escalate_ticket = fraud, hacking, outage, server error, duplicate charge\n\n"
            "Reply with ONE word only."
        )

        try:
            res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=20,
            )
            raw = res.choices[0].message.content
        except Exception as e:
            print("[ERROR] " + str(e))
            time.sleep(5)
            break

        action = get_action(raw)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        step += 1

        if info.get("was_correct"):
            correct += 1

        print(
            "[STEP] ep=" + str(episode_num) +
            " step=" + str(step) +
            " ticket=" + str(info.get("ticket_id", "?")) +
            " action=" + action +
            " expected=" + str(info.get("correct_action", "?")) +
            " correct=" + str(info.get("was_correct", False)) +
            " reward=" + str(reward)
        )

    total = round(sum(rewards), 4)
    print("[END] episode=" + str(episode_num) +
          " correct=" + str(correct) + "/" + str(step) +
          " total_reward=" + str(total))
    return total

if __name__ == "__main__":
    print("[SYSTEM] SupportDesk-OpenEnv is running...")
    episode = 1
    all_rewards = []

    while True:
        try:
            r = run_episode(episode)
            all_rewards.append(r)
            avg = round(sum(all_rewards) / len(all_rewards), 4)
            print("[SUMMARY] episodes=" + str(episode) + " avg_reward=" + str(avg))
            episode += 1
            print("[SYSTEM] Waiting 30 seconds before next episode...")
            time.sleep(30)
        except Exception as e:
            print("[ERROR] " + str(e))
            time.sleep(10)
