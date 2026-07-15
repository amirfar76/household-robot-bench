"""Minimal usage example — runs without an LLM by taking random valid actions."""

import random
import re
from household_robot import HouseholdRobotTask


def extract_valid_actions(obs: str) -> list[str]:
    m = re.search(r"Valid actions: (.+)$", obs, re.MULTILINE)
    if not m:
        return []
    return [a.strip() for a in m.group(1).split("|")]


def run_random_episode(episode_idx: int = 0, seed: int = 42) -> None:
    task = HouseholdRobotTask(seed=0)
    obs = task.reset_to(episode_idx)

    print(f"=== Episode {episode_idx} ===")
    print(obs)
    print()

    rng = random.Random(seed)
    done = False
    total_reward = 0.0

    while not done:
        actions = extract_valid_actions(obs)
        action = rng.choice(actions) if actions else "go to kitchen"
        print(f">> Action: {action}")
        obs, reward, done = task.step(action)
        total_reward += reward
        print(obs)
        print()

    print(f"Episode finished. Reward: {total_reward}")


if __name__ == "__main__":
    run_random_episode(episode_idx=0)
