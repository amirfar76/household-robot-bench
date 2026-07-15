# Household Robot Benchmark

A lightweight, text-based household manipulation benchmark for evaluating LLM agents on multi-step planning tasks. Inspired by the [SayCan](https://say-can.github.io/) task taxonomy (Ahn et al., 2022).

## Overview

A robot navigates a four-room house (kitchen, living room, bedroom, bathroom) and manipulates household objects to complete tasks from four families:

| Family | Example task |
|--------|-------------|
| **Fetch** | *Bring the apple to the living room.* |
| **Place** | *Move the book from the bedroom to the kitchen.* |
| **Clean** | *Clean the cup and bring it to the living room.* |
| **Heat** | *Heat the mug and bring it to the bedroom.* |

**Key properties:**
- Pure Python standard library — no dependencies
- Fully deterministic given a seed; 16 objects, 4 rooms, 4 task families
- Binary terminal reward: 1.0 on success, 0.0 on timeout (default max 6 steps)
- ReAct-compatible: structured `Action: <action>` output format

## Installation

```bash
pip install git+https://github.com/amirfar76/household-robot-bench.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/amirfar76/household-robot-bench.git
cd household-robot-bench
pip install -e .
```

## Quickstart

```python
from household_robot import HouseholdRobotTask

task = HouseholdRobotTask(seed=0, max_steps=6)

# Start an episode
obs = task.reset()
print(obs)
# Task: Bring the apple to the living room.
#
# You are in: living room
# You see here: coffee table, sofa, bookshelf
# You are holding: nothing  |  Step: 0/6
#
# Valid actions: go to kitchen | go to bedroom | go to bathroom

# Wrap the observation in the two-shot instruction prompt for your LLM
prompt = task.make_prompt(obs)

# Parse an action from the model's response
model_output = "I need to go to the kitchen first to find the apple.\nAction: go to kitchen"
action = task.extract_action(model_output)  # "go to kitchen"

# Step the environment
obs, reward, done = task.step(action)
```

## Running the example

```bash
python example.py
```

This runs a single episode with random action selection to demonstrate the API.

## API Reference

### `HouseholdRobotTask(seed=0, max_steps=6, base_seed=31337)`

| Method | Description |
|--------|-------------|
| `reset() -> str` | Advance to the next episode; return initial observation. |
| `reset_to(episode_idx) -> str` | Jump to a specific episode index. |
| `step(action) -> (obs, reward, done)` | Execute one action. |
| `make_prompt(obs) -> str` | Wrap observation in the two-shot instruction prompt. |
| `extract_action(text) -> str \| None` | Parse `Action: ...` from a model output string. |
| `score(pred, gold) -> float` | 1.0 if strings match (case-insensitive), else 0.0. |

### Action space

| Action | Precondition |
|--------|-------------|
| `go to <room>` | Always available (for other rooms) |
| `pick up <object>` | Object is in current room; not holding anything |
| `put down <object>` | Currently holding that object |
| `heat <object> in microwave` | In kitchen, holding a heatable object |
| `clean <object> in sink` | In kitchen or bathroom, holding a cleanable object |

## Reproducibility

Episodes are generated deterministically:

```python
ep_seed = base_seed + episode_idx * 1000 + seed
```

Use `task.reset_to(episode_idx)` to reproduce any specific episode.

## License

MIT
