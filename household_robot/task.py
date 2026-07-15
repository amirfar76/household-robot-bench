"""Household Robot benchmark — task environment.

A SayCan-inspired (Ahn et al., 2022) text-based household manipulation
benchmark. A robot navigates four rooms and manipulates objects to complete
tasks drawn from four families: fetch, place, clean, heat.

Pure Python standard library. No external dependencies. Deterministic given a seed.

Quickstart
----------
>>> from household_robot import HouseholdRobotTask
>>> task = HouseholdRobotTask()
>>> obs = task.reset()
>>> prompt = task.make_prompt(obs)          # feed to your LLM
>>> obs, reward, done = task.step("go to kitchen")
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional


# ── World constants ───────────────────────────────────────────────────────────

ROOMS = ["kitchen", "living room", "bedroom", "bathroom"]

SURFACES = {
    "kitchen":     ["counter", "stove", "sink"],
    "living room": ["coffee table", "sofa", "bookshelf"],
    "bedroom":     ["bed", "dresser", "nightstand"],
    "bathroom":    ["bathtub", "shelf"],
}

APPLIANCES = {
    "kitchen":  ["microwave", "refrigerator"],
    "bathroom": ["sink"],
}

# (name, home_rooms, can_clean, can_heat)
OBJECTS = [
    ("apple",        ["kitchen", "living room"],          False, True),
    ("banana",       ["kitchen", "living room"],          False, False),
    ("orange",       ["kitchen"],                         False, True),
    ("cup",          ["kitchen", "bedroom", "bathroom"],  True,  False),
    ("mug",          ["kitchen", "bedroom"],              True,  True),
    ("plate",        ["kitchen"],                         True,  False),
    ("bowl",         ["kitchen"],                         True,  True),
    ("sponge",       ["kitchen", "bathroom"],             False, False),
    ("milk carton",  ["kitchen"],                         False, True),
    ("water bottle", ["kitchen", "bedroom"],              False, False),
    ("book",         ["bedroom", "living room"],          False, False),
    ("remote",       ["living room"],                     False, False),
    ("towel",        ["bathroom", "bedroom"],             False, False),
    ("keys",         ["bedroom"],                         False, False),
    ("soda can",     ["kitchen", "living room"],          False, False),
    ("bread",        ["kitchen"],                         False, True),
]

PROMPT_TEMPLATE = """\
You are a household robot executing tasks in a house with four rooms: \
kitchen, living room, bedroom, and bathroom.

Each step you receive an observation and a list of valid actions. \
Choose exactly ONE action that best advances the task.

EXAMPLE 1 — fetch task
Task: Bring the apple to the living room.
You are in: kitchen
You see here: apple (on counter), mug (on stove), sink, microwave, refrigerator
You are holding: nothing  |  Step: 1/6
Valid actions: go to living room | go to bedroom | go to bathroom | pick up apple | pick up mug
Action: pick up apple
---
You are in: kitchen
You see here: mug (on stove), sink, microwave, refrigerator
You are holding: apple  |  Step: 2/6
Valid actions: go to living room | go to bedroom | go to bathroom | put down apple
Action: go to living room
---
You are in: living room
You see here: coffee table, sofa, bookshelf
You are holding: apple  |  Step: 3/6
Valid actions: go to kitchen | go to bedroom | go to bathroom | put down apple
Action: put down apple
---
Task completed. ✓

EXAMPLE 2 — heat and deliver
Task: Heat the mug and bring it to the bedroom.
You are in: living room
You see here: coffee table, sofa, bookshelf, remote (on coffee table)
You are holding: nothing  |  Step: 1/6
Valid actions: go to kitchen | go to bedroom | go to bathroom | pick up remote
Action: go to kitchen
---
You are in: kitchen
You see here: mug (on stove), bread (on counter), sink, microwave, refrigerator
You are holding: nothing  |  Step: 2/6
Valid actions: go to living room | go to bedroom | go to bathroom | pick up mug | pick up bread
Action: pick up mug
---
You are in: kitchen
You see here: bread (on counter), sink, microwave, refrigerator
You are holding: mug  |  Step: 3/6
Valid actions: go to living room | go to bedroom | go to bathroom | put down mug | heat mug in microwave | clean mug in sink
Action: heat mug in microwave
---
[mug is now heated]
You are in: kitchen
You see here: bread (on counter), sink, microwave, refrigerator
You are holding: mug (heated)  |  Step: 4/6
Valid actions: go to living room | go to bedroom | go to bathroom | put down mug
Action: go to bedroom
---
You are in: bedroom
You see here: bed, dresser, nightstand
You are holding: mug (heated)  |  Step: 5/6
Valid actions: go to kitchen | go to living room | go to bathroom | put down mug
Action: put down mug
---
Task completed. ✓

Now solve the current task.

{observation}

Reason briefly about the task goal and the next best action, then output exactly ONE action on a new line as:
Action: <your chosen action>
"""


# ── Internal state dataclasses ────────────────────────────────────────────────

@dataclass
class _Obj:
    location: str
    is_clean: bool = False
    is_heated: bool = False


@dataclass
class _Task:
    description: str
    target_object: str
    target_room: str
    require_clean: bool
    require_heated: bool


# ── Episode generation ────────────────────────────────────────────────────────

def _generate_episode(episode_seed: int, max_objects: int = 8) -> tuple[dict, _Task]:
    rng = random.Random(episode_seed)
    objects: dict[str, _Obj] = {}

    catalog = list(OBJECTS)
    rng.shuffle(catalog)
    chosen = catalog[:max_objects]
    for name, home_rooms, _, _ in chosen:
        room = rng.choice(home_rooms)
        objects[name] = _Obj(location=room)

    template_idx = rng.randint(0, 3)

    if template_idx == 0:
        eligible = [n for n, _, _, _ in chosen]
        if not eligible:
            eligible = ["apple"]
        obj = rng.choice(eligible)
        current = objects.get(obj, _Obj(location="kitchen")).location
        targets = [r for r in ROOMS if r != current]
        target_room = rng.choice(targets)
        task = _Task(
            description=f"Bring the {obj} to the {target_room}.",
            target_object=obj, target_room=target_room,
            require_clean=False, require_heated=False,
        )

    elif template_idx == 1:
        cleanable = [(n, rms, cc, ch) for n, rms, cc, ch in chosen if cc]
        if not cleanable:
            cleanable = [("cup", ["kitchen"], True, False)]
            objects["cup"] = _Obj(location="bedroom")
        entry = rng.choice(cleanable)
        obj = entry[0]
        current = objects.get(obj, _Obj(location="bedroom")).location
        targets = [r for r in ["living room", "kitchen", "bedroom"] if r != current]
        target_room = rng.choice(targets)
        task = _Task(
            description=f"Clean the {obj} and bring it to the {target_room}.",
            target_object=obj, target_room=target_room,
            require_clean=True, require_heated=False,
        )
        if obj not in objects:
            objects[obj] = _Obj(location=rng.choice(["kitchen", "bathroom", "bedroom"]))

    elif template_idx == 2:
        heatable = [(n, rms, cc, ch) for n, rms, cc, ch in chosen if ch]
        if not heatable:
            heatable = [("mug", ["kitchen"], True, True)]
            objects["mug"] = _Obj(location="kitchen")
        entry = rng.choice(heatable)
        obj = entry[0]
        current = objects.get(obj, _Obj(location="kitchen")).location
        targets = [r for r in ["living room", "bedroom"] if r != current]
        target_room = rng.choice(targets) if targets else "living room"
        task = _Task(
            description=f"Heat the {obj} and bring it to the {target_room}.",
            target_object=obj, target_room=target_room,
            require_clean=False, require_heated=True,
        )
        if obj not in objects:
            objects[obj] = _Obj(location="kitchen")

    else:
        eligible = [n for n, _, _, _ in chosen]
        if not eligible:
            eligible = ["book"]
            objects["book"] = _Obj(location="bedroom")
        obj = rng.choice(eligible)
        current = objects.get(obj, _Obj(location="kitchen")).location
        targets = [r for r in ROOMS if r != current]
        target_room = rng.choice(targets)
        task = _Task(
            description=f"Move the {obj} from the {current} to the {target_room}.",
            target_object=obj, target_room=target_room,
            require_clean=False, require_heated=False,
        )

    return objects, task


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_success(objects: dict, task: _Task, holding: Optional[str],
                   robot_room: str) -> bool:
    obj = task.target_object
    state = objects.get(obj)
    if state is None:
        return False
    in_target = (state.location == task.target_room) or (
        holding == obj and robot_room == task.target_room
    )
    if not in_target:
        return False
    if task.require_clean and not state.is_clean:
        return False
    if task.require_heated and not state.is_heated:
        return False
    return True


def _valid_actions(objects: dict, robot_room: str,
                   holding: Optional[str]) -> list[str]:
    actions: list[str] = []
    for room in ROOMS:
        if room != robot_room:
            actions.append(f"go to {room}")
    if holding is None:
        for name, state in objects.items():
            if state.location == robot_room:
                actions.append(f"pick up {name}")
    else:
        actions.append(f"put down {holding}")
        if robot_room == "kitchen":
            _, _, can_clean, can_heat = next(
                (t for t in OBJECTS if t[0] == holding),
                (holding, [], False, False),
            )
            if can_heat:
                actions.append(f"heat {holding} in microwave")
            if can_clean:
                actions.append(f"clean {holding} in sink")
        if robot_room == "bathroom":
            _, _, can_clean, _ = next(
                (t for t in OBJECTS if t[0] == holding),
                (holding, [], False, False),
            )
            if can_clean:
                actions.append(f"clean {holding} in sink")
    return actions


def _render_obs(objects: dict, robot_room: str, holding: Optional[str],
                step: int, max_steps: int, task: _Task) -> str:
    here = [n for n, s in objects.items() if s.location == robot_room]
    here_strs = []
    surfaces = SURFACES.get(robot_room, [])
    for name in here:
        s = objects[name]
        surf = surfaces[0] if surfaces else robot_room
        qualifier = " (heated)" if s.is_heated else (" (clean)" if s.is_clean else "")
        here_strs.append(f"{name}{qualifier} (on {surf})")
    appliances = APPLIANCES.get(robot_room, [])
    furniture = SURFACES.get(robot_room, [])
    see_parts = here_strs + furniture + appliances

    holding_str = "nothing"
    if holding:
        hs = objects.get(holding, _Obj(location="held"))
        q = " (heated)" if hs.is_heated else (" (clean)" if hs.is_clean else "")
        holding_str = f"{holding}{q}"

    valid = _valid_actions(objects, robot_room, holding)
    valid_str = " | ".join(valid) if valid else "(no valid actions)"

    return (
        f"Task: {task.description}\n\n"
        f"You are in: {robot_room}\n"
        f"You see here: {', '.join(see_parts) if see_parts else '(empty)'}\n"
        f"You are holding: {holding_str}  |  Step: {step}/{max_steps}\n\n"
        f"Valid actions: {valid_str}"
    )


# ── Public API ────────────────────────────────────────────────────────────────

class HouseholdRobotTask:
    """Text-based household robot manipulation benchmark.

    A household robot navigates four rooms — kitchen, living room, bedroom,
    bathroom — picking up and manipulating objects to complete tasks drawn
    from four families:

    - **Fetch**: bring an object to a target room.
    - **Place** (rearrange): move an object from one room to another.
    - **Clean**: clean an object (kitchen sink / bathroom sink) and deliver it.
    - **Heat**: heat an object (kitchen microwave) and deliver it.

    Episodes are procedurally generated and fully deterministic given a seed.
    Binary terminal reward: 1.0 on success, 0.0 on timeout (max_steps reached).

    Parameters
    ----------
    seed : int
        Per-task RNG seed mixed into every episode index for reproducibility.
    max_steps : int
        Maximum number of steps per episode before timeout (default 6).
    base_seed : int
        Base seed for episode generation; change to get a different episode order.

    Usage
    -----
    >>> task = HouseholdRobotTask(seed=0)
    >>> obs = task.reset()                       # start episode 0
    >>> prompt = task.make_prompt(obs)           # LLM-ready prompt string
    >>> obs, reward, done = task.step("go to kitchen")
    >>> obs, reward, done = task.step("pick up apple")
    """

    name = "household"

    def __init__(self, seed: int = 0, max_steps: int = 6, base_seed: int = 31337):
        self._seed = seed
        self.max_steps = max_steps
        self._base_seed = base_seed
        self._cursor = 0
        self._objects: dict[str, _Obj] = {}
        self._task: Optional[_Task] = None
        self._robot_room: str = "living room"
        self._holding: Optional[str] = None
        self._step: int = 0

    def _init_episode(self, episode_idx: int) -> str:
        ep_seed = self._base_seed + episode_idx * 1000 + self._seed
        self._objects, self._task = _generate_episode(ep_seed)
        self._robot_room = "living room"
        self._holding = None
        self._step = 0
        return _render_obs(
            self._objects, self._robot_room, self._holding,
            self._step, self.max_steps, self._task,
        )

    def reset(self, seed: int = 0) -> str:
        """Advance to the next episode and return the initial observation."""
        obs = self._init_episode(self._cursor)
        self._cursor += 1
        return obs

    def reset_to(self, episode_idx: int) -> str:
        """Jump directly to episode `episode_idx` and return the initial observation."""
        self._cursor = episode_idx + 1
        return self._init_episode(episode_idx)

    def step(self, action: str) -> tuple[str, float, bool]:
        """Execute one action.

        Parameters
        ----------
        action : str
            One of the strings listed in the ``Valid actions:`` line of the
            current observation, e.g. ``"go to kitchen"`` or ``"pick up apple"``.

        Returns
        -------
        obs : str
            The next observation string.
        reward : float
            1.0 if the task was completed this step, else 0.0.
        done : bool
            True if the episode ended (success or timeout).
        """
        if self._task is None:
            return "No active task. Call reset() first.", 0.0, True

        action = (action or "").strip().lower()
        self._step += 1

        if action.startswith("go to "):
            dest = action[len("go to "):].strip()
            if dest in ROOMS:
                self._robot_room = dest

        elif action.startswith("pick up "):
            obj = action[len("pick up "):].strip()
            if (self._holding is None
                    and obj in self._objects
                    and self._objects[obj].location == self._robot_room):
                self._holding = obj
                self._objects[obj].location = "held"

        elif action.startswith("put down"):
            obj = (action[len("put down "):].strip()
                   if action.startswith("put down ") else self._holding)
            if obj and self._holding == obj:
                self._objects[obj].location = self._robot_room
                self._holding = None

        elif action.startswith("heat ") and "microwave" in action:
            obj_part = action[len("heat "):].replace(" in microwave", "").strip()
            if (self._robot_room == "kitchen"
                    and self._holding == obj_part
                    and obj_part in self._objects):
                self._objects[obj_part].is_heated = True

        elif action.startswith("clean ") and ("sink" in action or "bathtub" in action):
            obj_part = (action[len("clean "):]
                        .replace(" in sink", "").replace(" in bathtub", "").strip())
            if (self._robot_room in ("kitchen", "bathroom")
                    and self._holding == obj_part
                    and obj_part in self._objects):
                self._objects[obj_part].is_clean = True

        done = False
        reward = 0.0
        if _check_success(self._objects, self._task, self._holding, self._robot_room):
            done = True
            reward = 1.0
        elif self._step >= self.max_steps:
            done = True

        obs = _render_obs(
            self._objects, self._robot_room, self._holding,
            self._step, self.max_steps, self._task,
        )
        return obs, reward, done

    @staticmethod
    def make_prompt(obs: str) -> str:
        """Wrap an observation in the two-shot instruction prompt."""
        return PROMPT_TEMPLATE.format(observation=obs)

    _ACTION_RE = re.compile(r"action\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)

    @classmethod
    def extract_action(cls, text: str) -> Optional[str]:
        """Extract the last ``Action: ...`` line from a model's output string."""
        last = None
        for m in cls._ACTION_RE.finditer(text):
            last = m.group(1).strip()
        if last is None:
            lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
            if lines:
                last = lines[-1]
        return last.strip().strip("\"'`").lower() if last else None

    @staticmethod
    def score(pred: Optional[str], gold: str) -> float:
        """Return 1.0 if pred matches gold (case-insensitive), else 0.0."""
        if pred is None:
            return 0.0
        return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0
