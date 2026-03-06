import random
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import gymnasium as gym
import ray
import requests
import yaml
from dotenv import load_dotenv
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

load_dotenv("../../.env")


@dataclass
class GameResult:
    task_id: str
    run_id: int
    experiment_name: str
    success: bool
    steps: int
    reward: float
    trajectory: List[Dict]
    map_config: Dict[str, Any]


@ray.remote
class FrozenLakeReactAgent:
    """A ReAct Agent for FrozenLake game with task memory learning."""

    def __init__(
        self,
        index: int,
        task_configs: List[Dict],
        experiment_name: str,
        model_name: str = "qwen3-8b",
        temperature: float = 0.7,
        max_steps: int = 50,
        num_runs: int = 1,
        use_task_memory: bool = False,
        make_task_memory: bool = False,
    ):

        self.index = index
        self.task_configs = task_configs
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.temperature = temperature
        self.max_steps = max_steps
        self.num_runs = num_runs
        self.use_task_memory = use_task_memory
        self.make_task_memory = make_task_memory

        self.llm_client = OpenAI()
        self.action_map = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

        # Load prompts
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from yaml file"""
        try:
            with open("frozenlake_prompts.yaml", "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Prompt file not found, using default prompts")
            raise FileNotFoundError(
                "Prompt file not found. Please check your current path (should be ./cook/frozenlake) and try again.",
            )

    def call_llm(self, messages: List[Dict]) -> str:
        """Call LLM with retry logic"""
        for i in range(5):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    extra_body={"enable_thinking": False},
                    seed=0,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {i + 1}): {e}")
                time.sleep(1 + i * 2)
        return "LLM call failed"

    def observe_state(self, env, observation: int) -> str:
        """Convert environment observation to text description"""
        desc = env.unwrapped.desc
        nrow, ncol = desc.shape

        # Convert to string grid
        grid = [[cell.decode("utf-8") for cell in row] for row in desc]

        # Get current position
        row, col = observation // ncol, observation % ncol

        # Create visual representation
        state_text = "Current State:\n"
        for i in range(nrow):
            for j in range(ncol):
                if i == row and j == col:
                    state_text += f"[{grid[i][j]}]"
                else:
                    state_text += f" {grid[i][j]} "
            state_text += "\n"

        state_text += "\nLegend: S=Start, F=Frozen, H=Hole, G=Goal, []=Your Position"
        return state_text

    def build_system_prompt(self, is_slippery: bool) -> str:
        """Build system prompt based on game configuration"""
        if is_slippery:
            return self.prompts["frozenlake_sys_prompt_slippery"]
        else:
            return self.prompts["frozenlake_sys_prompt_no_slippery"]

    def get_task_memory(self, map_desc: str, is_slippery: bool) -> str:
        """Retrieve relevant task memory from task memory service"""
        if not self.use_task_memory:
            return ""

        try:
            query = f"FrozenLake game map: {map_desc}, slippery: {is_slippery}"
            base_url = "http://0.0.0.0:8002/"
            workspace_id = self.experiment_name

            response = requests.post(
                url=base_url + "retrieve_task_memory",
                json={
                    "workspace_id": workspace_id,
                    "query": query,
                },
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("answer", "")
            else:
                logger.warning(f"Task memory retrieval failed: {response.status_code}")
                return ""

        except Exception as e:
            logger.warning(f"Failed to get task memory: {e}")
            return ""

    def action_parser(self, response: str) -> int:
        """Parse action from LLM response"""
        # Look for {"action":"X"} pattern
        patterns = [
            r'["\']action["\']\s*:\s*["\']([0-3])["\']',
            r'"action"\s*:\s*"([0-3])"',
            r"'action'\s*:\s*'([0-3])'",
            r'\baction["\']?\s*[:=]\s*["\']?([0-3])',
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                action = int(match.group(1))
                if 0 <= action <= 3:
                    return action

        # Random fallback
        action = random.randint(0, 3)
        logger.warning(f"Could not parse action from response, using random: {action}")
        return action

    def run_single_episode(self, task_config: Dict, run_id: int) -> GameResult:
        """Run a single episode of the game"""
        map_size = task_config.get("map_size", 4)
        is_slippery = task_config.get("is_slippery", True)
        map_desc = task_config.get("map_desc", None)

        # Create environment
        env_kwargs = {
            "render_mode": None,
            "is_slippery": is_slippery,
        }

        if map_desc is not None:
            env_kwargs["desc"] = map_desc
        else:
            env_kwargs["desc"] = generate_random_map(size=map_size)

        env = gym.make("FrozenLake-v1", **env_kwargs)

        # Get map description for task memory
        map_str = "\n".join(["".join([cell.decode("utf-8") for cell in row]) for row in env.unwrapped.desc])

        # Build messages
        system_prompt = self.build_system_prompt(is_slippery)
        task_memory = self.get_task_memory(map_str, is_slippery)

        messages = [{"role": "system", "content": system_prompt}]

        if task_memory:
            memory_content = f"Here are some relevant tips from previous successful games:\n\n{task_memory}\n\nUse these tips to help you succeed."
            messages.append({"role": "user", "content": memory_content})
            messages.append(
                {"role": "assistant", "content": "I'll use these tips to navigate the frozen lake successfully."},
            )

        # Initialize game
        observation, info = env.reset()
        trajectory = []

        # Add initial state
        initial_state = self.observe_state(env, observation)
        messages.append({"role": "user", "content": initial_state})

        success = False
        total_reward = 0

        for step in range(self.max_steps):
            # Get action from LLM
            response = self.call_llm(messages)
            logger.info(response)
            action = self.action_parser(response)

            messages.append({"role": "assistant", "content": response})

            # Take action
            next_observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Record trajectory step
            trajectory.append(
                {
                    "step": step,
                    "state": observation,
                    "action": action,
                    "action_name": self.action_map[action],
                    "reward": reward,
                    "next_state": next_observation,
                    "done": done,
                    "llm_response": response,
                },
            )

            if done:
                if terminated and reward > 0:
                    success = True
                    result_msg = f"Success! You reached the goal in {step + 1} steps!"
                else:
                    result_msg = f"Game over! You fell into a hole or ran out of time."

                messages.append({"role": "user", "content": result_msg})
                break
            else:
                # Continue game
                next_state = self.observe_state(env, next_observation)
                step_msg = f"Step {step + 1}: You moved {self.action_map[action]}. Reward: {reward}\n{next_state}"
                messages.append({"role": "user", "content": step_msg})
                observation = next_observation

        env.close()

        # Create result
        map_id = task_config.get("map_id", f"unknown_{self.index}_{run_id}")
        task_id = f"{task_config.get('task_type', 'test')}_map{map_id}_{run_id}"
        result = GameResult(
            task_id=task_id,
            run_id=run_id,
            experiment_name=self.experiment_name,
            success=success,
            steps=len(trajectory),
            reward=total_reward,
            trajectory=trajectory,
            map_config={
                "map_desc": map_str,
                "map_id": map_id,
                "is_slippery": is_slippery,
                "map_size": map_size,
                "use_task_memory": self.use_task_memory,
            },
        )

        return result, messages

    def save_task_memory(self, results: List[GameResult], messages_list: List[List[Dict]]):
        """Save successful trajectories as task memory"""
        if not self.make_task_memory:
            return

        trajectories = []
        for result, messages in zip(results, messages_list):
            if result.success:
                # Create trajectory for task memory service
                traj = {
                    "messages": messages,
                    "score": 1.0,  # Success
                }
                trajectories.append(traj)
            else:
                traj = {
                    "messages": messages,
                    "score": 0.0,  # Failure
                }
                trajectories.append(traj)

        if trajectories:
            try:
                base_url = "http://0.0.0.0:8002/"
                workspace_id = self.experiment_name

                response = requests.post(
                    url=base_url + "summary_task_memory",
                    json={
                        "workspace_id": workspace_id,
                        "trajectories": trajectories,
                    },
                    timeout=300,
                )

                if response.status_code == 200:
                    logger.info(f"Saved {len(trajectories)} trajectories as task memory")
                else:
                    logger.warning(f"Failed to save task memory: {response.status_code}")

            except Exception as e:
                logger.error(f"Error saving task memory: {e}")

    def execute(self) -> List[Dict]:
        """Execute all tasks"""
        all_results = []
        all_messages = []

        for task_index, task_config in tqdm(enumerate(self.task_configs), desc="Processing tasks:"):
            for run_id in range(self.num_runs):
                logger.info(f"Ray {self.index}, Task {task_index}, Run {run_id}")

                result, messages = self.run_single_episode(task_config, run_id)
                all_results.append(result)
                all_messages.append(messages)

                # Convert result to dict for JSON serialization
                result_dict = {
                    "task_id": result.task_id,
                    "run_id": result.run_id,
                    "experiment_name": result.experiment_name,
                    "task_completed": result.success,
                    "success": result.success,
                    "steps": result.steps,
                    "reward": result.reward,
                    "map_config": result.map_config,
                    "trajectory": result.trajectory,
                }
                all_results[-1] = result_dict

        # Save task memory if needed
        if self.make_task_memory:
            # Convert back to GameResult objects for task memory saving
            game_results = []
            for i, result_dict in enumerate(all_results):
                game_result = GameResult(
                    task_id=result_dict["task_id"],
                    run_id=result_dict["run_id"],
                    experiment_name=result_dict["experiment_name"],
                    success=result_dict["success"],
                    steps=result_dict["steps"],
                    reward=result_dict["reward"],
                    trajectory=result_dict["trajectory"],
                    map_config=result_dict["map_config"],
                )
                game_results.append(game_result)

            self.save_task_memory(game_results, all_messages)

        return all_results
