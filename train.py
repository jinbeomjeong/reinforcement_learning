"""
Train a DQN (Deep Q-Network) agent on CartPole.

The training loop uses shaped reward consistently for:
1. DQN updates
2. best-model saving
3. success detection
4. training plots
"""

import os
os.environ["KERAS_BACKEND"] = "torch"  # must be set before importing keras

from collections import deque
from pathlib import Path
import random

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import keras


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
RESULT_IMAGE_PATH = OUTPUT_DIR / "dqn_cartpole_result.png"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.keras"

# Shaped reward threshold for saving and success detection.
SAVE_REWARD_THRESHOLD = 400

# if matplotlib.get_backend().lower() == "agg":
#     matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
#     matplotlib.rcParams["axes.unicode_minus"] = False


# Fixed seeds for reproducible runs.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


# Hyperparameters.
EPISODES = 1000
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 10_000
TARGET_UPDATE = 10
HIDDEN_SIZE = 256

# Position-based reward shaping.
# A small penalty is subtracted as the cart moves away from the center.
POSITION_PENALTY_COEFF = 0.8
CART_POS_LIMIT = 2.4

# Render a live episode every N training episodes (0 = disabled).
RENDER_EVERY = 0


def build_model(state_size: int, action_size: int) -> keras.Model:
    """Small MLP that predicts Q-values for each CartPole action.

    With KERAS_BACKEND=torch the returned model is also a torch.nn.Module,
    so model.parameters(), torch.optim, and autograd all work as usual.
    """
    return keras.Sequential([
        keras.layers.Input(shape=(state_size,)),
        keras.layers.Dense(HIDDEN_SIZE, activation="gelu"),
        keras.layers.Dense(HIDDEN_SIZE, activation="gelu"),
        keras.layers.Dense(action_size),
    ])


def model_device(model: nn.Module) -> torch.device:
    """Return the device of the first trainable parameter."""
    return next(model.parameters()).device


class ReplayBuffer:
    """Stores past transitions and samples random mini-batches."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Owns the policy network, target network, optimizer, and replay buffer."""

    def __init__(self, state_size: int, action_size: int):
        self.action_size = action_size
        self.epsilon = EPSILON_START

        self.policy_net = build_model(state_size, action_size).to(DEVICE)
        self.target_net = build_model(state_size, action_size).to(DEVICE)
        self.target_net.set_weights(self.policy_net.get_weights())

        # Keras+torch model is a torch.nn.Module → use torch optimizer directly.
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, state) -> int:
        """Select an action using epsilon-greedy exploration."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return int(self.policy_net(state_t).argmax().item())

    def learn(self) -> None:
        """Update the Q-network from one sampled replay mini-batch."""
        if len(self.buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        states_t      = torch.as_tensor(states,      device=DEVICE)
        actions_t     = torch.as_tensor(actions,     dtype=torch.long, device=DEVICE)
        rewards_t     = torch.as_tensor(rewards,     device=DEVICE)
        next_states_t = torch.as_tensor(next_states, device=DEVICE)
        dones_t       = torch.as_tensor(dones,       device=DEVICE)

        with torch.no_grad():
            next_q   = self.target_net(next_states_t).max(dim=1).values
            target_q = rewards_t + GAMMA * next_q * (1 - dones_t)

        self.optimizer.zero_grad()
        q_values  = self.policy_net(states_t)
        current_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        loss      = nn.HuberLoss()(current_q, target_q)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target(self) -> None:
        self.target_net.set_weights(self.policy_net.get_weights())

    def decay_epsilon(self) -> None:
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


def calculate_shaped_reward(env_reward: float, next_state) -> float:
    """Return environment reward minus a cart-position penalty."""
    pos_ratio = abs(next_state[0]) / CART_POS_LIMIT
    return float(env_reward - POSITION_PENALTY_COEFF * (pos_ratio ** 2))


def render_episode(agent: "DQNAgent") -> float:
    """Run one greedy episode with a human-render window to visualise current policy."""
    render_env = gym.make("CartPole-v1", render_mode="human")
    state, _ = render_env.reset()
    total_reward = 0.0
    for _ in range(500):
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action = int(agent.policy_net(state_t).argmax().item())
        state, reward, terminated, truncated, _ = render_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    render_env.close()
    return total_reward


def train():
    env = gym.make("CartPole-v1")
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)

    agent = DQNAgent(state_size, action_size)
    print(f"device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"policy_net device: {model_device(agent.policy_net)}")
    print(f"target_net device: {model_device(agent.target_net)}")

    rewards_history = []
    avg_history = []
    best_episode_reward = 0

    print("=" * 55)
    print("  DQN CartPole training started")
    print("=" * 55)

    for episode in range(1, EPISODES + 1):
        state, _ = env.reset(seed=episode)
        total_env_reward = 0
        total_shaped_reward = 0.0

        for _ in range(500):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            shaped_reward = calculate_shaped_reward(reward, next_state)
            agent.buffer.push(state, action, shaped_reward, next_state, float(done))
            agent.learn()

            state = next_state
            total_env_reward += reward
            total_shaped_reward += shaped_reward

            if done:
                break

        agent.decay_epsilon()
        if episode % TARGET_UPDATE == 0:
            agent.update_target()

        rewards_history.append(total_shaped_reward)
        avg = np.mean(rewards_history[-50:])
        avg_history.append(avg)

        if total_shaped_reward >= SAVE_REWARD_THRESHOLD and total_shaped_reward > best_episode_reward:
            best_episode_reward = total_shaped_reward
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            agent.policy_net.save(BEST_MODEL_PATH)
            print(
                f"  [model saved] episode {episode:4d} | "
                f"shaped: {total_shaped_reward:.1f} | "
                f"env: {total_env_reward:.0f} -> {BEST_MODEL_PATH}"
            )

        if episode % 50 == 0:
            print(
                f"  episode {episode:4d} | "
                f"shaped: {total_shaped_reward:6.1f} | "
                f"env: {total_env_reward:6.1f} | "
                f"avg(50): {avg:6.1f} | "
                f"epsilon: {agent.epsilon:.3f}"
            )

        if RENDER_EVERY and episode % RENDER_EVERY == 0:
            vis_reward = render_episode(agent)
            print(f"  [render] episode {episode:4d} | greedy env reward: {vis_reward:.0f}")

        # if avg >= SAVE_REWARD_THRESHOLD and episode >= 50:
        #     print(f"\n  [success] episode {episode} | shaped avg: {avg:.1f}")
        #     break

    env.close()
    return rewards_history, avg_history, agent


def plot_results(rewards, avgs) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f0f")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaaaaa")
        ax.spines["bottom"].set_color("#333355")
        ax.spines["left"].set_color("#333355")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    episodes = range(1, len(rewards) + 1)

    axes[0].fill_between(episodes, rewards, alpha=0.3, color="#4ecdc4")
    axes[0].plot(episodes, rewards, color="#4ecdc4", linewidth=0.8, alpha=0.7)
    axes[0].plot(episodes, avgs, color="#ff6b6b", linewidth=2.0, label="50 episode average")
    axes[0].axhline(
        y=SAVE_REWARD_THRESHOLD,
        color="#ffd93d",
        linestyle="--",
        linewidth=1.2,
        label=f"success threshold ({SAVE_REWARD_THRESHOLD})",
    )
    axes[0].set_xlabel("Episode", color="#aaaaaa", fontsize=11)
    axes[0].set_ylabel("Shaped reward", color="#aaaaaa", fontsize=11)
    axes[0].set_title("Training Curve", color="white", fontsize=13, fontweight="bold")
    axes[0].legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white")

    axes[1].plot(episodes, avgs, color="#ff6b6b", linewidth=2.5)
    axes[1].fill_between(episodes, avgs, alpha=0.2, color="#ff6b6b")
    axes[1].axhline(
        y=SAVE_REWARD_THRESHOLD,
        color="#ffd93d",
        linestyle="--",
        linewidth=1.2,
        label=f"success threshold ({SAVE_REWARD_THRESHOLD})",
    )
    axes[1].set_xlabel("Episode", color="#aaaaaa", fontsize=11)
    axes[1].set_ylabel("Average shaped reward", color="#aaaaaa", fontsize=11)
    axes[1].set_title("Moving Average (50 Episodes)", color="white", fontsize=13, fontweight="bold")
    axes[1].legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white")

    last_avg = avgs[-1]
    axes[1].annotate(
        f"final avg: {last_avg:.1f}",
        xy=(len(avgs), last_avg),
        xytext=(-80, 15),
        textcoords="offset points",
        color="white",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#aaaaaa"),
    )

    plt.suptitle("DQN CartPole Training Result", color="white", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULT_IMAGE_PATH, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"\n  [saved] training result plot: {RESULT_IMAGE_PATH}")


if __name__ == "__main__":
    rewards, avgs, agent = train()
    plot_results(rewards, avgs)

    print("\n" + "=" * 55)
    print(f"  total training episodes : {len(rewards)}")
    print(f"  best shaped reward      : {max(rewards):.1f}")
    print(f"  final shaped avg        : {avgs[-1]:.1f}")
    print(f"  final epsilon           : {agent.epsilon:.3f}")
    if BEST_MODEL_PATH.exists():
        print(f"  model path              : {BEST_MODEL_PATH}")
    print("=" * 55)
