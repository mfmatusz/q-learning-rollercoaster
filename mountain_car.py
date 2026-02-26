"""
Q-Learning agent for the MountainCar-v0 environment (Gymnasium).

The agent discretises the continuous (position, velocity) state space into a
finite grid and learns an optimal policy using the Q-learning update rule with
an epsilon-greedy exploration strategy.
"""

import pickle

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Discretisation grid
# ---------------------------------------------------------------------------
NUM_BINS = 20
POSITION_BINS = np.linspace(-1.2, 0.6, NUM_BINS)
VELOCITY_BINS = np.linspace(-0.07, 0.07, NUM_BINS)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE = 0.4
DISCOUNT_RATE = 0.9
EPSILON_START = 1.0
MIN_TOTAL_REWARD = -1000  # early-exit threshold per episode

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
Q_TABLE_PATH = "Q_table.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discretize(state: np.ndarray) -> tuple:
    """Map a continuous (position, velocity) pair to discrete bin indices.

    np.digitize can return values in [0, NUM_BINS], so we clip to keep
    indices valid for the Q-table shape (NUM_BINS, NUM_BINS, 3).
    """
    pos_idx = int(np.clip(np.digitize(state[0], POSITION_BINS), 0, NUM_BINS - 1))
    vel_idx = int(np.clip(np.digitize(state[1], VELOCITY_BINS), 0, NUM_BINS - 1))
    return pos_idx, vel_idx


# ---------------------------------------------------------------------------
# Core training / evaluation loop
# ---------------------------------------------------------------------------

def run(
    num_episodes: int,
    is_training: bool = True,
    render: bool = False,
    initial_p: float = None,
) -> list:
    """Train or evaluate the Q-learning agent on MountainCar-v0.

    Args:
        num_episodes: Number of episodes to run.
        is_training:  If True, update the Q-table and save it on exit.
        render:       If True, open a graphical window (slow).
        initial_p:    Fixed starting position; random reset if None.

    Returns:
        List of total rewards, one entry per episode.
    """
    env = gym.make("MountainCar-v0", render_mode="human" if render else None)
    env = env.unwrapped  # direct state access needed for initial_p
    env.metadata["render_fps"] = 1

    if is_training:
        q_table = np.zeros((NUM_BINS, NUM_BINS, 3))
    else:
        with open(Q_TABLE_PATH, "rb") as f:
            q_table = pickle.load(f)

    epsilon = EPSILON_START
    episode_rewards = []

    for episode in range(num_episodes):
        if initial_p is not None:
            env.state = np.array([initial_p, 0.0])
        else:
            env.reset()

        pos_idx, vel_idx = discretize(env.state)
        total_reward = 0.0
        done = False

        while not done and total_reward > MIN_TOTAL_REWARD:
            # Epsilon-greedy action selection
            if is_training and np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[pos_idx, vel_idx]))

            next_state, reward, done, _, _ = env.step(action)
            next_pos_idx, next_vel_idx = discretize(next_state)

            if is_training:
                # Bellman update
                q_table[pos_idx, vel_idx, action] += LEARNING_RATE * (
                    reward
                    + DISCOUNT_RATE * np.max(q_table[next_pos_idx, next_vel_idx])
                    - q_table[pos_idx, vel_idx, action]
                )

            pos_idx, vel_idx = next_pos_idx, next_vel_idx
            total_reward += reward

        epsilon = max(epsilon - 2 / num_episodes, 0.0)
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1:>5} | reward: {total_reward:>8.1f} | ε: {epsilon:.3f}")

    env.close()

    if is_training:
        with open(Q_TABLE_PATH, "wb") as f:
            pickle.dump(q_table, f)
        plot_q_table(q_table)

    plot_rewards(episode_rewards)
    plot_moving_average(episode_rewards)
    return episode_rewards


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_q_table(q_table: np.ndarray) -> None:
    """Save a heatmap of the max Q-value at every (position, velocity) state."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(np.max(q_table, axis=2), cmap="viridis", ax=ax)
    ax.set_title("Q-table: max Q-value per state")
    ax.set_xlabel("Velocity bin")
    ax.set_ylabel("Position bin")
    fig.savefig("mountain_car_q_table.png")
    plt.close(fig)


def plot_rewards(episode_rewards: list) -> None:
    """Save a plot of the raw total reward per episode."""
    fig, ax = plt.subplots()
    ax.plot(episode_rewards)
    ax.set_title("Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    fig.savefig("mountain_car_rewards_per_episode.png")
    plt.close(fig)


def plot_moving_average(episode_rewards: list, window: int = 100) -> None:
    """Save a smoothed reward curve using a rolling mean."""
    means = [
        np.mean(episode_rewards[max(0, i - window + 1): i + 1])
        for i in range(len(episode_rewards))
    ]
    fig, ax = plt.subplots()
    ax.plot(means)
    ax.set_title(f"Reward – {window}-episode moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean reward")
    fig.savefig("mountain_car_moving_average.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run(5000, is_training=True, render=False, initial_p=-0.5)
    # To evaluate a trained agent uncomment:
    # run(10, is_training=False, render=True, initial_p=-0.5)

