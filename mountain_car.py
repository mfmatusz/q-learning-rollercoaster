import random
import gymnasium as gym
import numpy as np

def run(num_episodes):
    env = gym.make('MountainCar-v0', render_mode='human')
    env.metadata['render_fps']= 0
    position_states = np.linspace(-1.2, 0.6, 20)
    velocity_states = np.linspace(-0.07, 0.07, 20)
    Q_table = np.zeros((len(position_states), len(velocity_states), 3))  # initialize Q-table with zeros

    learning_rate = 0.1
    discount_rate = 0.9
    epsilon = 1.0  # Start with full exploration

    for episode in range(num_episodes):
        state = env.reset()[0]
        state_p = np.digitize(state[0], position_states)
        state_v = np.digitize(state[1], velocity_states)

        total_reward = 0
        done = False

        while not done:  # Run until the environment signals termination
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state_p, state_v, :])

            next_state, reward, done, _, _ = env.step(action)
            next_state_p = np.digitize(next_state[0], position_states)
            next_state_v = np.digitize(next_state[1], velocity_states)

            # Update Q-table using Q-Learning formula
            Q_table[state_p, state_v, action] += learning_rate * (
                reward + discount_rate * np.max(Q_table[next_state_p, next_state_v, :]) - Q_table[state_p, state_v, action]
            )

            # Move to the next state
            state_p, state_v = next_state_p, next_state_v
            total_reward += reward  # Accumulate reward for the episode

        # Decay epsilon
        epsilon = max(0.01, epsilon - 1 / num_episodes)

        # Log results
        print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

    env.close()

if __name__ == '__main__':
    run(200)
