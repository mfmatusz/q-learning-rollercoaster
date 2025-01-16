import random
import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def run(num_episodes, is_training = True, render = False):
   # Initialize environment
   env = gym.make('MountainCar-v0', render_mode='human' if render else None)
   env.metadata['render_fps']= 60
   position_states = np.linspace(-1.2, 0.6, 20)
   velocity_states = np.linspace(-0.07, 0.07, 20)
   if is_training:
       Q_table = np.zeros((len(position_states), len(velocity_states), 3))  # initialize Q-table with zeros
   else:
       f = open("Q_table.pkl", "rb")
       Q_table = pickle.load(f)
       f.close()

   learning_rate = 0.9
   discount_rate = 0.9
   epsilon = 1  # Start with full exploration

   episode_rewards = []

   for episode in range(num_episodes):
       state = env.reset()[0]
       state_p = np.digitize(state[0], position_states)
       state_v = np.digitize(state[1], velocity_states)

       total_reward = 0
       done = False

       while not done and total_reward > -1000:  # Run until the environment signals termination or the car does too many actions
           # Epsilon-greedy action selection
           if random.uniform(0, 1) < epsilon and is_training:
               action = env.action_space.sample()
           else:
               action = np.argmax(Q_table[state_p, state_v, :])

           next_state, reward, done, _, _ = env.step(action)
           next_state_p = np.digitize(next_state[0], position_states)
           next_state_v = np.digitize(next_state[1], velocity_states)

           if is_training:
           # Update Q-table using Q-Learning formula
               Q_table[state_p, state_v, action] += learning_rate * (
                   reward + discount_rate * np.max(Q_table[next_state_p, next_state_v, :]) - Q_table[state_p, state_v, action]
               )

           # Move to the next state
           state_p, state_v = next_state_p, next_state_v
           total_reward += reward  # Accumulate reward for the episode

       # Decay epsilon
       epsilon = max(epsilon - 2/num_episodes, 0)

       episode_rewards.append(total_reward)

       # Log results
       print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

   env.close()
   
   # Save Q-table and plot
   if is_training:
       f = open("Q_table.pkl", "wb")
       pickle.dump(Q_table, f)
       f.close()
       # Save Q-table visualization
       plot_q_table(Q_table)

   # Save rewards per episode plot
   plot_rewards_per_episode(episode_rewards)

   # Save moving average plot
   plot_moving_average(episode_rewards)

def plot_q_table(Q_table):
   fig, ax = plt.subplots(figsize=(10, 10))
   heatmap = sns.heatmap(np.max(Q_table, axis=2), cmap="viridis", ax=ax)
   ax.set_title("Q-table Visualization")
   ax.set_xlabel("Velocity State")
   ax.set_ylabel("Position State")
   fig = heatmap.get_figure()
   fig.savefig('mountain_car_q_table.png')
   plt.close(fig)  

def plot_moving_average(episode_rewards):
   window_size = 100
   mean_rewards = []
   for i in range(len(episode_rewards)):
       start_idx = max(0, i - window_size + 1)
       mean_rewards.append(np.mean(episode_rewards[start_idx:i+1]))
   plt.plot(mean_rewards)
   plt.title('Mean Rewards per Episode (Moving Average)')
   plt.xlabel('Episode')
   plt.ylabel('Mean Reward')
   plt.savefig('mountain_car_moving_average.png')
   plt.close()

def plot_rewards_per_episode(episode_rewards):
   plt.figure()
   plt.title('Rewards per Episode')
   plt.plot(episode_rewards)
   plt.xlabel('Episode')
   plt.ylabel('Reward')
   plt.savefig('mountain_car_rewards_per_episode.png')
   plt.close()   

if __name__ == '__main__':
    # For training
    run(2000, is_training=True, render=False)
    # For testing (with a trained Q-table)
    #run(10, is_training=False, render=True)
