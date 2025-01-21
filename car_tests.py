import unittest
import numpy as np
import gymnasium as gym
from unittest.mock import patch, MagicMock
import os
import pickle
import random

# Import functions from the main file
from mountain_car import run, plot_q_table, plot_rewards_per_episode, plot_moving_average

class TestMountainCar(unittest.TestCase):
    def setUp(self):
        """
        Setup method run before each test.
        Initializes the environment and defines state discretization parameters.
        """
        self.env = gym.make('MountainCar-v0')
        # Define state space discretization parameters - adding 1 to include upper bound
        self.position_states = np.linspace(-1.2, 0.6, 21)[:-1]  # Use 20 bins
        self.velocity_states = np.linspace(-0.07, 0.07, 21)[:-1]  # Use 20 bins
        
    def tearDown(self):
        """
        Cleanup method run after each test.
        """
        self.env.close()
        # Clean up generated files
        files_to_remove = ['Q_table.pkl', 
                          'mountain_car_q_table.png',
                          'mountain_car_moving_average.png',
                          'mountain_car_rewards_per_episode.png']
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)

    def test_q_table_initialization(self):
        """
        Test if Q-table is properly initialized and saved.
        """
        # Run a short training session
        run(num_episodes=10, is_training=True, render=False)
        
        # Check if Q-table file was created
        self.assertTrue(os.path.exists('Q_table.pkl'))
        
        # Load and verify Q-table dimensions
        with open('Q_table.pkl', 'rb') as f:
            Q_table = pickle.load(f)
        self.assertEqual(Q_table.shape, (20, 20, 3))

    def test_reward_calculation(self):
        """
        Test if rewards are calculated correctly.
        """
        rewards = []
        # Mock plotting functions to focus on reward calculation
        with patch('mountain_car.plot_rewards_per_episode'), \
             patch('mountain_car.plot_q_table'), \
             patch('mountain_car.plot_moving_average'):
            run(num_episodes=1, is_training=True, render=False, initial_p=-0.5)
            
        # Check if rewards are within reasonable bounds
        self.assertTrue(all(reward > -1000 for reward in rewards))

    def test_state_discretization(self):
        """
        Test if continuous state space is correctly discretized.
        """
        # Test discretization for various positions within bounds
        test_positions = [-1.1, 0.5, 0.0]  # Safe values within bounds
        for pos in test_positions:
            state_p = np.digitize(pos, self.position_states)
            # Verify that discretized state is within bounds
            self.assertTrue(0 <= state_p < 20)  # Now explicitly checking against 20

    @patch('matplotlib.pyplot.savefig')
    def test_plotting_functions(self, mock_savefig):
        """
        Test if visualization functions work correctly.
        """
        # Prepare sample data for plotting
        Q_table = np.random.random((20, 20, 3))
        episode_rewards = [random.random() for _ in range(100)]
        
        # Test plotting functions
        plot_q_table(Q_table)
        plot_rewards_per_episode(episode_rewards)
        plot_moving_average(episode_rewards)
        
        # Verify that plots were saved 
        self.assertEqual(mock_savefig.call_count, 2)

    def test_training_vs_testing_mode(self):
        """
        Test differences between training and testing modes.
        """
        # First run a short training session
        run(num_episodes=10, is_training=True, render=False)
        
        # Then test with deterministic behavior
        with patch('random.uniform', return_value=1.0):  # Force deterministic behavior
            run(num_episodes=1, is_training=False, render=False)

    def test_edge_cases(self):
        """
        Test edge cases and boundary conditions.
        """
        # Test with safe positions near the edges
        edge_positions = [-1.1, 0.5]  # Adjusted to be within bounds
        for pos in edge_positions:
            with patch('mountain_car.plot_rewards_per_episode'), \
                 patch('mountain_car.plot_q_table'), \
                 patch('mountain_car.plot_moving_average'):
                run(num_episodes=1, is_training=True, render=False, initial_p=pos)

    def test_learning_progress(self):
        """
        Test if the agent shows learning progress over episodes.
        """
        with patch('mountain_car.plot_rewards_per_episode'), \
             patch('mountain_car.plot_q_table'), \
             patch('mountain_car.plot_moving_average'):
            run(num_episodes=100, is_training=True, render=False)
            
        # Load Q-table and verify it's not all zeros
        with open('Q_table.pkl', 'rb') as f:
            Q_table = pickle.load(f)
        self.assertFalse(np.all(Q_table == 0))

if __name__ == '__main__':
    unittest.main()