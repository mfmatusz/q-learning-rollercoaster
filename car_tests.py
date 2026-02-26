"""Tests for the MountainCar Q-learning agent."""

import matplotlib
matplotlib.use("Agg")  # headless backend – must be set before any other matplotlib import

import os
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from mountain_car import (
    NUM_BINS,
    POSITION_BINS,
    VELOCITY_BINS,
    discretize,
    plot_moving_average,
    plot_q_table,
    plot_rewards,
    run,
)

_GENERATED_FILES = [
    "Q_table.pkl",
    "mountain_car_q_table.png",
    "mountain_car_moving_average.png",
    "mountain_car_rewards_per_episode.png",
]


class TestDiscretize(unittest.TestCase):
    """Unit tests for the discretize() helper."""

    def test_typical_values_in_range(self):
        """Indices for in-range states must fall within [0, NUM_BINS)."""
        for pos, vel in [(-1.0, 0.0), (0.0, 0.03), (0.5, -0.05)]:
            p, v = discretize(np.array([pos, vel]))
            self.assertGreaterEqual(p, 0)
            self.assertLess(p, NUM_BINS)
            self.assertGreaterEqual(v, 0)
            self.assertLess(v, NUM_BINS)

    def test_boundary_clipping(self):
        """Values at or beyond grid edges must be clipped, not raise IndexError."""
        # Exact upper boundary – np.digitize returns NUM_BINS without clipping
        p, v = discretize(np.array([0.6, 0.07]))
        self.assertEqual(p, NUM_BINS - 1)
        self.assertEqual(v, NUM_BINS - 1)

        # Below lower boundary
        p, v = discretize(np.array([-1.2, -0.07]))
        self.assertGreaterEqual(p, 0)
        self.assertGreaterEqual(v, 0)

    def test_returns_valid_q_table_indices(self):
        """Indices must always be valid for a Q-table of shape (NUM_BINS, NUM_BINS, 3)."""
        q_table = np.zeros((NUM_BINS, NUM_BINS, 3))
        test_states = [
            np.array([-1.2, -0.07]),
            np.array([0.6, 0.07]),
            np.array([0.0, 0.0]),
        ]
        for state in test_states:
            p, v = discretize(state)
            # Should not raise IndexError
            _ = q_table[p, v, :]


class TestTraining(unittest.TestCase):
    """Integration tests for the training loop."""

    def tearDown(self):
        for path in _GENERATED_FILES:
            if os.path.exists(path):
                os.remove(path)

    def test_q_table_saved_with_correct_shape(self):
        """Training must create a Q-table file with shape (NUM_BINS, NUM_BINS, 3)."""
        run(num_episodes=10, is_training=True, render=False)

        self.assertTrue(os.path.exists("Q_table.pkl"))
        with open("Q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
        self.assertEqual(q_table.shape, (NUM_BINS, NUM_BINS, 3))

    def test_q_table_updated_after_training(self):
        """Q-table should contain non-zero values after a training run."""
        with patch("mountain_car.plot_q_table"), \
             patch("mountain_car.plot_rewards"), \
             patch("mountain_car.plot_moving_average"):
            run(num_episodes=50, is_training=True, render=False)

        with open("Q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
        self.assertFalse(np.all(q_table == 0))

    def test_episode_rewards_within_bounds(self):
        """Each episode reward must be at or above the early-exit threshold."""
        with patch("mountain_car.plot_q_table"), \
             patch("mountain_car.plot_rewards"), \
             patch("mountain_car.plot_moving_average"):
            rewards = run(num_episodes=5, is_training=True, render=False, initial_p=-0.5)

        self.assertEqual(len(rewards), 5)
        self.assertTrue(all(r >= -1000 for r in rewards))

    def test_fixed_start_position(self):
        """Agent must accept a fixed initial position without error."""
        with patch("mountain_car.plot_q_table"), \
             patch("mountain_car.plot_rewards"), \
             patch("mountain_car.plot_moving_average"):
            run(num_episodes=3, is_training=True, render=False, initial_p=-0.5)

    def test_training_then_evaluation(self):
        """A model saved during training must load cleanly for evaluation."""
        with patch("mountain_car.plot_q_table"), \
             patch("mountain_car.plot_rewards"), \
             patch("mountain_car.plot_moving_average"):
            run(num_episodes=10, is_training=True, render=False)
            run(num_episodes=2, is_training=False, render=False)


class TestPlotHelpers(unittest.TestCase):
    """Tests for visualisation helpers."""

    def tearDown(self):
        for path in _GENERATED_FILES:
            if os.path.exists(path):
                os.remove(path)

    def test_plot_functions_create_files(self):
        """Each plot helper must create its output PNG without raising."""
        q_table = np.random.random((NUM_BINS, NUM_BINS, 3))
        rewards = list(range(-200, 0))

        plot_q_table(q_table)
        plot_rewards(rewards)
        plot_moving_average(rewards)

        self.assertTrue(os.path.exists("mountain_car_q_table.png"))
        self.assertTrue(os.path.exists("mountain_car_rewards_per_episode.png"))
        self.assertTrue(os.path.exists("mountain_car_moving_average.png"))

    def test_moving_average_shorter_than_window(self):
        """plot_moving_average must not crash when rewards < window size."""
        plot_moving_average([-200, -180, -150], window=100)
        self.assertTrue(os.path.exists("mountain_car_moving_average.png"))


if __name__ == "__main__":
    unittest.main()