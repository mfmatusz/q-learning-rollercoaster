# Q-Learning MountainCar

A tabular Q-learning agent that solves the classic
[MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/)
control problem from Gymnasium.

The car starts in a valley and must reach the flag on the right hill.
Because the engine is too weak to drive straight up, the agent must learn
to build momentum by rocking back and forth — a counter-intuitive strategy
that Q-learning discovers on its own through trial and error.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Language |
| [Gymnasium](https://gymnasium.farama.org/) | RL environment |
| NumPy | Q-table and numerical operations |
| Matplotlib / Seaborn | Training visualisations |

---

## How to Run

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Train the agent** (saves `Q_table.pkl` and three PNG plots)

```bash
python mountain_car.py
```

**3. Evaluate a trained agent** (uncomment the last line in `mountain_car.py`)

```python
# mountain_car.py – bottom of file
run(10, is_training=False, render=True, initial_p=-0.5)
```

**4. Run the test suite**

```bash
python -m pytest car_tests.py -v
```

---

## How It Works

### State discretisation
The continuous state space `(position, velocity)` is mapped to a 20 × 20 grid
using `numpy.digitize`. This converts an infinite state space into a finite
Q-table of shape `(20, 20, 3)` — one entry per `(position_bin, velocity_bin, action)`.

### Q-learning update (Bellman equation)

```
Q(s, a) ← Q(s, a) + α · [r + γ · max Q(s', a') − Q(s, a)]
```

| Symbol | Meaning | Value |
|---|---|---|
| α | Learning rate | 0.4 |
| γ | Discount factor | 0.9 |
| ε | Exploration rate | 1.0 → 0.0 (linear decay) |

### Epsilon-greedy exploration
The agent starts by taking random actions (ε = 1) and gradually shifts to
exploiting its learned Q-values as ε decays to 0 over all episodes.

---

## Project Structure

```
q-learning-rollercoaster/
├── mountain_car.py       # Agent, training loop, visualisation helpers
├── car_tests.py          # Unit + integration tests
├── requirements.txt      # Python dependencies
└── README.md
```

---

## What I Learned

- **Q-learning fundamentals** — how the Bellman equation propagates future
  rewards back through a Q-table via temporal-difference learning.
- **State discretisation** — converting a continuous observation space into
  a finite grid, including the boundary-clipping edge case that `np.digitize`
  would otherwise mis-index.
- **Exploration vs. exploitation** — the epsilon-greedy trade-off and the
  effect of the decay schedule on convergence speed.
- **Gymnasium API** — interacting with `step()`, `reset()`, and `action_space`
  in a standard RL environment loop.
- **Visualising RL training** — plotting reward curves and Q-table heatmaps
  to diagnose whether the agent is actually learning.
