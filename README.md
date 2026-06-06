# Deep Reinforcement Learning — Autonomous Vehicle Navigation

**Authors:** Nasos Karas, Taha Choksi  
**Programme:** MSc Artificial Intelligence, City, University of London  
**Date:** May 2026  

---

## Overview

This project explores Deep Reinforcement Learning (DRL) for autonomous vehicle navigation. It progresses from a custom tabular Q-learning environment to DQN variants and PPO on a complex highway simulation.

---

## Environments

### 1. GridCarEnv (Custom)
A fully parametrizable 6×6 grid-world simulating an autonomous car navigating from `(0,0)` to `(5,5)`.

- **State space:** 32 valid positions (36 cells minus 4 barriers)
- **Action space:** `{up, down, left, right}`
- **Obstacles:** Barriers (impassable) and pitfalls (traversable with penalty)
- **Episode limit:** 30 steps

**Reward structure:**

| Event | Reward |
|---|---|
| Reach goal | +100 |
| Each step | −1 |
| Hit barrier | −10 |
| Enter pitfall | −5 |
| Move out of bounds | −20 |

### 2. Highway-v0 (highway-env)
A multi-lane highway environment used for DQN and PPO experiments.

- **Observation space:** 5×5 matrix (5 nearest vehicles × position/velocity features)
- **Action space:** 5 discrete actions (lane changes, accelerate, decelerate)
- **Reward:** Positive for high speed; penalties for collisions and going off-road

---

## Algorithms

### Q-Learning (GridCarEnv)

Tabular Q-learning with a 36×4 Q-table, initialised to zero.

**Policies compared:**
- **ε-greedy** — random exploration with probability ε, greedy otherwise
- **Softmax** — temperature-weighted probabilistic action selection

**Hyperparameters tuned:**

| Parameter | Values tested |
|---|---|
| Learning rate α | 0.1, 0.3, 0.9 |
| Discount factor γ | 0.5, 0.9, 0.99 |
| Decay | 0.99, 0.995, 0.999 |

**Best configurations:**

| Policy | α | γ | Decay | Avg. Reward | Success Rate |
|---|---|---|---|---|---|
| Softmax | 0.9 | 0.9 | 0.99 | 89.61 | 99.80% |
| ε-greedy | 0.9 | 0.9 | 0.9 | 88.84 | 99.40% |

Baseline ε-greedy (α=0.1, γ=0.1, decay=0.995) achieved a success rate of **96.56%**.

---

### DQN Variants (Highway-v0)

All DQN models use a simple Q-network: `Input(25) → Linear(64) → ReLU → Linear(5)`  
Trained for **3,000 episodes** on GPU.

| Hyperparameter | Value |
|---|---|
| Learning rate | 0.0001 |
| Discount factor γ | 0.99 |
| Initial ε | 1.0 |
| ε decay | 0.995 |
| Target update frequency | Every 10 episodes |

**Results:**

| Model | Avg. Reward (last 100 eps.) | Max Reward | Avg. Loss (last 100 eps.) |
|---|---|---|---|
| Vanilla DQN | 27.68 | 33.68 | 175.22 |
| Target Network DQN | 27.76 | 32.48 | 78.58 |
| **Double DQN** | **28.56** | **34.35** | **18.59** |

- **Target Network** addresses training instability by freezing target weights and updating every N episodes.
- **Double DQN** addresses Q-value overestimation by decoupling action selection (policy network) from action evaluation (target network).

---

### PPO (Highway-v0)

Actor-critic architecture trained for **1,000 episodes**.

**Architecture:** Two hidden layers (64 units, Tanh activation) for both actor and critic.

| Hyperparameter | Value |
|---|---|
| Learning rate | 3e-4 |
| Clipping parameter ε | 0.2 |
| Discount factor γ | 0.99 |
| Episodes | 1,000 |

**Results:** Avg. reward 27.94 | Max reward 36.83 | Avg. loss 23.74 (last 100 episodes)  
PPO achieved comparable rewards to DQN in significantly fewer episodes.

---

## Project Structure

```
ReinforcementLearning/
├── GridCarEnv/
│   ├── grid_env.py          # Custom 6×6 environment
│   ├── q_learning.py        # Tabular Q-learning + ε-greedy/Softmax
│   └── experiments.py       # Hyperparameter sweep & plotting
├── DQN/
│   ├── q_network.py         # QNetwork architecture
│   ├── vanilla_dqn.py       # Vanilla DQN training
│   ├── target_dqn.py        # Target network DQN
│   └── double_dqn.py        # Double DQN
├── PPO/
│   └── ppo_agent.py         # PPO actor-critic agent & training loop
├── checkpoints/             # Saved model weights (.pth)
├── *.json                   # Training history files
└── README.md
```

---

## Dependencies

```bash
pip install numpy pandas matplotlib torch gymnasium highway-env
```

GPU training is supported automatically via PyTorch's CUDA detection.

---

## Running the Code

**Q-Learning baseline:**
```python
from grid_env import GridCarEnv
reward_list, success_rate = train(alpha=0.1, gamma=0.1, epsilon_decay=0.995, episodes=5000)
```

**Hyperparameter sweep:**
```python
results, summary_df = experiments(
    alphas=[0.1, 0.3, 0.9],
    gammas=[0.1, 0.3, 0.9],
    decay_values=[0.9, 0.99, 0.999, 0.9999],
    policies=["epsilon_greedy", "softmax"],
    episodes=5000
)
```

**DQN training:**
```python
train_vanilla(episodes=3000)
train_target_qnetwork(episodes=3000)
train_double_dqn(episodes=3000)
```

**PPO training:**
```python
agent, rewards, losses = train_ppo(episodes=1000)
```

---

## Key Findings

- Softmax policy consistently outperformed ε-greedy in the grid world due to more controlled exploration.
- High α (0.9) and γ (0.9) worked best for the small 6×6 grid.
- Double DQN produced the best stability and reward in the highway environment, with loss nearly 10× lower than Vanilla DQN.
- PPO matched DQN performance in one-third of the training episodes, suggesting superior sample efficiency.

---

## References

- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540).
- Hessel et al. (2018). Rainbow: Combining improvements in deep reinforcement learning. *AAAI*.
- Lv et al. (2019). Stochastic double deep Q-network. *IEEE Access*, 7.
- Schulman et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Roderick et al. (2017). Implementing the deep Q-network. *arXiv:1711.07478*.
