# Flappy Bird Agents

A compact reinforcement-learning project that compares four ways to play
Flappy Bird from the environment's 12 numerical state values:

- a model-based beam-search planner;
- a Double DQN trained from scratch;
- a PPO baseline trained from scratch;
- an improved PPO policy initialized from planner demonstrations.

The project uses
[`flappy-bird-gymnasium`](https://github.com/markub3327/flappy-bird-gymnasium)
with `use_lidar=False`. No agent reads screen pixels or the 180-value LIDAR
observation.

## Results

| Agent | Learning method | Episodes | Score cap | Mean | Median | Maximum |
|---|---|---:|---:|---:|---:|---:|
| Model-based planner | Beam search, no training | 5 | 1,100 | **810.60** | **1,100** | **1,100** |
| Scratch Double DQN | Replay-based RL | 100 | 200 | 33.73 | 25 | 158 |
| Improved PPO | Planner imitation, then PPO | 50 | 200 | 6.76 | 4 | 38 |
| Original PPO | PPO from scratch | 50 | 100 | 5.30 | 4 | 28 |

The benchmark protocols are shown because the planner and learned agents were
evaluated with different score limits. Scores are measured on seeded evaluation
episodes, and neural-network results can vary when training is repeated.

## The 12 input values

The compact state contains three values for each of three pipes:

1. horizontal position;
2. top of the opening;
3. bottom of the opening.

The final three values are the bird's vertical position, vertical velocity, and
rotation. Explicit vertical velocity makes this compact state suitable for a
feed-forward neural network.

## Installation

The project was tested with Python 3.13 and an NVIDIA GPU using CUDA 12.6.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

The game window is disabled during training for speed and enabled only in play
mode.

## Run the agents

### Model-based planner

```powershell
python planner_agent.py
python planner_agent.py evaluate --episodes 5 --score-limit 1100
```

### Scratch Double DQN

The best included DQN used 80% idle and 20% flap whenever it selected a random
exploration action.

```powershell
python dqn_agent.py play
python dqn_agent.py evaluate --episodes 100 --score-limit 200
```

Train a new 80/20 model from randomly initialized weights:

```powershell
python dqn_agent.py train --flap-probability 0.2 --steps 300000 --fresh
```

Repeat the complete exploration experiment:

```powershell
python dqn_agent.py experiment --flap-probabilities 0.1 0.2 0.3 --steps 300000
```

The three independent 300,000-step runs produced:

| Random idle/flap mix | Mean | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|
| 90% / 10% | 35.33 | 24.0 | 4 | 97 |
| **80% / 20%** | **37.90** | **28.5** | 2 | **137** |
| 70% / 30% | 3.57 | 2.0 | 0 | 19 |

All three were evaluated on the same 30 unseen seeds. A 50/50 random policy was
intentionally excluded because excessive random flapping produces poor
exploration trajectories.

### Improved PPO

The improved PPO first learns to imitate planner demonstrations and is then
fine-tuned with reinforcement learning.

```powershell
python ppo_agent.py play
python ppo_agent.py evaluate --episodes 50 --score-limit 200
python ppo_agent.py train --steps 500000
```

### Original PPO baseline

```powershell
python main.py play
python main.py evaluate --episodes 50
python main.py train --steps 500000
```

## How the agents work

### Planner

The planner reproduces the deterministic bird and pipe physics, simulates both
actions into the future, rejects collisions, and keeps the best 200 candidate
states at each step. It replans when new pipe information becomes visible. This
is the strongest agent because the physics are known and the action space has
only two choices.

### Double DQN

The DQN starts with random weights and an empty replay memory. Its 256-by-256
network estimates the future value of idle and flap. The current network picks
the next action while a target network evaluates it, reducing the optimistic
errors of ordinary DQN.

The observation is still 12 values, but pipes are reordered so the upcoming one
comes first and pipe positions are represented relative to the bird. Training
uses a 100,000-transition replay buffer, batches of 128, and dense reward
feedback for survival, gap alignment, passed pipes, and collisions.

### PPO

PPO directly learns action probabilities and a state-value estimate. The
original version learns only from interaction. The improved version first uses
supervised imitation to copy planner decisions before PPO fine-tuning.

See [`METHOD_EXPLANATION.txt`](METHOD_EXPLANATION.txt) for the complete method,
hyperparameters, reward design, and results for every agent.

## Project structure

```text
.
|-- compact_planner.py        Beam-search controller
|-- planner_agent.py          Planner play and evaluation commands
|-- dqn_agent.py              Biased-exploration Double DQN
|-- ppo_agent.py              Imitation-pretrained PPO
|-- main.py                   Original PPO baseline
|-- game_env.py               Shared compact environment setup
|-- METHOD_EXPLANATION.txt    Detailed explanation of every agent
|-- requirements.txt          Python dependencies
`-- models/                   Included trained checkpoints
```

## Included checkpoints

- `dqn_scratch_best.zip`: winning 80/20 Double DQN;
- `dqn_scratch_flap_10.zip`, `20.zip`, and `30.zip`: exploration experiments;
- `compact_ppo.zip`: original PPO;
- `compact_ppo_pretrained.zip`: imitation checkpoint;
- `compact_ppo_improved.zip`: fine-tuned PPO.

