# RL-Tetris

<img src="assets/demo_v5.gif" width="500" />

Experimental playground for teaching AI to play Tetris.

The project is a research and learning sandbox focused on planning policies, reinforcement learning, imitation, and representation learning. The codebase is YAML-driven (Hydra + Pydantic) to support fast experimentation.

---

## What is inside

### Learning approaches

This project combines planning, imitation learning, and reinforcement learning for Tetris:

- Heuristic planning policies (Codemy-style linear heuristics with lookahead)
- Genetic Algorithm (GA) optimization of heuristic weights
- Temporal-difference (TD) learning for heuristic weights (GAE-style value updates)
- Imitation learning (behavior cloning from expert data)
- Reinforcement learning (PPO, Maskable PPO)

### Models and representations

Configurable via YAML:

- CNN
- ViT
- MLP-Mixer
- Multiple tokenization/embedding strategies

### Environment

- Custom Rust Tetris engine (PyO3 + maturin)
- Gymnasium-compatible Python envs
- Pygame rendering

---

## Requirements

### Python
- Python >= 3.10
- pip

### System / native
This project includes a Rust-based Tetris engine built via PyO3 + maturin.

You need a working Rust toolchain:
- rustc and cargo (install via rustup)

Windows (MSVC):
- Visual Studio Build Tools
- Desktop development with C++

### Python dependencies
Installed automatically via `pip install -e .` (see `pyproject.toml`).

---

## Installation

```bash
pip install -e .
```

Optional visualization extras:

```bash
pip install -e .[viz]
```

---

## CLI tools

After installation, the following commands are available:

- `tetris-train` - train a model (RL, imitation, GA)
- `tetris-watch` - watch a trained agent play (pygame)
- `tetris-datagen` - generate expert datasets
- `tetris-engine-speedtest` - Rust engine speed test

All training and evaluation is driven via YAML configs.

Example:

```bash
tetris-train -cfg .\configs\ppo\ppo_vit.yaml
```

TD heuristic training:

```bash
tetris-train -cfg .\configs\td\td_heuristic.yaml
```

Config layout (Hydra):

- `configs/<algo>/*.yaml` are entrypoints (GA/TD/PPO/imitation).
- `conf/` holds building blocks: envs, rewards, planning policies, trains, sb3 policies.

Reproducibility:

- `run.seed` seeds Python/NumPy/Torch once per run; env/episode seeds are derived from it.

---

## Project status and goals

This is not a polished framework. It is a personal playground to:

- explore NN architectures
- compare planning vs imitation vs RL
- iterate fast on envs and configs

---

## Disclaimer

Expect:

- fast iteration
- sharp edges
- evolving APIs
