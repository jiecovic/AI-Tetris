# RL-Tetris 🧩🤖

<img src="assets/demo_v2.gif" width="500" />


**Experimental playground for teaching AI to play Tetris.**

This project is a research / learning sandbox focused on **model-free reinforcement learning**, neural network architectures, and representation learning for classic games.
The codebase is intentionally flexible and YAML-driven, allowing rapid experimentation with environments, models, and training setups.

---

## What’s inside

### 🧠 Learning approaches

* **Pure RL (model-free)** using Stable-Baselines3 (PPO / Maskable PPO / DQN)
* **Behavior Cloning / Policy Distillation**

  * From a heuristic Tetris agent with:

    * depth-0 (greedy)
    * depth-1 lookahead
  * Inspired by:

    * *The Near Perfect Tetris Bot* (CodemyRoad)
* Planned: **RL fine-tuning of BC agents** (annealing, hybrid training)

### 🧱 Models & representations

Configurable entirely via **YAML**:

* CNN
* ViT (Different kind of tokenization methods + transformer layers)
* MLP-Mixer
* Multiple embedding/tokenization strategies:

  * patch / row / column
  * row + column
  * 1D conv (row-wise / column-wise)
  * linear
  * symbolic (for active and next tetromino and for discrete row or columns)
* Components are composable and easy to mix & match

### 🎮 Environment

* Custom Tetris engine
* Gymnasium-compatible, PyGame Rendering
---

## Requirements

### Python
- Python **≥ 3.10**
- `pip`

### System / Native dependencies
This project includes a Rust-based Tetris engine built via **PyO3 + maturin**.

You need a working **Rust toolchain**:
- `rustc` and `cargo`
- Install via **rustup**

**Windows (MSVC)**:
- Visual Studio Build Tools
- Enable *Desktop development with C++*

### Python dependencies
Installed automatically via `pip install -e .` (see `pyproject.toml`).


## Installation

Install locally via `pyproject.toml`:

```bash
pip install -e .
```

Optional visualization extras:

```bash
pip install -e .[viz]
```

Requires **Python ≥ 3.10**.

---

## CLI tools

After installation, the following commands are available:

* `tetris-train` – train an RL or imitation model
* `tetris-watch` – watch a trained agent play (pygame)
* `tetris-gen-expert-data` – generate heuristic expert datasets
* `tetris-fit-reward` – fit reward models from datasets

All training and evaluation is driven via YAML configs.

---

## Project status & goals

This is **not a polished framework** — it’s a personal playground to:

* explore different NN architectures
* understand representation choices
* experiment with RL vs BC vs hybrid approaches
* share code and ideas that might help others

### Planned improvements

* Clean up & refactor the codebase
* Add a proper benchmarking / evaluation CLI
* More systematic agent comparison
* Better documentation of experiments & results

---

## Disclaimer

Expect:

* fast iteration
* sharp edges
* evolving APIs
