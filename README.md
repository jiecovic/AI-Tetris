# RL-Tetris 🧩🤖

<img src="assets/demo_v5.gif" width="500" />

**Experimental playground for teaching AI to play Tetris.**

This project is a research / learning sandbox focused on **model-free reinforcement learning**, neural network architectures, and representation learning for classic games.
The codebase is intentionally flexible and YAML-driven, allowing rapid experimentation with environments, models, and training setups.

---

## What’s inside

### 🧠 Learning approaches

This project combines **heuristic planning**, **imitation learning**, and **reinforcement learning** for Tetris.

#### Heuristic expert (Codemy-style)

A hand-crafted expert inspired by *The Near-Perfect Tetris Bot* (CodemyRoad) is used as a starting point.

The expert uses a **fixed linear heuristic value function** over global board features (e.g. holes, aggregate height, bumpiness) and performs **one-step lookahead**:

* all legal placements of the current tetromino are simulated
* the resulting boards are scored by the heuristic
* the best placement is selected

This corresponds to **depth-1 evaluation / horizon-1 control** and works extremely well under the **7-bag tetromino generator**, but degrades under **uniform random tetromino sampling**, where future variance is much higher.

---

#### Extended stochastic lookahead (same heuristic)

Without changing or re-optimizing the heuristic score function, the expert is extended to **deeper evaluation under uncertainty**:

* for each candidate placement of the current tetromino,
* the **expected heuristic score after the next tetromino** is computed,
* assuming a **uniform distribution over the 7 tetromino types (1/7 each)**.

This yields **depth-2 evaluation with horizon-1 control**: only the current action is chosen, but future randomness is explicitly accounted for.

Despite using the *same heuristic*, this extension produces **near-perfect play under uniform tetromino sampling**, at the cost of very high computational expense.

---

#### Behavior cloning & distillation

Because the stochastic lookahead expert is too slow, its decisions are **behavior-cloned** into a fast neural policy.

The resulting agent:

* runs at **feed-forward network speed**,
* observes only the **current board**, **current tetromino**, and **next tetromino**,
* retains the performance of the depth-2 expert.

This yields a **near-perfect Tetris agent for uniformly random tetromino sequences**, without online search or lookahead.

---

#### Reinforcement learning

The project also supports **pure model-free RL** (PPO, Maskable PPO, DQN) and is set up for future **BC → RL fine-tuning** experiments, enabling systematic comparison between planning-based experts and learned policies.


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
