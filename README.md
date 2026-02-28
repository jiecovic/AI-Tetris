# RL-Tetris

<img src="assets/tetris_demo_cnn_imitation_run_001.gif" width="500" />

Experimental playground for teaching AI to play Tetris.

The project is a research and learning sandbox focused on planning policies, reinforcement learning, imitation, and representation learning. The codebase is YAML-driven (Hydra + Pydantic) to support fast experimentation.

## Example Run (Included)

This repo includes a small imitation checkpoint you can load directly:

```bash
# Watch (pygame)
tetris-watch --run examples/runs/cnn_imitation_run_001 --which best --reload 0

# Benchmark (no rendering)
tetris-benchmark --run examples/runs/cnn_imitation_run_001 --which best --episodes 50 --progress
```

---

## What is inside

### Learning approaches

This project combines planning, imitation learning, and reinforcement learning for Tetris:

- Heuristic planning policies (Codemy-style linear heuristics with lookahead)
- Genetic Algorithm (GA) optimization of heuristic weights
- Temporal-difference (TD) learning for heuristic weights (custom planning/value TD for lookahead, with GAE-style updates and optional EMA target net)
- Imitation learning (behavior cloning from expert data)
- Reinforcement learning (PPO, Maskable PPO)

### Models and representations

Configurable via YAML:

- SB3 policies with custom feature extractors (CNN/ViT/MLP-Mixer)
- Planning policies currently use a linear value scorer
- Roadmap: MLP/CNN value models on post-action states for planning

### Environment

- Custom Rust Tetris engine (PyO3 + maturin)
- Gymnasium-compatible Python envs
- Pygame rendering

---

## Requirements

- Python >= 3.10 with pip
- Rust toolchain (rustc + cargo via rustup) for the PyO3/maturin engine
- Windows: Visual Studio Build Tools with "Desktop development with C++"

Python dependencies are declared in `pyproject.toml` and installed via `pip install -e .`.

---

## Installation

For dev tooling (ruff/pyright), use a local venv so Pyright can resolve imports:

```bash
python -m venv .venv
.\.venv\Scripts\pip install -e .[dev]
```

Base install:

```bash
pip install -e .
```

CUDA (optional):

Install a CUDA-enabled PyTorch wheel first (Windows example), then install the project:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

Optional visualization extras:

```bash
pip install -e .[viz]
```

---

## Dev workflow

Run one command set for local checks/fixes:

```bash
# Auto-fix lint + format
tetris-dev fix

# Validate lint + format + types
tetris-dev check
```

Install and run git hooks with the same toolchain:

```bash
tetris-dev hooks --install
```

`tetris-dev check` runs:
- `ruff check .`
- `ruff format --check .`
- `pyright`

Pre-commit runs:
- `ruff --fix`
- `ruff-format`
- `pyright`

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

TD config supports weight normalization/scale, EMA target updates, and auto feature-clear mode (pre/post).

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

## Best results so far (informal)

- Current best (Feb 2026): PPO + PatchScan transformer policy (CNN stem -> sliding-window patch tokens -> transformer mixer).
  - Configs: `configs/ppo/ppo_patchscan_ablation.yaml` + `conf/sb3_policies/patchscan_ablation.yaml`.
  - PatchScan uses full-width board patches as tokens (slide over height with overlap), pools token features (mean+max), and adds tokens for active/next piece.
  - Pretrained selection: `conf/sb3_policies/pretrained.yaml` currently points at `runs/ppo_patchscan_ablation_run_008`.
- Planning baseline: Codemy-style heuristic with plies=3 and 3rd-ply expectimax.
- Distill baseline: behavior cloning from the planner into a reactive policy.
- Prior models: ViT with column tokens learns faster; CNN reaches similar quality and is faster at inference.
- Outcome: fast, reactive inference with near-planning performance (no search at runtime).
- Demo: the GIF above shows a CNN reactive policy distilled from Codemy plies=3 expectimax; weights will be shared after refinement.

---

## Disclaimer

Expect:

- fast iteration
- sharp edges
- evolving APIs
