# TileSmith

TileSmith is an educational project for learning how AI models are compiled down to CPUs, GPUs, and other accelerators.

It demonstrates how a simple MLP block (`Y = GELU(X @ W + b)`) can be lowered step by step:

- From **graph IR** (tensor algebra),
- To **loop IR** (explicit nested loops),
- To **tiling and bufferization** (blocking into cache- and shared-memory–friendly tiles),
- To **GPU block/thread mapping** (each tile assigned to a GPU block),
- And finally to **autotuning** (searching tile sizes for best runtime or energy-like score).

## Features

- Numerically robust **GELU** implementation (float64 accumulation, float32 outputs).
- **Graph reference** (`mlp_graph64`) to check correctness.
- **GPU-style block simulator** (`mlp_gpu_block_sim`) with `(tileM, tileN, tileK)` tiling.
- Unified **autotuner**:
  - Objective can be `time` (runtime) or `energy` (runtime × proxy power).
  - Returns all configs, the best config, and the Pareto frontier.
- Optional **Pareto frontier plot** of time vs energy trade-offs.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### CLI

```bash
python main.py --M 512 --K 768 --N 512 --objective energy --repeats 2 --plot
```

Options:
- `--M, --K, --N`: matrix dimensions.
- `--objective`: `time` or `energy`.
- `--repeats`: number of timing repetitions.
- `--plot`: show Pareto frontier plot.

### As a Library

```python
import numpy as np
from tilesmith import autotune_tiles

rng = np.random.default_rng(2025)
M, K, N = 512, 768, 512
x = rng.standard_normal((M, K), dtype=np.float32)
W = rng.standard_normal((K, N), dtype=np.float32)
b = rng.standard_normal((N,), dtype=np.float32)

res = autotune_tiles(x, W, b, repeats=2, objective="energy", include_frontier=True, plot=True)
print("Best schedule:", res["best"])
```

## Project Structure

```
tilesmith/
├─ tilesmith/          # Python package
│  ├─ __init__.py
│  ├─ kernel.py        # GELU, graph reference, GPU-block simulator
│  └─ tune.py          # autotuner, Pareto frontier utilities
├─ main.py             # CLI entrypoint
├─ requirements.txt
└─ README.md
```

## Next Steps

- Add unit tests (correctness for small shapes).
- Extend autotuner with more search strategies (random, cost models, RL).
- Experiment with dynamic shapes and quantization.
- Map TileSmith concepts to MLIR dialects (`linalg` → `scf` → `gpu` → `nvvm`).

## Why the name?

The heart of this project is **tiling** matrix operations for hardware.  
A *smith* is a craftsperson who shapes raw material into a useful tool.  
TileSmith is a small workshop for crafting tiled kernels and schedules.
