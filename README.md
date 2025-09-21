# Tilesmith

Tilesmith is an educational project for learning how AI models are compiled down to CPUs, GPUs, and other accelerators.

It demonstrates how a simple MLP block (`Y = GELU(X @ W + b)`) can be lowered step by step:

- From **graph IR** (tensor algebra, SSA form),
- To **loop IR** (explicit nested loops, buffers, loads/stores),
- To **tiling and bufferization** (blocking into cache- and shared-memory–friendly tiles),
- To **GPU block/thread mapping** (each tile assigned to a GPU block),
- And finally to **autotuning** (searching tile sizes for best runtime or energy-like score).

---

## Features

- Numerically robust GELU (float64 accumulation, float32 outputs).
- Tensor IR and Loop IR with interpreters for correctness checking.
- Fusion pass: `matmul + add + gelu → fused_mlp`.
- Lowering pass: fused tensor ops → i/j/k loops with explicit accumulators.
- GPU-style block simulator (`mlp_gpu_block_sim`) with `(tileM, tileN, tileK)` tiling.
- Unified autotuner: objective can be `time` (runtime) or `energy` (runtime × proxy power).
- Pareto frontier computation for time vs energy trade-offs.

---

## Installation

```bash
pip install -r requirements.txt
```

---

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

---

## Project Structure

```
tilesmith/
├─ exec/
│  ├─ interpreter.py        # Tensor IR interpreter
│  └─ loop_interpreter.py   # Loop IR interpreter
├─ ir/
│  ├─ tiny_ir.py            # Tensor IR (SSA form: matmul, add, gelu, const, etc.)
│  ├─ loop_ir.py            # Loop IR (memrefs, alloc/load/store, for-loops, scalar ops)
│  └─ lower.py              # Lowering pass: fused_mlp → explicit i/j/k loop nest
├─ kernel.py                # GELU, NumPy graph reference, GPU block simulator
├─ tune.py                  # Autotuner, Pareto frontier utilities
examples/
├─ run_interpreter_demo.py  # Run tensor IR module
├─ run_fusion_demo.py       # Show fusion pass
└─ run_execute_lowered_demo.py  # Compare tensor IR vs loop IR execution
main.py                     # CLI entrypoint
requirements.txt
README.md
```

---

## Next Steps

- Add more unit tests (correctness for small shapes).
- Extend autotuner with smarter search (randomized, learned cost models, RL).
- Experiment with dynamic shapes and quantization.
- Map Tilesmith concepts onto MLIR dialects (`linalg` → `scf` → `gpu` → `nvvm`).

---

## Why the name?

The heart of this project is **tiling**: breaking large tensor computations into cache- and GPU-friendly blocks. “Tilesmith” = a craftsperson shaping tiles.
