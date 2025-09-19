#!/usr/bin/env python3
import argparse, json
import numpy as np
from tilesmith import autotune_tiles

def main():
    ap = argparse.ArgumentParser(description="TileSmith: educational tiled matmul autotuner.")
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--K", type=int, default=768)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--objective", choices=["time","energy"], default="time")
    ap.add_argument("--no-frontier", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    M, K, N = args.M, args.K, args.N
    x = rng.standard_normal((M, K), dtype=np.float32)
    W = rng.standard_normal((K, N), dtype=np.float32)
    b = rng.standard_normal((N,), dtype=np.float32)

    res = autotune_tiles(x, W, b,
                         repeats=args.repeats,
                         objective=args.objective,
                         include_frontier=not args.no_frontier,
                         plot=args.plot)

    print("\n=== RESULT ===")
    print(json.dumps(res["best"], indent=2, default=float))

if __name__ == "__main__":
    main()
