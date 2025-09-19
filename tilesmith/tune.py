import time
from math import ceil
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .kernel import mlp_graph64, mlp_gpu_block_sim

# --- Power proxy (toy heuristic) ---
def power_proxy(M:int, K:int, N:int, tileM:int, tileN:int, tileK:int,
                w_shared:float=1e-6, w_global:float=2e-6, base_power:float=1e-3) -> float:
    """Estimate proxy for power consumption based on tile dimensions."""
    blocks_y = int(ceil(M / tileM))
    blocks_x = int(ceil(N / tileN))
    k_slabs  = int(ceil(K / tileK))

    per_blk_k = (tileM * tileK + tileK * tileN)
    per_blk   = (tileM * tileN)
    per_block_activity = w_shared * (k_slabs * per_blk_k) + w_global * per_blk
    grid_activity = (blocks_y * blocks_x) * per_block_activity
    return base_power + grid_activity

# --- Timing helpers ---
def _time_once(fn, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0

def _best_of(fn, repeats:int=3, *args, **kwargs) -> float:
    fn(*args, **kwargs)  # warm-up
    return min(_time_once(fn, *args, **kwargs) for _ in range(repeats))

# --- Pareto utilities ---
def _is_dominated(a:Dict[str,Any], b:Dict[str,Any]) -> bool:
    return (b["time"] <= a["time"] and b["energy"] <= a["energy"] and
            (b["time"] < a["time"] or b["energy"] < a["energy"]))

def _pareto_frontier(results:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    clean = [r for r in results if np.isfinite(r["time"]) and np.isfinite(r["energy"])]
    frontier = []
    for a in clean:
        if not any(_is_dominated(a, b) for b in clean if a is not b):
            frontier.append(a)
    frontier.sort(key=lambda r: (r["time"], r["energy"]))
    return frontier

def _print_frontier(frontier:List[Dict[str,Any]]) -> None:
    if not frontier:
        print("Pareto frontier is empty."); return
    print("\nPareto frontier (nondominated by time & energy):")
    print(" idx   tile(M,N,K)    time_ms    proxyP        energy_score")
    for i, r in enumerate(frontier):
        print(f"{i:>3}   {tuple(r['tile'])!s:>12}   {r['time']*1e3:8.2f}   {r['proxyP']:.6f}   {r['energy']:.6f}")

# --- Unified autotuner ---
def autotune_tiles(x:np.ndarray, W:np.ndarray, b:np.ndarray,
                   candidates:Optional[List[Tuple[int,int,int]]]=None,
                   repeats:int=3, atol:float=1e-5,
                   objective:str="time",          
                   include_frontier:bool=True,    
                   plot:bool=False,               
                   power_weights:Dict[str,float]=dict(w_shared=1e-6, w_global=2e-6, base_power=1e-3)
                   ) -> Dict[str,Any]:
    """Autotune tile sizes for matmul+bias+GELU.
    
    Parameters
    ----------
    objective : str
        Either "time" (minimize runtime) or "energy" (minimize runtime*proxy_power).
    include_frontier : bool
        Whether to compute and print Pareto frontier.
    plot : bool
        Whether to plot time vs energy scatter and frontier.
    """
    M, K = x.shape
    _, N = W.shape

    if candidates is None:
        Ms = [8, 16, 32, 64]
        Ns = [16, 32, 64, 128]
        Ks = [8, 16, 32, 64]
        candidates = [(m, n, k) for m in Ms for n in Ns for k in Ks]

    ref = mlp_graph64(x, W, b)
    all_measured: List[Dict[str,Any]] = []

    best = {"score": float("inf"), "tile": None, "time": None, "proxyP": None}
    for (tm, tn, tk) in candidates:
        if tm <= 0 or tn <= 0 or tk <= 0:
            continue
        try:
            y = mlp_gpu_block_sim(x, W, b, tileM=tm, tileN=tn, tileK=tk)
            if not np.allclose(ref, y, atol=atol):
                continue

            t = _best_of(mlp_gpu_block_sim, repeats=repeats, x=x, W=W, b=b,
                         tileM=tm, tileN=tn, tileK=tk)

            P = power_proxy(M, K, N, tm, tn, tk, **power_weights)
            E = t * P
            all_measured.append({"tile": (tm, tn, tk), "time": t, "proxyP": P, "energy": E})

            score = t if objective == "time" else E
            if score < best["score"]:
                best.update({"score": score, "tile": (tm, tn, tk), "time": t, "proxyP": P})

        except Exception:
            continue

    result = {"best": best, "all": all_measured}
    frontier = _pareto_frontier(all_measured) if include_frontier else []
    if include_frontier:
        _print_frontier(frontier)
        result["frontier"] = frontier

    if plot:
        try:
            import matplotlib.pyplot as plt
            xs = [r["time"]*1e3 for r in all_measured]
            ys = [r["energy"] for r in all_measured]
            plt.figure()
            plt.scatter(xs, ys, s=12, label="all configs")
            if include_frontier and frontier:
                xs_f = [r["time"]*1e3 for r in frontier]
                ys_f = [r["energy"] for r in frontier]
                plt.scatter(xs_f, ys_f, s=30, marker="x", label="Pareto frontier")
                plt.plot(xs_f, ys_f, linewidth=1)
            plt.xlabel("Time (ms)")
            plt.ylabel("Energy-like score (time × proxy power)")
            plt.title("Schedule trade-offs")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"[plot disabled] {e!r}")

    return result
