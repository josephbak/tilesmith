import numpy as np
from math import ceil
from scipy import special

# --- Numerically robust GELU (internally float64) ---
def gelu_erf(x: np.ndarray) -> np.ndarray:
    """Compute GELU activation using erf formulation, promoting to float64 for stability."""
    x64 = x.astype(np.float64, copy=False)
    y64 = 0.5 * x64 * (1.0 + special.erf(x64 / np.sqrt(2.0)))
    return y64.astype(np.float32, copy=False)

# --- Graph reference in float64 up to GELU ---
def mlp_graph64(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reference implementation: matmul + bias + GELU, all in float64 then cast back to float32."""
    x64 = x.astype(np.float64, copy=False)
    W64 = W.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    z64 = x64 @ W64 + b64
    return gelu_erf(z64)

# --- GPU block mapping simulator with tiled K and float64 accumulation ---
def mlp_gpu_block_sim(x: np.ndarray, W: np.ndarray, b: np.ndarray,
                      tileM: int = 64, tileN: int = 64, tileK: int = 64) -> np.ndarray:
    """Simulate GPU-style block tiling for matmul+bias+GELU. 
    Each block computes a tile (tileM x tileN) and reduction is tiled along K.
    Accumulation is performed in float64 for numerical stability."""
    M, K = x.shape
    K2, N = W.shape
    assert K == K2
    Y = np.empty((M, N), dtype=np.float32)

    grid_y = int(ceil(M / tileM))
    grid_x = int(ceil(N / tileN))

    for by in range(grid_y):
        for bx in range(grid_x):
            i0 = by * tileM
            j0 = bx * tileN
            Mi = min(tileM, M - i0)
            Nj = min(tileN, N - j0)

            # Block-level accumulator buffer
            Cblk64 = np.zeros((Mi, Nj), dtype=np.float64)
            for k0 in range(0, K, tileK):
                Kt = min(tileK, K - k0)
                # Subtiles (conceptually in shared memory)
                Ablk64 = x[i0:i0+Mi, k0:k0+Kt].astype(np.float64, copy=False)
                Bblk64 = W[k0:k0+Kt, j0:j0+Nj].astype(np.float64, copy=False)
                # Accumulate partial results
                Cblk64 += Ablk64 @ Bblk64

            # Add bias and apply GELU before writing back
            Cblk64 += b[j0:j0+Nj].astype(np.float64, copy=False)
            Y[i0:i0+Mi, j0:j0+Nj] = gelu_erf(Cblk64)
    return Y
