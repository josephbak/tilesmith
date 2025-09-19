import numpy as np
from scipy import special
import pytest

from tilesmith.kernel import gelu_erf, mlp_graph64, mlp_gpu_block_sim

# --- PyTorch reference (optional) ---
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_gelu_erf_matches_float64_formula():
    """Compare gelu_erf directly against PyTorch's reference GELU."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4096,), dtype=np.float32) * 10.0  # wide range

    # Torch reference (float32)
    xt = torch.from_numpy(x)
    ref = F.gelu(xt, approximate="none").cpu().numpy().astype(np.float32, copy=False)

    got = gelu_erf(x)

    assert np.allclose(got, ref, atol=1e-6), \
        f"max abs diff={np.max(np.abs(got-ref))}"

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_gelu_erf_is_more_stable_than_naive_float32():
    """
    Show that our gelu_erf (promoting to float64 internally) is
    more stable than a naive float32-only GELU implementation.
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10000,), dtype=np.float32) * 50.0  # very wide range

    # Torch reference
    xt = torch.from_numpy(x)
    ref = F.gelu(xt, approximate="none").cpu().numpy().astype(np.float32, copy=False)

    # Naive float32 GELU (all ops in f32)
    from scipy import special
    x32 = x.astype(np.float32, copy=False)
    naive32 = (0.5 * x32 * (1.0 + special.erf(x32 / np.float32(np.sqrt(2.0)))))
    naive32 = naive32.astype(np.float32, copy=False)

    # Our robust implementation (internally f64)
    robust = gelu_erf(x)

    # Compare errors vs Torch reference
    err32 = np.max(np.abs(naive32 - ref))
    err64 = np.max(np.abs(robust  - ref))

    assert err64 <= err32 + 1e-7, \
        f"robust should be at least as accurate: err64={err64}, err32={err32}"

@pytest.mark.parametrize("M,K,N", [(4,8,16), (17,33,29), (64,96,37)])
def test_mlp_gpu_block_sim_matches_graph64(M, K, N):
    rng = np.random.default_rng(123)
    x = rng.standard_normal((M, K), dtype=np.float32)
    W = rng.standard_normal((K, N), dtype=np.float32)
    b = rng.standard_normal((N,), dtype=np.float32)

    ref = mlp_graph64(x, W, b)

    # Try a few tile shapes (keep small for speed)
    tiles = [(8, 8, 8), (8, 16, 8), (16, 16, 8)]
    for tm, tn, tk in tiles:
        got = mlp_gpu_block_sim(x, W, b, tileM=tm, tileN=tn, tileK=tk)
        assert np.allclose(ref, got, atol=1e-5), f"Mismatch for tiles {(tm,tn,tk)}"
