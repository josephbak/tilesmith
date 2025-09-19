import numpy as np
import pytest

from tilesmith.kernel import gelu_erf, mlp_graph64, mlp_gpu_block_sim

def test_gelu_erf_matches_float64_formula():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1024,), dtype=np.float32) * 5.0  # wide range
    # Expected via explicit float64 path
    x64 = x.astype(np.float64, copy=False)
    exp = 0.5 * x64 * (1.0 + np.vectorize(lambda t: np.math.erf(t))(x64 / np.sqrt(2.0)))
    exp32 = exp.astype(np.float32, copy=False)
    got = gelu_erf(x)
    assert np.allclose(got, exp32, atol=1e-6)

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
