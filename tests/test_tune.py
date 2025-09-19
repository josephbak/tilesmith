import numpy as np
from tilesmith.tune import autotune_tiles

def _nondominated(points):
    # simple checker for Pareto nondomination on (time, energy)
    def dominated(a, b):
        return (b['time'] <= a['time'] and b['energy'] <= a['energy'] and
                (b['time'] < a['time'] or b['energy'] < a['energy']))
    out = []
    for a in points:
        if not any(dominated(a, b) for b in points if a is not b):
            out.append(a)
    return sorted(out, key=lambda r: (r['time'], r['energy']))

def test_autotune_time_and_frontier():
    rng = np.random.default_rng(7)
    M, K, N = 128, 96, 128
    x = rng.standard_normal((M, K), dtype=np.float32)
    W = rng.standard_normal((K, N), dtype=np.float32)
    b = rng.standard_normal((N,), dtype=np.float32)

    candidates = [(8,16,8), (16,16,8), (16,32,16)]
    res = autotune_tiles(x, W, b, candidates=candidates, repeats=1, objective="time", include_frontier=True, plot=False)
    best = res["best"]; allr = res["all"]; front = res.get("frontier", [])
    assert best["tile"] in [r["tile"] for r in allr]
    assert best["time"] > 0
    # Frontier correctness
    expected_front = _nondominated(allr)
    assert [r["tile"] for r in front] == [r["tile"] for r in expected_front]

def test_autotune_energy_consistency():
    rng = np.random.default_rng(11)
    M, K, N = 96, 96, 96
    x = rng.standard_normal((M, K), dtype=np.float32)
    W = rng.standard_normal((K, N), dtype=np.float32)
    b = rng.standard_normal((N,), dtype=np.float32)

    candidates = [(8,16,8), (16,16,8), (16,32,16)]
    res = autotune_tiles(x, W, b, candidates=candidates, repeats=1, objective="energy", include_frontier=True, plot=False)
    best = res["best"]; allr = res["all"]
    # Energy equals time * proxyP for all measured
    for r in allr:
        assert abs(r["energy"] - r["time"]*r["proxyP"]) < 1e-12
    # Best score equals its time * proxyP
    assert abs(best["score"] - best["time"]*best["proxyP"]) < 1e-12
