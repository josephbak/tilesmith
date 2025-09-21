import numpy as np
from scipy import special

def _gelu_erf_np(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64, copy=False)
    y64 = 0.5 * x64 * (1.0 + special.erf(x64 / np.sqrt(2.0)))
    return y64.astype(np.float32, copy=False)

def run_module(mod, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """
    Execute a single-function module.
    inputs: mapping from input name (e.g., "X") -> np.ndarray
    Returns the np.ndarray produced by the function's return.
    """
    assert len(mod.funcs) == 1, "exec supports single-function modules"
    f = mod.funcs[0]
    env = {}

    for op in f.body.ops:
        if op.opname == "input":
            name = op.attrs.get("name")
            arr = inputs[name]
            env[op.result.name] = arr
        elif op.opname == "const":
            arr = op.attrs.get("value")
            if arr is None:
                raise RuntimeError(f"const {op.attrs.get('name','?')} has no payload")
            env[op.result.name] = arr
        elif op.opname == "matmul":
            a = env[op.operands[0].name]
            b = env[op.operands[1].name]
            env[op.result.name] = a @ b
        elif op.opname == "add":
            a = env[op.operands[0].name]
            b = env[op.operands[1].name]
            env[op.result.name] = a + b  # rely on NumPy broadcasting (N) or (1,N)
        elif op.opname == "gelu":
            a = env[op.operands[0].name]
            env[op.result.name] = _gelu_erf_np(a)
        elif op.opname == "fused_mlp":
            X = env[op.operands[0].name]
            W = env[op.operands[1].name]
            b = env[op.operands[2].name]
            env[op.result.name] = _gelu_erf_np(X @ W + b)
        elif op.opname == "return":
            return env[op.operands[0].name]
        else:
            raise NotImplementedError(f"Unsupported op: {op.opname}")

    raise RuntimeError("No return encountered")
