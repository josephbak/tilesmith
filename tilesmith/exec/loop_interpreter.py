# tilesmith/exec/loop_interpreter.py
from __future__ import annotations
import numpy as np
from scipy import special

# ---- helpers ----

_DTYPE_MAP = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
}

def _as_dtype(kind: str):
    if kind not in _DTYPE_MAP:
        raise ValueError(f"unsupported dtype: {kind}")
    return _DTYPE_MAP[kind]

def _gelu_erf_np(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64, copy=False)
    y64 = 0.5 * x64 * (1.0 + special.erf(x64 / np.sqrt(2.0)))
    return y64.astype(np.float32, copy=False)

# ---- interpreter ----

def run_loop_module(lmod, inputs: dict[str, np.ndarray]):
    """
    Execute a loop-IR module (single function).
    inputs: mapping of function arg names (e.g., "%X","%W","%B") -> NumPy arrays
    Returns: the object passed to `return` (usually a NumPy array / buffer)
    """
    assert len(lmod.funcs) == 1, "only single-function modules supported"
    lfunc = lmod.funcs[0]

    # Environment maps SSA value names -> runtime objects (scalars or numpy arrays)
    env = {}

    # Seed args: the Func.args are Values with names like "%X", "%W", "%B"
    for arg in lfunc.args:
        if arg.name not in inputs:
            raise KeyError(f"missing input for {arg.name}")
        env[arg.name] = inputs[arg.name]

    def eval_block(block, env):
        # returns (maybe) a value when a 'return' is hit
        for op in block.ops:
            on = op.opname

            if on == "alloc":
                # create zero-initialized buffer
                shape = op.result.type.shape
                dtype = _as_dtype(op.result.type.elem.kind)
                env[op.result.name] = np.zeros(shape, dtype=dtype)

            elif on == "const":
                val = op.attrs["value"]
                # store scalars as numpy scalar with correct dtype
                dtype = _as_dtype(op.result.type.kind)
                env[op.result.name] = dtype(type(val)(val))

            elif on == "load":
                # operands: [buffer] + indices (as SSA values)
                buf = env[op.operands[0].name]
                idx_vals = [env[v.name] for v in op.operands[1:]]
                # indices should be integers (numpy scalar ok)
                idx_tuple = tuple(int(iv) for iv in idx_vals)
                env[op.result.name] = buf[idx_tuple]

            elif on == "store":
                # operands: [value, buffer] + indices
                val = env[op.operands[0].name]
                buf = env[op.operands[1].name]
                idx_vals = [env[v.name] for v in op.operands[2:]]
                idx_tuple = tuple(int(iv) for iv in idx_vals)
                buf[idx_tuple] = val

            elif on == "addf":
                a = env[op.operands[0].name]
                b = env[op.operands[1].name]
                env[op.result.name] = type(a)(a + b)

            elif on == "mulf":
                a = env[op.operands[0].name]
                b = env[op.operands[1].name]
                env[op.result.name] = type(a)(a * b)

            elif on == "fmaf":
                a = env[op.operands[0].name]
                b = env[op.operands[1].name]
                c = env[op.operands[2].name]
                env[op.result.name] = type(a)(a * b + c)

            elif on == "gelu":
                a = env[op.operands[0].name]
                # scalar GELU: use vector path then pull back scalar
                out = _gelu_erf_np(np.array([a], dtype=np.float32))[0]
                env[op.result.name] = out

            elif on == "for":
                lb = op.attrs["lb"]; ub = op.attrs["ub"]; st = op.attrs.get("step", 1)
                # The induction var is the op.result Value; bind it each iteration
                iv_name = op.result.name
                for i in range(lb, ub, st):
                    # set scalar i64
                    env[iv_name] = np.int64(i)
                    ret = eval_block(op.region, env)
                    if ret is not None:
                        return ret  # early return bubbled up

            elif on == "return":
                # return the runtime object bound to the operand
                if not op.operands:
                    return None
                return env[op.operands[0].name]

            else:
                raise NotImplementedError(f"unsupported loop op: {on}")

        return None  # no return in this block

    ret = eval_block(lfunc.body, env)
    return ret
