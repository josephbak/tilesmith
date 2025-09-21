# examples/run_interpreter_demo.py
import numpy as np
from scipy import special

from tilesmith.ir.tiny_ir import IRBuilder, TensorType, Func, Block, Module
from tilesmith.exec.interpreter import run_module

def gelu_ref(Z: np.ndarray) -> np.ndarray:
    Z64 = Z.astype(np.float64, copy=False)
    Y64 = 0.5 * Z64 * (1.0 + special.erf(Z64 / np.sqrt(2.0)))
    return Y64.astype(np.float32, copy=False)

def build_mlp_const_module(M=4, K=8, N=16):
    rng = np.random.default_rng(0)
    # runtime input
    X = rng.standard_normal((M, K), dtype=np.float32)
    # frozen weights/bias (embedded as consts)
    W = rng.standard_normal((K, N), dtype=np.float32)  # IR expects [K, N]
    b = rng.standard_normal((N,),    dtype=np.float32)

    bld = IRBuilder()
    tX, tW, tb, tY = (
        TensorType("f32", (M, K)),
        TensorType("f32", (K, N)),
        TensorType("f32", (N,)),
        TensorType("f32", (M, N)),
    )

    opX = bld.input("X", tX)
    opW = bld.const("W", tW, value=W)   # const payload attached
    opb = bld.const("b", tb, value=b)

    op0 = bld.matmul(opX.result, opW.result)   # (M,K) @ (K,N) -> (M,N)
    op1 = bld.add(op0.result, opb.result)      # (M,N) + (N) -> (M,N)
    op2 = bld.gelu(op1.result)
    opR = bld.ret(op2.result)

    mod = Module(funcs=[Func("mlp_exec", [opX.result], tY, Block(ops=[opX, opW, opb, op0, op1, op2, opR]))])
    return mod, X, W, b

if __name__ == "__main__":
    M, K, N = 4, 8, 16
    mod, X, W, b = build_mlp_const_module(M, K, N)

    # Run the IR
    Y_ir = run_module(mod, {"X": X})

    # Reference computation
    Z = X @ W + b
    Y_ref = gelu_ref(Z)

    ok = np.allclose(Y_ir, Y_ref, atol=1e-5)
    print("allclose:", ok)
    if not ok:
        diff = np.max(np.abs(Y_ir - Y_ref))
        print("max abs diff:", float(diff))
    else:
        print("Y shape:", Y_ir.shape, "dtype:", Y_ir.dtype)
