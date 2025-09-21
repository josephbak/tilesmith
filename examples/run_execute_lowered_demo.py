# examples/run_execute_lowered_demo.py
import numpy as np

from tilesmith.ir.tiny_ir import IRBuilder, TensorType, Func, Block, Module
from tilesmith.ir import fuse_mlp_pass
from tilesmith.ir.lower import lower_fused_mlp_to_loops

from tilesmith.exec.interpreter import run_module as run_tensor_ir
from tilesmith.exec.loop_interpreter import run_loop_module as run_loop_ir

# Build unfused tensor IR with const W,b
M, K, N = 4, 8, 16
rng = np.random.default_rng(0)
X = rng.standard_normal((M, K), dtype=np.float32)
W = rng.standard_normal((K, N), dtype=np.float32)
b = rng.standard_normal((N,), dtype=np.float32)

bld = IRBuilder()
tX, tW, tb, tY = (
    TensorType("f32", (M, K)),
    TensorType("f32", (K, N)),
    TensorType("f32", (N,)),
    TensorType("f32", (M, N)),
)

opX = bld.input("X", tX)
opW = bld.const("W", tW, value=W)
opb = bld.const("b", tb, value=b)
op0 = bld.matmul(opX.result, opW.result)
op1 = bld.add(op0.result, opb.result)
op2 = bld.gelu(op1.result)
opR = bld.ret(op2.result)
tmod = Module(funcs=[Func("mlp", [opX.result], tY, Block(ops=[opX, opW, opb, op0, op1, op2, opR]))])

# 1) Run tensor-IR (reference)
Y_tensor = run_tensor_ir(tmod, {"X": X})

# 2) Fuse then lower to loops, then run loop-IR
tmod_fused = fuse_mlp_pass(tmod)
lmod = lower_fused_mlp_to_loops(tmod_fused)
Y_loop = run_loop_ir(lmod, {"%X": X, "%W": W, "%B": b})

print("tensor vs loop allclose:", np.allclose(Y_tensor, Y_loop, atol=1e-5))
print("tensor:", Y_tensor.shape, Y_tensor.dtype, "loop:", Y_loop.shape, Y_loop.dtype)
