import numpy as np
from tilesmith.ir.tiny_ir import IRBuilder, TensorType, Func, Block, Module
from tilesmith.ir import fuse_mlp_pass
from tilesmith.exec.interpreter import run_module

# Build unfused IR with const W,b
M, K, N = 4, 8, 16
rng = np.random.default_rng(0)
X = rng.standard_normal((M, K), dtype=np.float32)
W = rng.standard_normal((K, N), dtype=np.float32)
b = rng.standard_normal((N,), dtype=np.float32)

bld = IRBuilder()
tX, tW, tb, tY = (TensorType("f32",(M,K)), TensorType("f32",(K,N)), TensorType("f32",(N,)), TensorType("f32",(M,N)))
opX = bld.input("X", tX)
opW = bld.const("W", tW, value=W)
opb = bld.const("b", tb, value=b)
op0 = bld.matmul(opX.result, opW.result)
op1 = bld.add(op0.result, opb.result)
op2 = bld.gelu(op1.result)
opR = bld.ret(op2.result)
mod = Module(funcs=[Func("mlp_exec", [opX.result], tY, Block(ops=[opX, opW, opb, op0, op1, op2, opR]))])

# Run before fusion
Y_before = run_module(mod, {"X": X})

# Run fusion pass
mod_fused = fuse_mlp_pass(mod)

# Run after fusion
Y_after = run_module(mod_fused, {"X": X})

print("equivalent:", np.allclose(Y_before, Y_after, atol=1e-5))
print("\n-- BEFORE --\n", mod, "\n-- AFTER --\n", mod_fused)
