# examples/run_lowering_demo.py
from tilesmith.ir.tiny_ir import IRBuilder, TensorType, Func, Block, Module
from tilesmith.ir import fuse_mlp_pass  # if you wired the pass export
from tilesmith.ir.lower import lower_fused_mlp_to_loops

# Build tensor IR with const W,b and fused_mlp (use your fusion pass or build fused directly)
b = IRBuilder()
M,K,N = 4,8,16
tX,tW,tb,tY = (TensorType("f32",(M,K)), TensorType("f32",(K,N)), TensorType("f32",(N,)), TensorType("f32",(M,N)))
opX = b.input("X", tX)
opW = b.const("W", tW)
opb = b.const("b", tb)
# unfused:
op0 = b.matmul(opX.result, opW.result)
op1 = b.add(op0.result, opb.result)
op2 = b.gelu(op1.result)
opR = b.ret(op2.result)
tmod = Module(funcs=[Func("mlp", [opX.result], tY, Block(ops=[opX, opW, opb, op0, op1, op2, opR]))])

# Fuse then lower
tmod_fused = fuse_mlp_pass(tmod)
lmod = lower_fused_mlp_to_loops(tmod_fused)
print(lmod)
