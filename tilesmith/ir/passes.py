# tilesmith/ir/passes.py
from typing import List
from .tiny_ir import Module, Func, Block, Op, IRBuilder

def fuse_mlp_pass(mod: Module) -> Module:
    """
    Find the linear chain matmul -> add -> gelu and replace it with fused_mlp.
    Single-function, single-block, first-match only (kept intentionally small).
    """
    assert len(mod.funcs) == 1, "demo pass assumes a single function"
    f: Func = mod.funcs[0]
    ops: List[Op] = f.body.ops
    if not ops:
        return mod

    # Locate pattern
    idx_mat = idx_add = idx_gelu = -1
    X = W = b = None
    for i, op in enumerate(ops):
        if op.opname != "matmul":
            continue
        # matmul
        idx_mat = i
        X, W = op.operands

        # add must immediately follow, consuming matmul result
        if i + 1 >= len(ops) or ops[i+1].opname != "add":
            continue
        add_op = ops[i+1]
        if add_op.operands[0] is not op.result:
            continue
        idx_add = i + 1
        b = add_op.operands[1]

        # gelu must immediately follow, consuming add result
        if i + 2 >= len(ops) or ops[i+2].opname != "gelu":
            continue
        gelu_op = ops[i+2]
        if gelu_op.operands[0] is not add_op.result:
            continue
        idx_gelu = i + 2
        break

    # No match → return unchanged
    if idx_mat < 0:
        return mod

    # Build fused op
    bld = IRBuilder()
    fused = bld.fused_mlp(X, W, b)

    # Rewrite ops: keep everything, but replace the 3-op chain with 1 fused op
    new_ops: List[Op] = []
    for j, op in enumerate(ops):
        if j == idx_mat:
            new_ops.append(fused)
        elif j in (idx_add, idx_gelu):
            continue
        else:
            new_ops.append(op)

    # Fix return if it referenced the old gelu result
    ret = new_ops[-1]
    if ret.opname == "return":
        old_ret_val = ops[idx_gelu].result  # the gelu result
        if ret.operands[0] is old_ret_val:
            new_ops[-1] = Op("return", [fused.result], fused.result.type)

    return Module(funcs=[Func(name=f.name, args=f.args, result_type=f.result_type, body=Block(ops=new_ops))])