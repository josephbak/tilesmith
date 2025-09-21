# tilesmith/ir/lower.py
from __future__ import annotations
from typing import Tuple
from .tiny_ir import Module as TModule, Func as TFunc, Op as TOp, TensorType
from .loop_ir import Module as LModule, Func as LFunc, Block as LBlock, Value as LValue, Op as LOp
from .loop_ir import ScalarType, MemRefType, IRBuilder as LBuilder, verify_module as lverify

def _shape_of(ty: TensorType) -> Tuple[int, ...]:
    # Require concrete ints for now (symbols -> future work)
    shp = []
    for d in ty.shape:
        if not isinstance(d, int):
            raise ValueError(f"lowering requires concrete dims for now, got {d!r}")
        shp.append(d)
    return tuple(shp)

def lower_fused_mlp_to_loops(tmod: TModule) -> LModule:
    """
    tensor IR:
        %X = input "X" : tensor<f32, MxK>
        %W = const "W" : tensor<f32, KxN>
        %b = const "b" : tensor<f32, N>
        %y = fused_mlp %X, %W, %b : (MxK, KxN, N) -> (MxN)
        return %y
    loop IR:
        func @lowered(%X: memref<f32,MxK>, %W: memref<f32,KxN>, %B: memref<f32,N>) { i/j/k loops … }
    """
    assert len(tmod.funcs) == 1, "demo: single-function module expected"
    tf: TFunc = tmod.funcs[0]
    ops = tf.body.ops

    # Find input/consts/fused_mlp
    op_input = next((o for o in ops if o.opname == "input" and o.attrs.get("name") == "X"), None)
    op_W     = next((o for o in ops if o.opname == "const" and o.attrs.get("name") == "W"), None)
    op_b     = next((o for o in ops if o.opname == "const" and o.attrs.get("name") == "b"), None)
    op_fused = next((o for o in ops if o.opname == "fused_mlp"), None)
    if not (op_input and op_W and op_b and op_fused):
        raise ValueError("expected input X, const W, const b, and a fused_mlp op")

    # Types / shapes
    x_ty: TensorType = op_input.result.type
    w_ty: TensorType = op_W.result.type
    b_ty: TensorType = op_b.result.type
    M, K = _shape_of(x_ty)
    K2, N = _shape_of(w_ty)
    (Nb,) = _shape_of(b_ty)
    if K != K2 or Nb != N:
        raise ValueError("shape mismatch: (MxK)@(KxN)+N required")

    # Build loop IR function signature: memrefs as args
    lb = LBuilder()
    f32 = ScalarType("f32")
    X = LValue("%X", MemRefType(f32, (M, K)))
    W = LValue("%W", MemRefType(f32, (K, N)))
    B = LValue("%B", MemRefType(f32, (N,)))
    body = LBlock(args=[X, W, B], ops=[])

    # Y buffer (result)
    Y_alloc = lb.alloc(f32, (M, N)); body.ops.append(Y_alloc)
    Y = Y_alloc.result

    # for i in [0,M)
    for_i = lb.for_(0, M, 1, iv_name="%i"); body.ops.append(for_i)
    # for j in [0,N)
    for_j = lb.for_(0, N, 1, iv_name="%j"); for_i.region.ops.append(for_j)

    # acc = 0.0  (scalar)
    c0 = lb.const(f32, 0.0); for_j.region.ops.append(c0)

    # Create a 1-element buffer to hold the running sum
    acc_buf = lb.alloc(f32, (1,));            for_j.region.ops.append(acc_buf)
    idx0    = lb.const(ScalarType("i64"), 0); for_j.region.ops.append(idx0)
    st0     = lb.store(c0.result, acc_buf.result, [idx0.result]); for_j.region.ops.append(st0)

    # for k in [0,K): acc = fmaf(X[i,k], W[k,j], acc)
    for_k = lb.for_(0, K, 1, iv_name="%k"); for_j.region.ops.append(for_k)
    Xi = lb.load(X, [for_i.result, for_k.result]);          for_k.region.ops.append(Xi)
    Wk = lb.load(W, [for_k.result, for_j.result]);          for_k.region.ops.append(Wk)
    acc_old = lb.load(acc_buf.result, [idx0.result]);       for_k.region.ops.append(acc_old)
    acc_new = lb.fmaf(Xi.result, Wk.result, acc_old.result);for_k.region.ops.append(acc_new)
    st_acc  = lb.store(acc_new.result, acc_buf.result, [idx0.result]); for_k.region.ops.append(st_acc)

    # tmp = acc_final + b[j]
    acc_final = lb.load(acc_buf.result, [idx0.result]);     for_j.region.ops.append(acc_final)
    bj        = lb.load(B, [for_j.result]);                 for_j.region.ops.append(bj)
    tmp       = lb.addf(acc_final.result, bj.result);       for_j.region.ops.append(tmp)

    # y = gelu(tmp)
    yv = lb.gelu(tmp.result);                                for_j.region.ops.append(yv)

    # Y[i,j] = y
    st = lb.store(yv.result, Y, [for_i.result, for_j.result]); for_j.region.ops.append(st)

    # return Y buffer handle
    ret = LOp("return", [Y])
    body.ops.append(ret)

    lfunc = LFunc(name=f"{tf.name}_lowered", args=[X, W, B], body=body)
    lmod = LModule(funcs=[lfunc])
    lverify(lmod)
    return lmod
