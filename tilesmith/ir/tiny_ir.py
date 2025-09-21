from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union, Optional

# ===== Types & Shapes =====

Dim = Union[int, str]  # int (static) or symbol (e.g., "M")
Shape = Tuple[Dim, ...]


@dataclass(frozen=True)
class TensorType:
    dtype: str
    shape: Shape  # e.g., ("M","K") or (64,128) or ("M",128)

    def __str__(self):
        dims = "x".join(str(d) for d in self.shape)
        return f"tensor<{self.dtype}, {dims}>"


# ===== IR Core =====

@dataclass
class Value:
    name: str
    type: TensorType
    def __str__(self): return f"{self.name}: {self.type}"


@dataclass
class Op:
    opname: str
    operands: List[Value]
    result_type: TensorType
    attrs: Dict[str, object] = field(default_factory=dict)
    result: Optional[Value] = None

    def set_result(self, v: Value):
        self.result = v

    def __str__(self):
        ops = ", ".join(o.name for o in self.operands)
        if self.opname in ("matmul", "add"):
            sig = "(" + ", ".join(str(o.type) for o in self.operands) + f") -> {self.result_type}"
            return f"{self.result.name} = {self.opname} {ops} : {sig}"
        elif self.opname in ("gelu",):
            return f"{self.result.name} = gelu {ops} : {self.result_type}"
        elif self.opname == "input":
            return f"{self.result.name} = input \"{self.attrs.get('name','?')}\" : {self.result_type}"
        elif self.opname == "return":
            return f"return {ops}"
        elif self.opname == "const":
            src = self.attrs.get("name", "const")
            return f"{self.result.name} = const \"{src}\" : {self.result_type}"
        elif self.opname == "fused_mlp":
            sig = "(" + ", ".join(str(o.type) for o in self.operands) + f") -> {self.result_type}"
            return f"{self.result.name} = fused_mlp " + ", ".join(o.name for o in self.operands) + f" : {sig}"
        else:
            return f"{self.result.name} = {self.opname} {ops} : {self.result_type}"


@dataclass
class Block:
    args: List[Value] = field(default_factory=list)
    ops: List[Op] = field(default_factory=list)


@dataclass
class Func:
    name: str
    args: List[Value]
    result_type: TensorType
    body: Block

    def __str__(self):
        args_s = ", ".join(f"{a.name}: {a.type}" for a in self.args)
        hdr = f"func @{self.name}({args_s}) -> {self.result_type} {{"
        lines = [hdr]
        for op in self.body.ops:
            lines.append(f"  {op}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class Module:
    funcs: List[Func]
    def __str__(self): return "\n\n".join(str(f) for f in self.funcs)


# ===== Builder / Verifier =====

class IRBuilder:
    def __init__(self):
        self._counter = 0

    def fresh(self, prefix="v") -> str:
        self._counter += 1
        return f"%{prefix}{self._counter}"

    def input(self, name: str, ty: TensorType) -> Op:
        v = Value(self.fresh("arg"), ty)
        op = Op("input", [], ty, attrs={"name": name})
        op.set_result(v)
        return op

    def matmul(self, a: Value, b: Value, out_dtype: Optional[str]=None) -> Op:
        aty, bty = a.type, b.type
        M, K1 = aty.shape
        K2, N = bty.shape
        _require_dim_eq(K1, K2, "matmul: K dims must match")
        out_ty = TensorType(out_dtype or aty.dtype, (M, N))
        v = Value(self.fresh("y"), out_ty)
        op = Op("matmul", [a, b], out_ty)
        op.set_result(v)
        return op

    def add(self, x: Value, y: Value) -> Op:
        xt, yt = x.type, y.type
        _require(xt.dtype == yt.dtype, "add: dtype mismatch")

        xs, ys = xt.shape, yt.shape

        # Only support (MxN) + (N) or (MxN) + (1xN)
        _require(len(xs) == 2, "add: left must be rank-2 (MxN)")

        # Normalize bias to rank-2 (1xN) if needed
        if len(ys) == 1:
            ys_norm = (1, ys[0])
        elif len(ys) == 2:
            ys_norm = ys
            _require_dim_eq(ys_norm[0], 1, "add: only row-broadcast (1xN) supported")
        else:
            raise ValueError("add: right rank must be 1 (N) or 2 (1xN)")

        # Last dim must match (use dim comparer, not shape comparer)
        _require_dim_eq(xs[1], ys_norm[1], "add: last dim must match")

        out_ty = TensorType(xt.dtype, xs)
        v = Value(self.fresh("y"), out_ty)
        op = Op("add", [x, y], out_ty)
        op.set_result(v)
        return op

    def gelu(self, x: Value) -> Op:
        out_ty = TensorType(x.type.dtype, x.type.shape)
        v = Value(self.fresh("y"), out_ty)
        op = Op("gelu", [x], out_ty)
        op.set_result(v)
        return op

    def ret(self, x: Value) -> Op:
        op = Op("return", [x], x.type)
        return op

    def const(self, name: str, ty: TensorType, value=None) -> Op:
        """Constant tensor (e.g., weights, bias). Value can be stored in attrs if you like."""
        v = Value(self.fresh("c"), ty)
        op = Op("const", [], ty, attrs={"name": name, "value": value})
        op.set_result(v)
        return op

    def fused_mlp(self, X: Value, W: Value, b: Value) -> Op:
        """Fused matmul+bias+gelu: (MxK,@KxN)+N -> (MxN). Types must already match."""
        M, K1 = X.type.shape
        K2, N = W.type.shape
        _require_dim_eq(K1, K2, "fused_mlp: K dims must match")
        _require_shape_eq((N,), (b.type.shape[0],), "fused_mlp: bias last dim mismatch")
        out_ty = TensorType(X.type.dtype, (M, N))
        v = Value(self.fresh("y"), out_ty)
        op = Op("fused_mlp", [X, W, b], out_ty)
        op.set_result(v)
        return op


# ---- Assert helpers (symbol-aware) ----

def _require(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)

def _require_dim_eq(a, b, msg: str):
    # a, b: single dimensions (int or str symbol)
    if isinstance(a, int) and isinstance(b, int):
        if a != b:
            raise ValueError(msg + f" (got {a} vs {b})")
    elif isinstance(a, str) and isinstance(b, str):
        if a != b:
            raise ValueError(msg + f" (got {a} vs {b})")
    else:
        # Mixed int vs symbol: keep strict for now
        raise ValueError(msg + f" (got {a} vs {b})")
        

def _require_shape_eq(sa, sb, msg: str):
    # sa, sb: tuples of dims
    _require(len(sa) == len(sb), msg + f" (rank {len(sa)} vs {len(sb)})")
    for ia, (da, db) in enumerate(zip(sa, sb)):
        _require_dim_eq(da, db, msg + f" @dim{ia}")



# ===== Demo: build and verify Y = GELU(X @ W + b) =====

def build_mlp_module() -> Module:
    bld = IRBuilder()
    tX = TensorType("f32", ("M", "K"))
    tW = TensorType("f32", ("K", "N"))
    tb = TensorType("f32", ("N",))
    tY = TensorType("f32", ("M", "N"))

    opX = bld.input("X", tX)
    opW = bld.const("W", tW)     # <- const instead of input
    opb = bld.const("b", tb)     # <- const instead of input

    op0 = bld.matmul(opX.result, opW.result)
    op1 = bld.add(op0.result, opb.result)
    op2 = bld.gelu(op1.result)
    opR = bld.ret(op2.result)

    func = Func(
        name="mlp",
        args=[opX.result],  # only X is a runtime arg now
        result_type=tY,
        body=Block(ops=[opX, opW, opb, op0, op1, op2, opR])
    )
    _require_shape_eq(func.result_type.shape, op2.result.type.shape, "func result shape mismatch")
    return Module(funcs=[func])


if __name__ == "__main__":
    import numpy as np

    # Concrete dims for a quick smoke test
    M, K, N = 4, 8, 16

    # Make some test weights/bias (like a frozen model)
    rng = np.random.default_rng(0)
    W_val = rng.standard_normal((K, N), dtype=np.float32)   # IR expects [K,N]
    b_val = rng.standard_normal((N,),    dtype=np.float32)

    bld = IRBuilder()

    # Types with concrete dims for this test
    tX = TensorType("f32", (M, K))
    tW = TensorType("f32", (K, N))
    tb = TensorType("f32", (N,))
    tY = TensorType("f32", (M, N))

    # X is a runtime input; W and b are graph constants
    opX = bld.input("X", tX)
    opW = bld.const("W", tW, value=W_val)
    opb = bld.const("b", tb, value=b_val)

    # y = GELU( X @ W + b )
    op0 = bld.matmul(opX.result, opW.result)
    op1 = bld.add(op0.result, opb.result)
    op2 = bld.gelu(op1.result)
    opR = bld.ret(op2.result)

    func = Func(
        name="mlp_const_demo",
        args=[opX.result],  # only X is a runtime arg
        result_type=tY,
        body=Block(ops=[opX, opW, opb, op0, op1, op2, opR])
    )
    _require_shape_eq(func.result_type.shape, op2.result.type.shape, "func result shape mismatch")
    mod = Module(funcs=[func])

    # Pretty-print the IR
    print(mod)

    # Quick sanity on const payloads
    # (attrs["value"] holds the actual numpy arrays you passed)
    W_payload = opW.attrs.get("value", None)
    b_payload = opb.attrs.get("value", None)
    assert isinstance(W_payload, np.ndarray) and W_payload.shape == (K, N)
    assert isinstance(b_payload, np.ndarray) and b_payload.shape == (N,)
    print("\n[const check] W:", W_payload.shape, W_payload.dtype, " b:", b_payload.shape, b_payload.dtype)