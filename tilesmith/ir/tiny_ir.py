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
        _require_shape_eq(K1, K2, "matmul: K dims must match")
        out_ty = TensorType(out_dtype or aty.dtype, (M, N))
        v = Value(self.fresh("y"), out_ty)
        op = Op("matmul", [a, b], out_ty)
        op.set_result(v)
        return op

    def add(self, x: Value, y: Value) -> Op:
        # Simple rule: allow broadcast along the last dim if one operand is rank-1 (N) or (1xN)
        xt, yt = x.type, y.type
        _require_shape_eq(xt.dtype, yt.dtype, "add: dtype mismatch")
        xs, ys = xt.shape, yt.shape
        # Normalize bias shapes: (N) -> (1,N)
        ys_norm = ys
        if len(ys) == 1:
            ys_norm = (1, ys[0])
        # Now require xs and ys_norm be broadcast-compatible where ys_norm[0] can be 1
        _require(len(xs) == 2 and len(ys_norm) == 2, "add: only (MxN) + (N) or (1xN) supported")
        _require_shape_eq(xs[1], ys_norm[1], "add: last dim must match")
        # No constraint on xs[0] vs ys_norm[0] (broadcast over rows)
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
    b = IRBuilder()

    # Types with symbolic dims
    tX = TensorType("f32", ("M", "K"))
    tW = TensorType("f32", ("K", "N"))
    tb = TensorType("f32", ("N",))      # rank-1; will broadcast over rows
    tY = TensorType("f32", ("M", "N"))

    # Inputs
    opX = b.input("X", tX)
    opW = b.input("W", tW)
    opb = b.input("b", tb)

    # Ops: y0 = X@W; y1 = y0 + b; y2 = gelu(y1)
    op0 = b.matmul(opX.result, opW.result)
    op1 = b.add(op0.result, opb.result)
    op2 = b.gelu(op1.result)
    opR = b.ret(op2.result)

    func = Func(
        name="mlp",
        args=[opX.result, opW.result, opb.result],
        result_type=tY,
        body=Block(ops=[opX, opW, opb, op0, op1, op2, opR])
    )
    mod = Module(funcs=[func])
    # Verify result type matches return
    _require_shape_eq(func.result_type.shape, op2.result.type.shape, "func result shape mismatch")
    return mod


if __name__ == "__main__":
    mod = build_mlp_module()
    print(mod)