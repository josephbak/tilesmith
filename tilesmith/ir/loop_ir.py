# tilesmith/ir/loop_ir.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional, Dict

## Simple assert helper (used throughout)
def _require(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)


# -----------------------------
# Types
# -----------------------------

Index = int  # loop bounds are Python ints for now

@dataclass(frozen=True)
class ScalarType:
    # Keep it narrow; extend with i32/i64/f16/bf16 as needed
    kind: str  # "f32" | "f64" | "i32" | "i64"
    def __str__(self): return self.kind

@dataclass(frozen=True)
class MemRefType:
    # Buffer with row-major layout (contiguous); shape is concrete ints for now
    elem: ScalarType
    shape: Tuple[int, ...]
    def __str__(self): return f"memref<{self.elem}, { 'x'.join(map(str, self.shape)) }>"

Type = Union[ScalarType, MemRefType]

# -----------------------------
# IR Core
# -----------------------------

@dataclass
class Value:
    name: str
    type: Type
    def __str__(self): return f"{self.name}: {self.type}"

@dataclass
class Op:
    opname: str
    operands: List[Value]
    result_type: Optional[Type] = None
    attrs: Dict[str, object] = field(default_factory=dict)
    result: Optional[Value] = None
    region: Optional["Block"] = None  # for ops that own a block (For)

    def set_result(self, v: Value):
        self.result = v

    def set_region(self, block: "Block"):
        self.region = block

    def __str__(self):
        # Pretty-printer for common ops
        on = self.opname
        ops = ", ".join(o.name for o in self.operands)
        ty = f" : {self.result_type}" if self.result_type else ""
        if on in ("alloc", "const"):
            return f"{self.result.name} = {on}{ty}"
        if on in ("addf", "mulf", "fmaf", "gelu"):
            return f"{self.result.name} = {on} {ops}{ty}"
        if on == "load":
            idxs = ", ".join(map(str, self.attrs.get("indices", [])))
            return f"{self.result.name} = load {ops}[{idxs}] : {self.result_type}"
        if on == "store":
            idxs = ", ".join(map(str, self.attrs.get("indices", [])))
            return f"store {ops}[{idxs}]"
        if on == "for":
            lb, ub, st = self.attrs["lb"], self.attrs["ub"], self.attrs.get("step", 1)
            header = f"for {self.result.name} in [{lb}, {ub}) step {st} {{"
            body = "\n".join("  " + line for line in str(self.region).splitlines())
            return header + ("\n" + body if body else "") + "\n}"
        if on == "return":
            return f"return {ops}"
        return f"{self.result.name} = {on} {ops}{ty}"

@dataclass
class Block:
    args: List[Value] = field(default_factory=list)
    ops: List[Op] = field(default_factory=list)
    def __str__(self):
        lines = []
        for op in self.ops:
            lines.append(str(op))
        return "\n".join(lines)

@dataclass
class Func:
    name: str
    args: List[Value]
    body: Block
    def __str__(self):
        sig = ", ".join(f"{a.name}: {a.type}" for a in self.args)
        hdr = f"func @{self.name}({sig}) {{"
        body = "\n".join("  " + ln for ln in str(self.body).splitlines())
        return hdr + ("\n" + body if body else "") + "\n}"

@dataclass
class Module:
    funcs: List[Func]
    def __str__(self): return "\n\n".join(str(f) for f in self.funcs)

# -----------------------------
# Builder
# -----------------------------

class IRBuilder:
    def __init__(self):
        self._n = 0

    def fresh(self, p="v"):
        self._n += 1
        return f"%{p}{self._n}"

    # ---- Values / buffers ----
    def arg(self, ty: Type) -> Value:
        return Value(self.fresh("arg"), ty)

    def alloc(self, elem: ScalarType, shape: Tuple[int, ...]) -> Op:
        v = Value(self.fresh("buf"), MemRefType(elem, shape))
        op = Op("alloc", [], v.type)
        op.set_result(v)
        return op

    def const(self, ty: ScalarType, value: Union[float, int]) -> Op:
        v = Value(self.fresh("c"), ty)
        op = Op("const", [], ty, attrs={"value": value})
        op.set_result(v)
        return op

    # ---- Memory ----
    def load(self, buf: Value, indices: List[Value]) -> Op:
        _require(isinstance(buf.type, MemRefType), "load: memref required")
        _require(len(indices) == len(buf.type.shape), "load: rank mismatch")
        v = Value(self.fresh("x"), buf.type.elem)
        op = Op("load", [buf] + indices, v.type, attrs={"indices": [i.name for i in indices]})
        op.set_result(v)
        return op

    def store(self, val: Value, buf: Value, indices: List[Value]) -> Op:
        _require(isinstance(buf.type, MemRefType), "store: memref required")
        _require(len(indices) == len(buf.type.shape), "store: rank mismatch")
        _require(val.type == buf.type.elem, "store: elem type mismatch")
        op = Op("store", [val, buf] + indices, None, attrs={"indices": [i.name for i in indices]})
        return op

    # ---- Arithmetic ----
    def addf(self, a: Value, b: Value) -> Op:
        _require(isinstance(a.type, ScalarType) and a.type == b.type, "addf: scalar dtype mismatch")
        v = Value(self.fresh("s"), a.type)
        op = Op("addf", [a, b], a.type); op.set_result(v); return op

    def mulf(self, a: Value, b: Value) -> Op:
        _require(isinstance(a.type, ScalarType) and a.type == b.type, "mulf: scalar dtype mismatch")
        v = Value(self.fresh("s"), a.type)
        op = Op("mulf", [a, b], a.type); op.set_result(v); return op

    def fmaf(self, a: Value, b: Value, c: Value) -> Op:
        # fused multiply-add: a*b + c
        _require(all(isinstance(x.type, ScalarType) and x.type == a.type for x in (b, c)), "fmaf: scalar dtype mismatch")
        v = Value(self.fresh("s"), a.type)
        op = Op("fmaf", [a, b, c], a.type); op.set_result(v); return op

    def gelu(self, a: Value) -> Op:
        _require(isinstance(a.type, ScalarType), "gelu: scalar required")
        v = Value(self.fresh("s"), a.type)
        op = Op("gelu", [a], a.type); op.set_result(v); return op

    # ---- Control ----
    def for_(self, lb: Index, ub: Index, step: Index = 1, iv_name: Optional[str] = None) -> Op:
        iv = Value(iv_name or self.fresh("i"), ScalarType("i64"))
        op = Op("for", [], None, attrs={"lb": lb, "ub": ub, "step": step})
        # The induction variable is represented as the op "result" to bind a name
        op.set_result(iv)
        op.set_region(Block(args=[iv], ops=[]))
        return op

# -----------------------------
# Verifier (minimal)
# -----------------------------

def verify_module(mod: Module):
    for f in mod.funcs:
        verify_func(f)

def verify_func(fn: Func):
    # Extremely light checks: load/store ranks, scalar types, for-bounds
    def walk(b: Block):
        for op in b.ops:
            if op.opname in ("load", "store"):
                buf = op.operands[1 if op.opname == "store" else 0]
                _require(isinstance(buf.type, MemRefType), f"{op.opname}: buffer must be memref")
                if op.opname == "store":
                    val = op.operands[0]
                    _require(val.type == buf.type.elem, "store: value type != buffer elem")
            if op.opname == "for":
                lb, ub, step = op.attrs["lb"], op.attrs["ub"], op.attrs.get("step", 1)
                _require(isinstance(lb, int) and isinstance(ub, int) and isinstance(step, int), "for: integer bounds")
                _require(step > 0 and ub >= lb, "for: invalid bounds/step")
                walk(op.region)
    walk(fn.body)

# -----------------------------
# Example (matmul+bias+gelu lowered skeleton)
# -----------------------------

if __name__ == "__main__":
    # Tiny hand-crafted loop IR for: Y = GELU(X@W + b)
    b = IRBuilder()
    f32 = ScalarType("f32")
    i64 = ScalarType("i64")

    M, K, N = 4, 8, 16
    X = Value("%X", MemRefType(f32, (M, K)))
    W = Value("%W", MemRefType(f32, (K, N)))
    B = Value("%B", MemRefType(f32, (N,)))
    Y = b.alloc(f32, (M, N))  # result buffer

    body = Block(args=[X, W, B, Y.result], ops=[])

    # for i in [0,M)
    for_i = b.for_(0, M, 1, iv_name="%i"); body.ops.append(for_i)
    # for j in [0,N)
    for_j = b.for_(0, N, 1, iv_name="%j"); for_i.region.ops.append(for_j)

    # acc = 0.0
    c0 = b.const(f32, 0.0); for_j.region.ops.append(c0)
    acc = c0.result

    # for k in [0,K): acc = fmaf(X[i,k], W[k,j], acc)
    for_k = b.for_(0, K, 1, iv_name="%k"); for_j.region.ops.append(for_k)
    Xi = b.load(X, [for_i.result, for_k.result]); for_k.region.ops.append(Xi)
    Wk = b.load(W, [for_k.result, for_j.result]); for_k.region.ops.append(Wk)
    acc2 = b.fmaf(Xi.result, Wk.result, acc);     for_k.region.ops.append(acc2)
    # NOTE: SSA update pattern: in a real IR we’d use block arguments for carries; here we keep it simple
    acc = acc2.result

    # tmp = acc + b[j]
    bj = b.load(B, [for_j.result]);               for_j.region.ops.append(bj)
    tmp = b.addf(acc, bj.result);                 for_j.region.ops.append(tmp)

    # y = gelu(tmp)
    yv = b.gelu(tmp.result);                      for_j.region.ops.append(yv)

    # store y into Y[i,j]
    st = b.store(yv.result, Y.result, [for_i.result, for_j.result]); for_j.region.ops.append(st)

    # return (no multi-result; in a real pipeline we'd return Y buffer handle or nothing)
    ret = Op("return", [Y.result]); body.ops.append(ret)

    fn = Func("lowered_mlp", [X, W, B], body)
    mod = Module([fn])
    verify_module(mod)
    print(mod)
