# tilesmith/ir/__init__.py
from .tiny_ir import (
    TensorType, Value, Op, Block, Func, Module,
    IRBuilder, build_mlp_module,
)