"""Deterministic arithmetic tools.

``CalculatorTool`` evaluates a restricted arithmetic expression using an
AST walk -- no ``eval``, no names, no calls, no attribute access.
``DeviationTool`` is the domain calculation from the inspection roadmap
(actual vs approved limit) and returns inputs + formula + output so the
verification layer can independently recompute it.
"""

from __future__ import annotations

import ast
import operator
from typing import ClassVar

from pydantic import BaseModel, Field

from ...core.exceptions import ToolExecutionError
from ..base_tool import BaseTool, ToolPermission

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ToolExecutionError("only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
            raise ToolExecutionError("exponent out of allowed range")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_node(node.operand))
    raise ToolExecutionError(f"disallowed expression element: {ast.dump(node)}")


class _CalcIn(BaseModel):
    expression: str = Field(..., max_length=200, description="e.g. '(120 - 100) / 100 * 100'")


class _CalcOut(BaseModel):
    expression: str
    result: float


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate a basic arithmetic expression (+ - * / // % **). No variables or functions."
    permissions: ClassVar = (ToolPermission.PURE,)
    InputModel = _CalcIn
    OutputModel = _CalcOut

    async def _run(self, args: _CalcIn) -> _CalcOut:
        try:
            tree = ast.parse(args.expression, mode="eval")
        except SyntaxError as exc:
            raise ToolExecutionError(f"could not parse expression: {exc}") from exc
        try:
            value = _eval_node(tree)
        except ZeroDivisionError:
            raise ToolExecutionError("division by zero") from None
        return _CalcOut(expression=args.expression, result=value)


class _DevIn(BaseModel):
    actual: float
    limit: float
    label: str | None = None


class _DevOut(BaseModel):
    actual: float
    limit: float
    absolute_deviation: float
    percent_deviation: float | None
    within_limit: bool
    formula: str
    label: str | None = None


class DeviationTool(BaseTool):
    name = "calculate_deviation"
    description = (
        "Compute absolute and percentage deviation of an observed value from an "
        "approved limit. Deterministic; returns inputs, formula and output."
    )
    permissions: ClassVar = (ToolPermission.PURE,)
    InputModel = _DevIn
    OutputModel = _DevOut

    async def _run(self, args: _DevIn) -> _DevOut:
        abs_dev = args.actual - args.limit
        pct = (abs_dev / args.limit * 100.0) if args.limit != 0 else None
        return _DevOut(
            actual=args.actual,
            limit=args.limit,
            absolute_deviation=round(abs_dev, 6),
            percent_deviation=round(pct, 4) if pct is not None else None,
            within_limit=args.actual <= args.limit,
            formula="absolute = actual - limit ; percent = absolute / limit * 100",
            label=args.label,
        )
