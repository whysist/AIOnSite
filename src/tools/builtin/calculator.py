"""Deterministic calculator and threshold deviation tools."""

from __future__ import annotations

import ast
import operator
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from src.core.exceptions import ToolExecutionError
from src.tools.base_tool import BaseTool, ToolPermission

# ---------------------------------------------------------------------------
# Calculator Tool
# ---------------------------------------------------------------------------

class CalculatorInput(BaseModel):
    expression: str = Field(..., description="Arithmetic expression (e.g. '2 * (3 + 4)')")

class CalculatorOutput(BaseModel):
    result: float

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("division by zero")
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    raise ValueError("unsupported syntax or operator")

class CalculatorTool(BaseTool[CalculatorInput]):
    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = "Safely evaluate deterministic arithmetic expressions."
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.PURE,)
    InputModel: ClassVar[type[BaseModel]] = CalculatorInput
    OutputModel: ClassVar[type[BaseModel] | None] = CalculatorOutput

    async def _run(self, args: CalculatorInput) -> dict[str, Any]:
        expr = args.expression.strip()
        try:
            parsed = ast.parse(expr, mode="eval")
            val = _safe_eval(parsed)
            return {"result": float(val)}
        except ZeroDivisionError as exc:
            raise ToolExecutionError(f"division by zero: {exc}") from exc
        except Exception as exc:
            raise ToolExecutionError(f"invalid expression: {exc}") from exc

# ---------------------------------------------------------------------------
# Deviation Tool
# ---------------------------------------------------------------------------

class DeviationInput(BaseModel):
    actual: float = Field(..., description="Observed parameter value")
    limit: float = Field(..., description="Operating or safety limit")

class DeviationOutput(BaseModel):
    actual: float
    limit: float
    absolute_deviation: float
    percent_deviation: float
    within_limit: bool

class DeviationTool(BaseTool[DeviationInput]):
    name: ClassVar[str] = "calculate_deviation"
    description: ClassVar[str] = "Compute absolute and percentage deviation against limits."
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.PURE,)
    InputModel: ClassVar[type[BaseModel]] = DeviationInput
    OutputModel: ClassVar[type[BaseModel] | None] = DeviationOutput

    async def _run(self, args: DeviationInput) -> dict[str, Any]:
        abs_dev = round(args.actual - args.limit, 4)
        pct_dev = round(((args.actual - args.limit) / args.limit) * 100.0, 2) if args.limit != 0 else 0.0
        within = args.actual <= args.limit

        return {
            "actual": args.actual,
            "limit": args.limit,
            "absolute_deviation": abs_dev,
            "percent_deviation": pct_dev,
            "within_limit": within,
        }