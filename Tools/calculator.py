"""
Calculator tool for the research/voice agent.

SECURITY NOTE:
    This deliberately does NOT use `eval()` or `exec()`. The input string
    here is ultimately constructed by an LLM from user speech/text, which
    makes raw `eval()` a real code-execution risk (e.g. an expression
    like "__import__('os').system('...')" would run arbitrary code).
    Instead, the expression is parsed into a Python AST and walked
    manually, allowing only a fixed whitelist of arithmetic operators,
    number literals, and a small set of math functions/constants —
    anything else (names, attribute access, calls to non-whitelisted
    functions, comprehensions, etc.) is rejected before evaluation.

Enhancements/design choices consistent with the rest of this codebase:
  - `logging` instead of `print`, so failed expressions and rejected
    (unsafe) inputs are visible in the same audit-style log stream as
    the permission gate.
  - Returns a clear, structured error string instead of letting an
    exception propagate into the agent loop, since a bad expression
    (division by zero, malformed syntax, disallowed operation) is
    expected user-facing input, not a bug.
  - Numeric results are formatted to avoid noisy float artifacts
    (e.g. `0.1 + 0.2` -> `0.3`, not `0.30000000000000004`) while still
    preserving precision for results that need it.
"""

import ast
import logging
import math
import operator
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


logger = logging.getLogger("hasini.tools.calculator")


# --------------------------------------------------------------------
# Whitelisted operators and functions. Nothing outside these can run.
# --------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "min": min,
    "max": max,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

# Guard against pathological inputs (e.g. absurdly large exponents)
# that could otherwise hang or exhaust memory.
_MAX_EXPRESSION_LENGTH = 200
_MAX_POWER_EXPONENT = 1000
_MAX_FACTORIAL_INPUT = 1000


class CalculatorError(ValueError):
    """Raised for invalid, unsafe, or unevaluable expressions."""


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"Unsupported constant: {node.value!r}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise CalculatorError(f"Operator not allowed: {op_type.__name__}")

        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        if op_type is ast.Pow and abs(right) > _MAX_POWER_EXPONENT:
            raise CalculatorError("Exponent too large")

        try:
            return _BIN_OPS[op_type](left, right)
        except ZeroDivisionError as exc:
            raise CalculatorError("Division by zero") from exc

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise CalculatorError(f"Unary operator not allowed: {op_type.__name__}")
        return _UNARY_OPS[op_type](_safe_eval(node.operand))

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise CalculatorError(f"Unknown identifier: {node.id!r}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            name = getattr(node.func, "id", "<expression>")
            raise CalculatorError(f"Function not allowed: {name!r}")
        if node.keywords:
            raise CalculatorError("Keyword arguments are not supported")

        fn_name = node.func.id
        args = [_safe_eval(arg) for arg in node.args]

        if fn_name == "factorial" and args and args[0] > _MAX_FACTORIAL_INPUT:
            raise CalculatorError("Factorial input too large")

        try:
            return _FUNCTIONS[fn_name](*args)
        except (ValueError, OverflowError) as exc:
            raise CalculatorError(f"{fn_name}(...) failed: {exc}") from exc

    raise CalculatorError(f"Expression element not allowed: {type(node).__name__}")


def evaluate_expression(expression: str) -> float:
    """Safely evaluate a basic arithmetic/math expression string.

    Raises CalculatorError on any invalid, unsafe, or unevaluable input.
    """
    if not expression or not expression.strip():
        raise CalculatorError("Empty expression")

    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise CalculatorError(
            f"Expression too long (max {_MAX_EXPRESSION_LENGTH} characters)"
        )

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid expression syntax: {exc}") from exc

    return _safe_eval(parsed)


def _format_result(value: float) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "undefined (NaN)"
        if math.isinf(value):
            return "infinity" if value > 0 else "-infinity"
        if value.is_integer():
            return str(int(value))
        # Round for display to avoid float noise, but keep enough
        # precision to be useful (e.g. financial-style calculations).
        return f"{value:.10g}"
    return str(value)


class CalculatorInput(BaseModel):
    expression: str = Field(
        description=(
            "A math expression to evaluate, e.g. '2 + 2 * 3', 'sqrt(16)', "
            "'(150 - 12.5) / 3', 'sin(pi / 2)'. Supports +, -, *, /, //, %, "
            "**, and functions: abs, round, sqrt, sin, cos, tan, log, "
            "log10, log2, exp, floor, ceil, factorial, min, max. "
            "Constants pi, e, tau are available."
        )
    )


def _calculate(expression: str) -> str:
    logger.debug("Evaluating expression: %r", expression)
    try:
        result = evaluate_expression(expression)
    except CalculatorError as exc:
        logger.info("Rejected/invalid calculator expression %r: %s", expression, exc)
        return f"Could not calculate '{expression}': {exc}"
    except Exception as exc:
        # Anything unforeseen still shouldn't crash the agent loop.
        logger.error(
            "Unexpected error evaluating %r", expression, exc_info=True
        )
        return f"Could not calculate '{expression}': unexpected error ({exc})"

    formatted = _format_result(result)
    logger.debug("Result: %r -> %s", expression, formatted)
    return formatted


calculator_tool = StructuredTool.from_function(
    func=_calculate,
    name="calculator",
    description=(
        "Evaluate a math expression and return the numeric result. "
        "Use for arithmetic, percentages, trig, logs, roots, etc. "
        "Does not handle unit conversion or symbolic algebra."
    ),
    args_schema=CalculatorInput,
)


if __name__ == "__main__":
    # Quick manual smoke test.
    logging.basicConfig(level=logging.DEBUG)
    for expr in [
        "2 + 2 * 3",
        "sqrt(16)",
        "(150 - 12.5) / 3",
        "sin(pi / 2)",
        "10 / 0",
        "2 ** 10000",
        "__import__('os').system('echo unsafe')",
        "factorial(5)",
        "1e400",
    ]:
        print(f"{expr!r} -> {_calculate(expr)}")