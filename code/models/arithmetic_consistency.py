"""Conservative arithmetic-equation checks for candidate reasoning text."""

import ast
import math
import re


_DISPLAY_MATH = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", re.DOTALL)
_NUMERIC_EXPRESSION = re.compile(r"^[0-9,.+\-*/()\s]+$")
_NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?"
_SIMPLE_FRACTION = re.compile(
    rf"\\frac\s*\{{\s*({_NUMBER})\s*\}}\s*\{{\s*({_NUMBER})\s*\}}"
)


def _normalize(fragment):
    fragment = fragment.replace("−", "-").replace("–", "-")
    fragment = fragment.replace("×", "*").replace("÷", "/")
    fragment = fragment.replace("\\times", "*")
    fragment = fragment.replace("\\cdot", "*")
    fragment = fragment.replace("\\div", "/")
    while _SIMPLE_FRACTION.search(fragment):
        fragment = _SIMPLE_FRACTION.sub(r"(\1/\2)", fragment)
    fragment = re.sub(r"\\boxed\s*\{([^{}]+)\}", r"\1", fragment)
    fragment = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", fragment)
    return fragment.replace("\\,", " ").replace("&", "")


def _safe_numeric_value(expression):
    expression = expression.strip().replace(",", "")
    if not expression or not _NUMERIC_EXPRESSION.fullmatch(expression):
        raise ValueError("Not a numeric-only arithmetic expression")

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ValueError("Division by zero")
            return left / right
        raise ValueError("Unsupported arithmetic expression")

    return evaluate(tree)


def arithmetic_counts(text):
    """Return `(checked, incorrect)` for high-confidence numeric equations.

    Expressions containing `\\text{...}` are skipped deliberately because a
    numeric change can be a valid unit conversion rather than an arithmetic
    error. Symbolic and unsupported expressions are ignored instead of guessed.
    """
    display_fragments = _DISPLAY_MATH.findall(text)
    remaining_text = _DISPLAY_MATH.sub(" ", text)
    fragments = display_fragments + _INLINE_MATH.findall(remaining_text)

    checked = 0
    incorrect = 0

    for fragment in fragments:
        if "\\text" in fragment:
            continue
        normalized = _normalize(fragment)
        lines = re.split(r"\\\\|\n|\\Rightarrow|\\implies", normalized)

        for line in lines:
            if "=" not in line:
                continue

            values = []
            for part in line.split("="):
                try:
                    values.append(_safe_numeric_value(part))
                except (SyntaxError, ValueError, ZeroDivisionError):
                    values.append(None)

            previous = None
            for value in values:
                if value is None:
                    previous = None
                    continue
                if previous is not None:
                    checked += 1
                    consistent = math.isclose(
                        previous, value, rel_tol=1e-6, abs_tol=1e-6
                    )
                    incorrect += int(not consistent)
                previous = value

    return checked, incorrect


def arithmetic_error_rate(text):
    """Return the fraction of checked equalities that are inconsistent."""
    checked, incorrect = arithmetic_counts(text)
    return incorrect / checked if checked else 0.0


def _broad_numeric_values(part):
    """Return plausible numeric readings for a broad-check equation side.

    A terminal percentage can be either a ratio (``0.25 = 25%``) or a
    displayed percentage value after multiplication by 100
    (``1 / 4 * 100 = 25%``).  Keeping both readings avoids declaring either
    conventional notation style arithmetically wrong.
    """
    fraction_form = re.sub(
        r"(\d+(?:\.\d+)?)\s*\\?%", r"(\1/100)", part
    )
    try:
        values = [_safe_numeric_value(fraction_form)]
    except (SyntaxError, ValueError, ZeroDivisionError):
        values = []

    display_match = re.fullmatch(
        r"\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\\?%\s*", part
    )
    if display_match:
        display_value = float(display_match.group(1).replace(",", ""))
        if not any(math.isclose(display_value, value) for value in values):
            values.append(display_value)
    return tuple(values) if values else None


def _broad_equation_comparisons(text):
    """Yield adjacent numeric readings from broad-check equation chains."""
    display_fragments = _DISPLAY_MATH.findall(text)
    remaining_text = _DISPLAY_MATH.sub(" ", text)
    fragments = display_fragments + _INLINE_MATH.findall(remaining_text)

    for fragment in fragments:
        normalized = _normalize(fragment)
        lines = re.split(r"\\\\|\n|\\Rightarrow|\\implies", normalized)
        for line in lines:
            if "=" not in line:
                continue
            values = []
            parts = line.split("=")
            for index, part in enumerate(parts):
                if "\\text" in part and index < len(parts) - 1:
                    values.append(None)
                    continue
                part = re.sub(r"\\text\s*\{[^{}]*\}", " ", part)
                values.append(_broad_numeric_values(part))
            previous = None
            for value in values:
                if value is None:
                    previous = None
                    continue
                if previous is not None:
                    yield previous, value
                previous = value


def _broad_values_match(left_values, right_values):
    return any(
        math.isclose(left, right, rel_tol=1e-3, abs_tol=1e-3)
        for left in left_values
        for right in right_values
    )


def broad_arithmetic_mismatches(text):
    """Return stable numeric signatures for inconsistent broad equations."""
    mismatches = []
    for left_values, right_values in _broad_equation_comparisons(text):
        if _broad_values_match(left_values, right_values):
            continue
        mismatches.append(
            (
                tuple(round(value, 6) for value in left_values),
                tuple(round(value, 6) for value in right_values),
            )
        )
    return mismatches


def broad_arithmetic_counts(text):
    """Check numeric equation chains after removing LaTeX unit labels.

    This broader variant recovers arithmetic checks hidden behind ``\\text``
    labels.  It is intentionally kept separate because unit conversions can
    create false positives and should receive a smaller downstream penalty.
    """
    checked = 0
    incorrect = 0
    for previous, value in _broad_equation_comparisons(text):
        checked += 1
        # The broader pass sees unit conversions that often use a rounded
        # intermediate (for example, ``0.0667 hours * 60 = 4 minutes``).  A
        # 0.1% tolerance accepts that displayed rounding while still rejecting
        # substantive errors such as ``0.6 / 9 = 2 / 15``.
        incorrect += int(not _broad_values_match(previous, value))
    return checked, incorrect


def plain_text_arithmetic_counts(text):
    """Check explicit numeric equations written outside LaTeX math blocks."""
    remaining = _DISPLAY_MATH.sub(" ", text)
    checked = 0
    incorrect = 0
    for raw_line in remaining.splitlines():
        if "=" not in raw_line:
            continue
        line = _normalize(raw_line)
        line = line.replace("$", " ").replace("**", "").replace("`", "")
        line = re.sub(r"(\d+(?:\.\d+)?)\s*\\?%", r"(\1/100)", line)
        line = re.sub(r"(?<=\d)c\b", "", line, flags=re.IGNORECASE)
        parts = line.split("=")
        parsed = []
        for index, part in enumerate(parts):
            # The left side may contain a prose label; later sides may end in
            # a plain-text unit.  Retain only the adjacent numeric expression.
            if index == 0:
                match = re.search(r"([0-9,.+\-*/()\s]+)\s*$", part)
            else:
                match = re.match(r"\s*([0-9,.+\-*/()\s]+)", part)
            expression = match.group(1).strip() if match else ""
            try:
                value = _safe_numeric_value(expression)
            except (SyntaxError, ValueError, ZeroDivisionError):
                value = None
            has_operation = bool(re.search(r"(?:\d|\))\s*[+*/-]\s*(?:\d|\()", expression))
            parsed.append((value, has_operation))

        previous = None
        for current in parsed:
            if current[0] is None:
                previous = None
                continue
            if previous is not None and (previous[1] or current[1]):
                checked += 1
                incorrect += int(
                    not math.isclose(
                        previous[0], current[0], rel_tol=1e-6, abs_tol=1e-6
                    )
                )
            previous = current
    return checked, incorrect
