"""Regression checks for answer/last-calculation comparison."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer_step_consistency import (
    answer_step_consistency,
    calculation_values,
    last_step_value,
)


def main():
    cases = {
        "Cost: $100 - $84 = $16\n#### 16": (16.0, True),
        "$$5 \\times 17 = 85 \\text{ ounces}$$\n### Final Answer:\n\\boxed{85}": (
            85.0,
            True,
        ),
        "Total = 12 + 18 + 4 = **34 miles**\n#### 34": (34.0, True),
        "$$7 * 12 = 84$$\n#### 72": (84.0, False),
        "$$19 / 2 = 9.5$$\n#### 19/2": (9.5, True),
        "$$20 / 350 * 100 \\approx 5.71\\%$$\n#### 6": (5.71, True),
        "$$24 \\times \\$5 = \\$120$$\n#### 120": (120.0, True),
        "#### Step 1\n$$5 + 5 = 10$$\n### Final Answer:\n#### 10": (
            10.0,
            True,
        ),
        "There are eighty items.\n#### 80": (None, None),
        # This deliberately counts as a mismatch: the derived 80 is prose,
        # not an explicit calculation.  Such cases motivate a small penalty.
        "Current total = 220.\nWe still need 80 more.\n#### 80": (220.0, False),
    }
    for text, expected in cases.items():
        actual = (last_step_value(text), answer_step_consistency(text))
        if actual != expected:
            raise AssertionError(f"{text!r}: expected {expected}, got {actual}")
    values = calculation_values("$2 + 3 = 5$\n$5 * 4 = 20$\n#### 20")
    if values != [5.0, 20.0]:
        raise AssertionError(f"expected ordered values [5.0, 20.0], got {values}")
    print("answer/last-step consistency checks passed")


if __name__ == "__main__":
    main()

