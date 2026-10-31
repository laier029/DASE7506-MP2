"""Regression checks for calculation-only number coverage extraction."""

from question_number_coverage_probe import calculation_context


def main():
    bullet = "- One jar uses 7 strawberries.\n- Each jar costs 4 dollars."
    if calculation_context(bullet):
        raise AssertionError("Markdown bullets must not count as calculations")
    subtraction = "Remaining: 16 - 3 = 13"
    if "16 - 3 = 13" not in calculation_context(subtraction):
        raise AssertionError("Numeric subtraction should remain a calculation")
    print("question-number coverage checks passed")


if __name__ == "__main__":
    main()
