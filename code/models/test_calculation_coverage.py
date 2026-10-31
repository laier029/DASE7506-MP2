"""Regression checks for calculation-only quantity coverage."""

from calculation_coverage import (
    calculation_context,
    calculation_occurrence_coverage,
    calculation_unique_coverage,
)


def main():
    bullet = "- One jar uses 7 strawberries.\n- Each jar costs 4 dollars."
    if calculation_context(bullet):
        raise AssertionError("Markdown bullets must not count as calculations")
    subtraction = "Remaining: 16 - 3 = 13"
    if "16 - 3 = 13" not in calculation_context(subtraction):
        raise AssertionError("Numeric subtraction should remain a calculation")
    inline = "Compute $16 - 3 = 13$ now."
    context = calculation_context(inline)
    if context.count("16 - 3 = 13") != 1:
        raise AssertionError(f"inline math was duplicated: {context!r}")
    currency = (
        "Tim hires two bodyguards. Each charges $20 per hour.\n"
        "### Step 2: weekly cost\n"
        "- Each day: $20/hour × 8 hours = $160\n"
        "- Weekly: $160 × 7 = $1120\n"
        "#### <numeric answer> 1120"
    )
    currency_context = calculation_context(currency)
    if "two bodyguards" in currency_context or "Step 2" in currency_context:
        raise AssertionError(
            f"currency markers created a false math fragment: {currency_context!r}"
        )
    if "20/hour × 8 hours" not in currency_context:
        raise AssertionError(
            f"currency calculation line was lost: {currency_context!r}"
        )

    question = (
        "Betty picked 16. Matthew picked 20 more and twice Natalie. "
        "One jar uses 7 and sells for 4."
    )
    bank = [
        {
            "question": question,
            "candidates": [
                {"text": "$16+20=36$ and $36/2=18$ and $70/7=10$ and $10*4=40$"},
                {"text": "$16+20=36$ and $2*16=32$ and $84/7=12$ and $12*4=48$"},
                {"text": "No arithmetic."},
                {"text": "$16+20=36$"},
            ],
        }
    ]
    coverage = calculation_unique_coverage(bank)[0]
    if coverage[0] != coverage[1]:
        raise AssertionError(
            "Bullet formatting must not give the wrong candidate extra coverage"
        )
    if not coverage[0] > coverage[3] > coverage[2]:
        raise AssertionError(f"unexpected coverage ordering: {coverage}")
    occurrence = calculation_occurrence_coverage(bank)[0]
    if occurrence.shape != (4,) or not occurrence[0] > occurrence[2]:
        raise AssertionError(f"unexpected occurrence coverage: {occurrence}")
    print("calculation coverage checks passed")


if __name__ == "__main__":
    main()
