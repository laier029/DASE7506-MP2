"""Minimal regression checks for the conservative arithmetic parser."""

from arithmetic_consistency import (
    arithmetic_counts,
    broad_arithmetic_counts,
    broad_arithmetic_mismatches,
    plain_text_arithmetic_counts,
)


def main():
    cases = {
        "$$7 * 12 = 84$$": (1, 0),
        "$$7 * 12 = 72$$": (1, 1),
        "$$x = 4 + 6 = 10$$": (1, 0),
        "$$\\frac{7}{1.4} = 5$$": (1, 0),
        "$$6 \\text{ hours} = 6 * 60 = 360 \\text{ minutes}$$": (0, 0),
    }
    for text, expected in cases.items():
        actual = arithmetic_counts(text)
        if actual != expected:
            raise AssertionError(f"{text}: expected {expected}, got {actual}")
    broad_cases = {
        "$$0.6 / 9 = 2 / 15 \\text{ hours}$$": (1, 1),
        "$$\\text{Population} = 15 * 58940 = 883100$$": (1, 1),
        "$$5 * 17 = 85 \\text{ ounces}$$": (1, 0),
        "$$255 \\text{ minutes} = 255 / 60 = 4.25 \\text{ hours}$$": (1, 0),
        "$$75\\% * 40 = 0.75 * 40 = 30$$": (2, 0),
        "$$8 / 25 * 100 = 32\\%$$": (1, 0),
        "$$0.25 = 25\\%$$": (1, 0),
        "$$8 / 25 * 100 = 31\\%$$": (1, 1),
        "$$t = \\frac{0.6}{9} = 0.0667 \\text{ hours}$$": (1, 0),
    }
    for text, expected in broad_cases.items():
        actual = broad_arithmetic_counts(text)
        if actual != expected:
            raise AssertionError(f"broad {text}: expected {expected}, got {actual}")
    mismatch = broad_arithmetic_mismatches(
        "$$\\frac{0.6}{9} = \\frac{2}{15} \\text{ hours}$$"
    )
    if mismatch != [((0.066667,), (0.133333,))]:
        raise AssertionError(f"unexpected mismatch signature: {mismatch}")
    plain_cases = {
        "- Cost per orange: $12.50 / 25 = $0.50": (1, 0),
        "- Population: 15 * 58940 = 883100 people": (1, 1),
        "Total = 5 + 5 = 11": (1, 1),
        "1 hour = 60 minutes": (0, 0),
        "$5 + 5 = 10$": (1, 0),
    }
    for text, expected in plain_cases.items():
        actual = plain_text_arithmetic_counts(text)
        if actual != expected:
            raise AssertionError(f"plain {text}: expected {expected}, got {actual}")
    print("arithmetic consistency checks passed")


if __name__ == "__main__":
    main()
