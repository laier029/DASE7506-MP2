"""Regression checks for explicit final-declaration group filtering."""

from final_declaration_consistency import (
    declared_final_answer,
    final_declaration_mismatch_groups,
    has_final_declaration_mismatch,
)


def main():
    mismatch = (
        "#### Job 3\n1 toilet.\n"
        "### Final Answer:\n$$\\boxed{250}$$"
    )
    if declared_final_answer(mismatch) != "250":
        raise AssertionError("explicit final answer was not parsed")
    if not has_final_declaration_mismatch(mismatch):
        raise AssertionError("protocol/declaration conflict was not detected")
    bank = [
        {
            "candidates": [
                {"text": mismatch},
                {"text": "#### <numeric answer> 250"},
                {"text": "#### <numeric answer> 300"},
                {"text": "No parseable answer"},
            ]
        }
    ]
    rejected = final_declaration_mismatch_groups(bank)
    if rejected != [{"3"}]:
        raise AssertionError(f"unexpected rejected groups: {rejected}")
    print("final declaration consistency checks passed")


if __name__ == "__main__":
    main()
