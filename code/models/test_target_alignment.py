"""Regression checks for terminal question-target retention."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from target_alignment import terminal_target_recall


def main():
    bank = [
        {
            "question": "What was the total reduction from the original price?",
            "candidates": [
                {"text": "The new price is 456.\n#### 456"},
                {"text": "The final total reduction is 44.\n#### 44"},
                {"text": "The total price is 500.\n#### 500"},
                {"text": "No target words.\n#### 1"},
            ],
        }
    ]
    feature = terminal_target_recall(bank)[0]
    if not feature[1] > feature[0] or feature[1] != feature[2]:
        raise AssertionError(f"unexpected target recall: {feature}")
    print("target alignment checks passed")


if __name__ == "__main__":
    main()
