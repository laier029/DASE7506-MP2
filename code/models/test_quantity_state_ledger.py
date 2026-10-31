"""Regression checks for the generic quantity/state ledger."""

import unittest

from quantity_state_ledger import FEATURE_NAMES, quantity_state_ledger_features


def _candidate(text, answer):
    return {"index": 0, "text": f"{text}\n#### {answer}"}


class QuantityStateLedgerTests(unittest.TestCase):
    def test_terminal_total_has_broader_dependency_than_intermediate(self):
        question = (
            "Helga tried 7 pairs, then 2 more than that, then 0. "
            "At the last store she tried twice the first three stores combined. "
            "What is the total number of pairs she tried?"
        )
        wrong = _candidate("7 + 2 = 9\n7 + 9 + 0 = 16\n2 * 16 = 32", 32)
        correct = _candidate(
            "7 + 2 = 9\n7 + 9 + 0 = 16\n2 * 16 = 32\n16 + 32 = 48",
            48,
        )
        filler = _candidate("7 + 2 = 9", 9)
        bank = [
            {
                "id": "synthetic",
                "question": question,
                "candidates": [wrong, correct, filler, filler],
            }
        ]
        features = quantity_state_ledger_features(bank)[0]
        leaf = FEATURE_NAMES.index("aggressive_leaf_log")
        depth = FEATURE_NAMES.index("aggressive_depth")
        growth = FEATURE_NAMES.index("terminal_leaf_growth")
        self.assertGreater(features[1, leaf], features[0, leaf])
        self.assertGreater(features[1, depth], features[0, depth])
        self.assertGreater(features[1, growth], features[0, growth])

    def test_missing_final_equation_is_visible(self):
        question = "A box starts with 10 balls and receives 5. How many now?"
        direct = _candidate("10 + 5 = 15", 15)
        unsupported = _candidate("The answer is five.", 5)
        bank = [
            {
                "id": "synthetic",
                "question": question,
                "candidates": [direct, unsupported, unsupported, unsupported],
            }
        ]
        features = quantity_state_ledger_features(bank)[0]
        has_equation = FEATURE_NAMES.index("has_final_equation")
        self.assertEqual(features[0, has_equation], 1.0)
        self.assertEqual(features[1, has_equation], 0.0)

    def test_reverse_equation_is_parsed(self):
        question = "There are 4 groups of 3. How many items?"
        reverse = _candidate("12 = 4 * 3", 12)
        filler = _candidate("4 + 3 = 7", 7)
        bank = [
            {
                "id": "synthetic",
                "question": question,
                "candidates": [reverse, filler, filler, filler],
            }
        ]
        features = quantity_state_ledger_features(bank)[0]
        has_equation = FEATURE_NAMES.index("has_final_equation")
        multiply = FEATURE_NAMES.index("final_multiply_share")
        self.assertEqual(features[0, has_equation], 1.0)
        self.assertEqual(features[0, multiply], 1.0)


if __name__ == "__main__":
    unittest.main()
