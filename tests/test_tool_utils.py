import unittest

from trainer.tool_utils import validate_gt_in_text


class ToolUtilsTest(unittest.TestCase):
    def test_numeric_gt_does_not_match_inside_larger_number(self):
        self.assertEqual(validate_gt_in_text("结果是420", ["20"]), set())

    def test_numeric_gt_matches_when_adjacent_to_chinese_text(self):
        self.assertEqual(validate_gt_in_text("计算结果等于20758280。", ["20758280"]), {"20758280"})

    def test_numeric_gt_matches_equivalent_decimal_and_commas(self):
        self.assertEqual(validate_gt_in_text("结果是 2,935.0", ["2935"]), {"2935"})

    def test_negative_number_uses_complete_numeric_token(self):
        self.assertEqual(validate_gt_in_text("最终答案为 -285。", ["-285"]), {"-285"})

    def test_multiple_numeric_ground_truth_values(self):
        actual = validate_gt_in_text("答案分别是13575、40878和4。", ["13575", "40878", "4"])
        self.assertEqual(actual, {"13575", "40878", "4"})

    def test_non_numeric_ground_truth_is_case_insensitive(self):
        self.assertEqual(validate_gt_in_text("Translation: Hello World", ["hello world"]), {"hello world"})


if __name__ == "__main__":
    unittest.main()
