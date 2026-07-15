import unittest

from scripts.analysis.generate_general_qa_cases_deepseek import (
    category_targets,
    normalize_case,
    normalized_prompt,
)


class GenerateGeneralQACasesTest(unittest.TestCase):
    def test_default_targets_sum_to_200(self):
        targets = category_targets(200)
        self.assertEqual(sum(targets.values()), 200)
        self.assertEqual(targets["稳定事实"], 40)
        self.assertEqual(targets["基础科学"], 40)

    def test_arbitrary_targets_preserve_total(self):
        self.assertEqual(sum(category_targets(73).values()), 73)

    def test_normalize_case_accepts_open_question(self):
        case = normalize_case(
            {
                "prompt": "为什么白天的天空通常呈蓝色？",
                "reference_answer": "大气分子对短波长光的瑞利散射更强。",
                "reference_points": ["太阳光包含不同波长", "瑞利散射对蓝光更强"],
                "requirements": ["不得解释为海洋反射"],
            },
            "基础科学",
        )
        self.assertFalse(case["is_code"])
        self.assertEqual(case["category"], "基础科学")

    def test_normalize_case_rejects_multiple_choice(self):
        with self.assertRaisesRegex(ValueError, "multiple-choice"):
            normalize_case(
                {
                    "prompt": "选择正确答案：\nA. 甲\nB. 乙",
                    "reference_answer": "甲",
                    "reference_points": ["甲", "原因"],
                    "requirements": ["选择"],
                },
                "稳定事实",
            )

    def test_prompt_normalization_detects_punctuation_only_changes(self):
        self.assertEqual(normalized_prompt("天空为什么是蓝色？"), normalized_prompt("天空为什么是蓝色"))


if __name__ == "__main__":
    unittest.main()
