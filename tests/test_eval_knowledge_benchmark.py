import unittest

from scripts.analysis.prepare_knowledge_benchmark import (
    arc_parquet_urls,
    normalize_arc,
    normalize_four_choice,
    subject_parquet_urls,
    stratified_take,
)
from scripts.eval_knowledge_benchmark import (
    fixed_random_predictions,
    parse_choice,
)


class KnowledgeBenchmarkTest(unittest.TestCase):
    def test_four_choice_uses_row_index_when_source_has_no_id(self):
        row = {
            "Question": "题目",
            "A": "甲",
            "B": "乙",
            "C": "丙",
            "D": "丁",
            "Answer": "B",
        }
        result = normalize_four_choice("cmmlu", "subject", "test", row, row_index=7)
        self.assertEqual(result["id"], "cmmlu:subject:7")

    def test_parse_choice_prefers_explicit_final_marker(self):
        text = "A看起来有一定道理，但根据题意应选择B。\n答案：B"
        self.assertEqual(parse_choice(text, list("ABCD")), ("B", "explicit_marker"))

    def test_parse_choice_accepts_fullwidth_letter(self):
        self.assertEqual(parse_choice("答案：Ｃ", list("ABCD"))[0], "C")

    def test_parse_choice_rejects_ambiguous_letters(self):
        self.assertEqual(parse_choice("A和B都可能", list("ABCD")), (None, "unparsed"))

    def test_parse_choice_can_match_unique_choice_text_prefix(self):
        parsed = parse_choice(
            "竹子，因为大熊猫主要以竹子为食。",
            list("ABCD"),
            {"A": "鱼", "B": "竹子", "C": "肉", "D": "水果"},
        )
        self.assertEqual(parsed, ("B", "choice_text_prefix"))

    def test_arc_numeric_labels_are_canonicalized(self):
        row = {
            "id": "x",
            "question": "Which one?",
            "choices": {"label": ["1", "2", "3"], "text": ["x", "y", "z"]},
            "answerKey": "2",
        }
        result = normalize_arc(row, "validation")
        self.assertEqual(result["answer"], "B")
        self.assertEqual([item["label"] for item in result["choices"]], ["A", "B", "C"])

    def test_arc_parquet_urls_try_configured_endpoint_then_official(self):
        urls = arc_parquet_urls("validation", "https://hf-mirror.com/")
        self.assertEqual(len(urls), 2)
        self.assertTrue(urls[0].startswith("https://hf-mirror.com/"))
        self.assertTrue(urls[1].startswith("https://huggingface.co/"))
        self.assertTrue(urls[0].endswith("ARC-Easy/validation-00000-of-00001.parquet"))

    def test_subject_parquet_url_contains_repo_config_and_split(self):
        urls = subject_parquet_urls(
            "ceval/ceval-exam", "middle_school_history", "val", None
        )
        self.assertEqual(
            urls,
            [
                "https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/"
                "middle_school_history/val-00000-of-00001.parquet"
            ],
        )

    def test_cmmlu_parquet_url_uses_published_snapshot(self):
        urls = subject_parquet_urls(
            "lmlmcat/cmmlu", "elementary_chinese", "test", None
        )
        self.assertIn(
            "/resolve/66b5419432fd24735235883b9220582e63aa9339/",
            urls[0],
        )

    def test_stratified_take_is_balanced_and_deterministic(self):
        pools = {
            "a": [{"id": f"a{i}"} for i in range(4)],
            "b": [{"id": f"b{i}"} for i in range(4)],
        }
        first = stratified_take(pools, 6, 7)
        second = stratified_take(pools, 6, 7)
        self.assertEqual(first, second)
        prefixes = [row["id"][0] for row in first]
        self.assertEqual(prefixes.count("a"), 3)
        self.assertEqual(prefixes.count("b"), 3)

    def test_fixed_random_predictions_are_reproducible(self):
        cases = [
            {"id": str(index), "choice_labels": list("ABCD"), "answer": "A"}
            for index in range(20)
        ]
        self.assertEqual(
            fixed_random_predictions(cases, 42), fixed_random_predictions(cases, 42)
        )


if __name__ == "__main__":
    unittest.main()
