"""Shared validation helpers for Agent/tool-use training and evaluation."""

import re


NUMBER_PATTERN = re.compile(r"(?<![\d.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?![\d.])")
NUMERIC_GT_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def validate_gt_in_text(text, gt_list):
    """Return ground-truth values explicitly present in ``text``.

    Numeric ground truths are matched against complete numeric tokens instead
    of substrings: GT ``20`` must not be accepted merely because the response
    contains ``420``. Non-numeric values retain case-insensitive substring
    matching for tool results such as translated text or city names.
    """
    raw_text = str(text)
    numeric_text = raw_text.replace(",", "")
    numbers = [float(value) for value in NUMBER_PATTERN.findall(numeric_text)]
    matched = set()

    for ground_truth in gt_list:
        value = str(ground_truth).strip()
        if not value:
            continue
        normalized = value.replace(",", "")
        if NUMERIC_GT_PATTERN.fullmatch(normalized):
            target = float(normalized)
            tolerance = max(1e-6, abs(target) * 1e-9)
            if any(abs(target - number) <= tolerance for number in numbers):
                matched.add(value)
        elif value.lower() in raw_text.lower():
            matched.add(value)
    return matched
