"""Generate a deterministic Agent-OPD math candidate reservoir.

Generation is programmatic and requires no model/API inference.  The output is
compatible with ``AgentRLDataset`` and intentionally over-samples the observed
Agent sweet spots: distractor routing, stopping after a tool observation,
simple arithmetic/powers, and a small set of mixed-expression probes.  Model
behavior is still verified later by ``select_agent_opd_prompts.py``.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.eval_agent_math import DISTRACTOR_NAMES, TOOLS, safe_calculate  # noqa: E402


CATEGORY_WEIGHTS = {
    "routing_distractor": 0.375,
    "termination": 0.225,
    "simple_or_power": 0.250,
    "positive_complex_probe": 0.150,
}

ROUTING_PROMPTS = [
    "请从提供的工具中选择语义正确的工具计算：{expression}",
    "不要被无关工具干扰，帮我算出 {expression}。",
    "Use the appropriate tool to calculate {expression}.",
    "Which tool can solve this? Compute {expression} and give me the result.",
    "只需要计算数学表达式 {expression}，不要执行其他任务。",
    "我想知道 {expression} 等于多少，请调用正确的函数。",
]

TERMINATION_PROMPTS = [
    "请用工具计算 {expression}；拿到工具结果后直接回答并结束。",
    "计算 {expression}。工具返回结果后不要再调用无关工具。",
    "Use a tool for {expression}. Once the result is returned, answer it and stop.",
    "Find the value of {expression}; do not make another tool call after obtaining it.",
    "帮我算 {expression}，最终只需给出正确结果。",
    "调用合适工具求 {expression}，得到结果后完成任务。",
]

PLAIN_PROMPTS = [
    "请使用合适的工具计算：{expression}",
    "Calculate {expression} with the appropriate tool.",
    "帮我算一下 {expression} 等于多少。",
    "What is {expression}? Please use a tool.",
    "求表达式 {expression} 的值，请调用工具。",
    "Use the math tool and report the value of {expression}.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Agent-OPD candidate prompts")
    parser.add_argument("--num_candidates", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "dataset" / "opd_agent_candidates" / "agent_math_candidates_8k.jsonl"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def allocate_counts(total, weights):
    raw = {name: total * weight for name, weight in weights.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(weights, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def simple_expression(rng, subtype=None):
    subtype = subtype or rng.choices(
        ["power", "add", "subtract", "multiply", "divide"],
        weights=[40, 15, 15, 15, 15],
        k=1,
    )[0]
    if subtype == "power":
        base = rng.randint(2, 40)
        exponent = rng.choice([2, 2, 2, 3])
        syntax = rng.choices(["**", " ** ", "^"], weights=[60, 20, 20], k=1)[0]
        return f"{base}{syntax}{exponent}", subtype, syntax.strip()
    if subtype == "add":
        return f"{rng.randint(10, 9999)}+{rng.randint(10, 9999)}", subtype, None
    if subtype == "subtract":
        return f"({rng.randint(10, 9999)})-{rng.randint(10, 9999)}", subtype, None
    if subtype == "multiply":
        return f"{rng.randint(2, 9999)}*{rng.randint(2, 9999)}", subtype, None
    quotient, divisor = rng.randint(2, 9999), rng.randint(2, 50)
    return f"({quotient * divisor})/{divisor}", subtype, None


def positive_complex_expression(rng, pattern=None):
    pattern = pattern or rng.choice(
        ["power_times_division", "multiply_plus_difference", "division_minus_product_plus"]
    )
    if pattern == "power_times_division":
        base, exponent = rng.randint(2, 50), rng.choice([2, 3])
        quotient, divisor = rng.randint(2, 999), rng.randint(2, 30)
        return f"({base}**{exponent})*(({quotient * divisor})/{divisor})", pattern
    if pattern == "multiply_plus_difference":
        a, b, c, d = (rng.randint(2, 999) for _ in range(4))
        return f"{a}*{b}+({c}-{d})", pattern
    quotient, divisor = rng.randint(2, 999), rng.randint(2, 30)
    a, b, c = (rng.randint(2, 99) for _ in range(3))
    return f"(({quotient * divisor}/{divisor})-({a}*{b}))+{c}", pattern


def expression_for_category(rng, category, ordinal):
    if category == "simple_or_power":
        cycle = ["power"] * 8 + ["add"] * 3 + ["subtract"] * 3 + ["multiply"] * 3 + ["divide"] * 3
        expression, subtype, power_syntax = simple_expression(rng, cycle[ordinal % len(cycle)])
        return expression, subtype, power_syntax
    if category == "positive_complex_probe":
        patterns = ["power_times_division", "multiply_plus_difference", "division_minus_product_plus"]
        expression, pattern = positive_complex_expression(rng, patterns[ordinal % len(patterns)])
        return expression, pattern, None
    # Routing and termination should exercise the same expression families so
    # the measured difference can be attributed to policy behavior as well as
    # arithmetic syntax.
    if rng.random() < 0.70:
        return simple_expression(rng)
    expression, pattern = positive_complex_expression(rng)
    return expression, pattern, None


def ordered_tools(rng, category):
    tool_count = rng.randint(2, len(DISTRACTOR_NAMES) + 1)
    distractors = rng.sample(DISTRACTOR_NAMES, tool_count - 1)
    names = distractors + ["calculate_math"]
    if category == "routing_distractor" and rng.random() < 0.85:
        # The 500-case analysis showed a strong Base first-tool bias. Keep the
        # correct math tool away from position zero in most routing candidates.
        math_position = rng.randint(1, tool_count - 1)
    else:
        math_position = rng.randrange(tool_count)
    names.remove("calculate_math")
    names.insert(math_position, "calculate_math")
    return names, math_position


def prompt_for_category(rng, category, expression):
    if category == "routing_distractor":
        templates = ROUTING_PROMPTS
    elif category == "termination":
        templates = TERMINATION_PROMPTS
    else:
        templates = PLAIN_PROMPTS
    template_index = rng.randrange(len(templates))
    return templates[template_index].format(expression=expression), template_index


def build_row(rng, category, ordinal, candidate_id):
    expression, pattern, power_syntax = expression_for_category(rng, category, ordinal)
    result = safe_calculate(expression)
    if not isinstance(result, (int, float)) or not float(result).is_integer():
        raise ValueError(f"generator produced a non-integer result: {expression} -> {result}")
    result = int(result)
    names, math_position = ordered_tools(rng, category)
    prompt, template_index = prompt_for_category(rng, category, expression)
    tools = [TOOLS[name] for name in names]
    return {
        "conversations": [
            {
                "role": "system",
                "content": "",
                "tools": json.dumps(tools, ensure_ascii=False),
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ],
        "gt": [str(result)],
        "metadata": {
            "candidate_id": f"agent_opd_{candidate_id:05d}",
            "category": category,
            "pattern": pattern,
            "expression": expression,
            "power_syntax": power_syntax,
            "tool_names": names,
            "math_tool_position": math_position,
            "prompt_template_index": template_index,
            "generator": "scripts/generate_agent_opd_candidates.py",
        },
    }


def main():
    args = parse_args()
    if args.num_candidates < 1:
        raise ValueError("--num_candidates must be positive")
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite to replace it: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    counts = allocate_counts(args.num_candidates, CATEGORY_WEIGHTS)
    rows = []
    seen = set()
    candidate_id = 1
    for category, count in counts.items():
        accepted = 0
        attempts = 0
        while accepted < count:
            row = build_row(rng, category, attempts, candidate_id)
            attempts += 1
            fingerprint = (
                row["conversations"][1]["content"],
                tuple(row["metadata"]["tool_names"]),
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            rows.append(row)
            candidate_id += 1
            accepted += 1
    rng.shuffle(rows)

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "output": str(output_path),
        "num_candidates": len(rows),
        "seed": args.seed,
        "category_counts": dict(Counter(row["metadata"]["category"] for row in rows)),
        "pattern_counts": dict(Counter(row["metadata"]["pattern"] for row in rows)),
        "math_tool_first": sum(row["metadata"]["math_tool_position"] == 0 for row in rows),
        "math_tool_not_first": sum(row["metadata"]["math_tool_position"] > 0 for row in rows),
        "requires_model_inference": False,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
