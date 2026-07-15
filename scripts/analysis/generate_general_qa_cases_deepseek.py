"""Generate a larger open-ended general-QA evaluation set with DeepSeek.

The output schema is directly accepted by ``scripts/eval_general_qa_deepseek.py``.
Progress is appended after every successful API batch so interrupted generation
can be resumed without paying for completed batches again.
"""

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import time
from collections import OrderedDict, Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

CATEGORY_SPECS = OrderedDict(
    [
        ("稳定事实", (40, "稳定、常见的事实知识，如人物、事物、语言和基本社会常识")),
        ("基础科学", (40, "中小学层面的生物、物理、化学、天文和环境科学")),
        ("历史文化", (35, "中国与世界的基础历史、文学、艺术和文化常识")),
        ("地理常识", (30, "中国与世界的自然地理、人文地理和典型地理现象")),
        ("概念解释", (30, "用简洁语言解释常见概念、机制、区别或因果关系")),
        ("实用常识", (25, "非高风险的生活、信息、安全、沟通和日常判断常识")),
    ]
)

CATEGORY_SLUGS = {
    "稳定事实": "fact",
    "基础科学": "science",
    "历史文化": "culture",
    "地理常识": "geography",
    "概念解释": "concept",
    "实用常识": "practical",
}

SYSTEM_PROMPT = """你是严谨的通用问答评测集编写者。你需要按指定类别生成相互独立的中文开放式问答题。

硬性规则：
1. 每题必须是非选择题，不得给A/B/C/D选项，不得要求写代码或复杂数学计算。
2. 答案应在2024年以前长期稳定，不涉及新闻、当前任职者、实时价格、政策时效或未来预测。
3. 避免政治立场、医疗诊断、法律建议、金融建议和其他高风险问题。
4. 问题必须有清晰、可核验的核心答案，不能是纯观点题、脑筋急转弯或依赖上下文的问题。
5. 难度以普通中文使用者可理解的基础和中等题为主；问题表达自然，不刻意模仿考试模板。
6. reference_answer控制在1到4句话；reference_points包含2到5个独立得分点；requirements包含1到3项判分约束。
7. 同一批问题不能同义重复。不得在问题中泄漏参考答案。
8. 只输出合法JSON，不要输出Markdown或额外说明。

JSON格式：
{
  "cases": [
    {
      "prompt": "问题",
      "reference_answer": "参考答案",
      "reference_points": ["得分点1", "得分点2"],
      "requirements": ["约束1"]
    }
  ]
}
"""


def parse_json_content(content):
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0].strip()
    return json.loads(content)


def normalized_prompt(text):
    return re.sub(r"[\s，。！？、：；,.!?:;]+", "", str(text)).lower()


def normalize_case(raw, category):
    if not isinstance(raw, dict):
        raise ValueError("case is not an object")
    prompt = str(raw.get("prompt", "")).strip()
    reference_answer = str(raw.get("reference_answer", "")).strip()
    reference_points = raw.get("reference_points")
    requirements = raw.get("requirements")
    if not prompt or not reference_answer:
        raise ValueError("empty prompt or reference_answer")
    if not isinstance(reference_points, list) or not 2 <= len(reference_points) <= 5:
        raise ValueError("reference_points must contain 2-5 items")
    if not isinstance(requirements, list) or not 1 <= len(requirements) <= 3:
        raise ValueError("requirements must contain 1-3 items")
    reference_points = [str(item).strip() for item in reference_points if str(item).strip()]
    requirements = [str(item).strip() for item in requirements if str(item).strip()]
    if len(reference_points) < 2 or not requirements:
        raise ValueError("empty scoring items")
    if re.search(r"(^|\n)\s*[A-DＡ-Ｄ][.、:：)]", prompt):
        raise ValueError("multiple-choice formatting is not allowed")
    if len(prompt) > 240 or len(reference_answer) > 500:
        raise ValueError("prompt or answer is too long")
    return {
        "category": category,
        "prompt": prompt,
        "is_code": False,
        "reference_answer": reference_answer,
        "reference_points": reference_points,
        "requirements": requirements,
    }


def category_targets(total):
    if total < len(CATEGORY_SPECS):
        raise ValueError(f"num_cases must be at least {len(CATEGORY_SPECS)}")
    weight_total = sum(weight for weight, _ in CATEGORY_SPECS.values())
    targets = {
        category: total * weight // weight_total
        for category, (weight, _) in CATEGORY_SPECS.items()
    }
    remainder = total - sum(targets.values())
    for category in CATEGORY_SPECS:
        if remainder <= 0:
            break
        targets[category] += 1
        remainder -= 1
    return targets


def create_client(args):
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"未设置环境变量 {args.api_key_env}")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少openai SDK，请执行 python -m pip install -U openai") from exc
    return OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=0,
    )


def request_batch(
    client, category, description, count, existing_prompts, args, batch_nonce=None
):
    payload = {
        "category": category,
        "category_description": description,
        "requested_count": count,
        "difficulty_mix": "约60%基础题、40%中等题",
        "batch_nonce": batch_nonce,
        "avoid_repeating_these_questions": existing_prompts[-40:],
    }
    request = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "请按以下参数生成题目：\n"
                + json.dumps(payload, ensure_ascii=False),
            },
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": args.max_tokens,
        "extra_body": {
            "thinking": {"type": "enabled" if args.thinking else "disabled"}
        },
    }
    if args.thinking:
        request["extra_body"]["reasoning_effort"] = args.reasoning_effort
    else:
        request["temperature"] = args.temperature

    last_error = None
    for attempt in range(args.retries + 1):
        try:
            response = client.chat.completions.create(**request)
            raw = parse_json_content(response.choices[0].message.content)
            cases = raw.get("cases")
            if not isinstance(cases, list):
                raise ValueError("response has no cases list")
            return cases
        except Exception as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            time.sleep(min(2 ** attempt, 16) + random.random())
    raise RuntimeError(f"DeepSeek生成失败: {last_error}") from last_error


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_final(path, rows, targets):
    ordered = []
    for category in CATEGORY_SPECS:
        category_rows = [row for row in rows if row["category"] == category]
        for index, row in enumerate(category_rows[: targets[category]], 1):
            item = dict(row)
            item["id"] = f"{CATEGORY_SLUGS[category]}_{index:03d}"
            ordered.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in ordered:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return ordered


def parse_args():
    parser = argparse.ArgumentParser(description="使用DeepSeek生成开放式通用问答评测集")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "dataset" / "eval_general_qa_200.jsonl"),
    )
    parser.add_argument("--num_cases", default=200, type=int)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument(
        "--workers",
        default=20,
        type=int,
        help="并发API请求数；200题、batch_size=10时最多约20个并发任务",
    )
    parser.add_argument("--resume", default=1, type=int, choices=[0, 1])
    parser.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--base_url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--timeout", default=180.0, type=float)
    parser.add_argument("--retries", default=3, type=int)
    parser.add_argument("--max_tokens", default=8192, type=int)
    parser.add_argument("--thinking", default=1, type=int, choices=[0, 1])
    parser.add_argument("--reasoning_effort", default="high", choices=["high", "max"])
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--max_batches_per_category", default=20, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.num_cases < 1 or args.workers < 1:
        raise ValueError("num_cases, batch_size and workers must be positive")
    output = Path(args.output).expanduser().resolve()
    progress = output.with_suffix(".progress.jsonl")
    if not args.resume and progress.exists():
        progress.unlink()
    rows = read_jsonl(progress) if args.resume else []
    targets = category_targets(args.num_cases)
    counts = Counter(row.get("category") for row in rows)
    seen = {normalized_prompt(row.get("prompt", "")) for row in rows}
    client = create_client(args)

    batches_used = Counter()
    round_index = 0
    while any(counts[category] < targets[category] for category in CATEGORY_SPECS):
        tasks = []
        for category, (_, description) in CATEGORY_SPECS.items():
            needed = targets[category] - counts[category]
            if needed <= 0:
                continue
            batch_count = math.ceil(needed / args.batch_size)
            remaining_budget = args.max_batches_per_category - batches_used[category]
            if remaining_budget <= 0:
                raise RuntimeError(
                    f"{category}已用完{args.max_batches_per_category}批: "
                    f"{counts[category]}/{targets[category]}"
                )
            batch_count = min(batch_count, remaining_budget)
            existing = [
                row["prompt"] for row in rows if row.get("category") == category
            ]
            for batch_index in range(batch_count):
                # Ask for two surplus rows per batch because malformed items are
                # filtered locally. Surplus rows beyond the target are discarded.
                request_count = min(args.batch_size + 2, needed + 2)
                tasks.append(
                    (
                        category,
                        description,
                        request_count,
                        existing,
                        f"round-{round_index}-batch-{batch_index}",
                    )
                )

        if not tasks:
            break
        print(
            f"[parallel] round={round_index} requests={len(tasks)} "
            f"workers={min(args.workers, len(tasks))}",
            flush=True,
        )
        accepted_this_round = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, len(tasks))
        ) as executor:
            future_map = {
                executor.submit(
                    request_batch,
                    client,
                    category,
                    description,
                    request_count,
                    existing,
                    args,
                    nonce,
                ): (category, nonce)
                for category, description, request_count, existing, nonce in tasks
            }
            for future in concurrent.futures.as_completed(future_map):
                category, nonce = future_map[future]
                batches_used[category] += 1
                try:
                    raw_cases = future.result()
                except Exception as exc:
                    print(f"[{category}] {nonce} failed: {exc}", flush=True)
                    continue
                accepted = []
                for raw in raw_cases:
                    if counts[category] + len(accepted) >= targets[category]:
                        break
                    try:
                        case = normalize_case(raw, category)
                    except ValueError as exc:
                        print(f"[{category}] rejected: {exc}", flush=True)
                        continue
                    key = normalized_prompt(case["prompt"])
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    accepted.append(case)
                if accepted:
                    append_jsonl(progress, accepted)
                    rows.extend(accepted)
                    counts[category] += len(accepted)
                    accepted_this_round += len(accepted)
                print(
                    f"[{category}] {counts[category]}/{targets[category]} "
                    f"| accepted={len(accepted)} | {nonce}",
                    flush=True,
                )
        if accepted_this_round == 0:
            raise RuntimeError("并发一轮未接受任何新题，请检查API返回或降低batch_size")
        round_index += 1

    final_rows = write_final(output, rows, targets)
    manifest = {
        "output": str(output),
        "progress": str(progress),
        "model": args.model,
        "total": len(final_rows),
        "categories": dict(Counter(row["category"] for row in final_rows)),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
