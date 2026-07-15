"""Build a small, deterministic knowledge benchmark from public MCQ datasets.

The default mix intentionally uses easier/basic-education subsets instead of the
full C-Eval and CMMLU suites:

* 100 C-Eval validation questions
* 100 CMMLU test questions
* 100 ARC-Easy validation questions

The output is a single normalized JSONL consumed by
``scripts/eval_knowledge_benchmark.py``.
"""

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


CEVAL_SUBJECTS = [
    "middle_school_biology",
    "middle_school_physics",
    "middle_school_chemistry",
    "middle_school_geography",
    "middle_school_history",
    "middle_school_politics",
    "high_school_biology",
    "high_school_geography",
    "high_school_history",
    "high_school_chinese",
]

CMMLU_SUBJECTS = [
    "elementary_chinese",
    "elementary_commonsense",
    "elementary_information_and_technology",
    "global_facts",
    "chinese_history",
    "chinese_food_culture",
    "high_school_biology",
    "high_school_geography",
    "world_history",
]

# The CMMLU parquet conversion is published in this immutable Hugging Face
# snapshot rather than at the repository's current ``main`` revision.
CMMLU_PARQUET_REVISION = "66b5419432fd24735235883b9220582e63aa9339"


def row_value(row, *names):
    for name in names:
        if name in row:
            return row[name]
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise KeyError(f"missing all fields {names!r}; available={sorted(row)}")


def format_prompt(question, choices, language):
    option_lines = "\n".join(
        f"{choice['label']}. {choice['text']}" for choice in choices
    )
    if language == "en":
        instruction = (
            "Answer this single-choice question. Briefly explain your reasoning, "
            "then end with a separate line in the exact form 'Answer: X', where X "
            "is the option letter."
        )
    else:
        instruction = (
            "回答下面的单项选择题。请先用一句话简要说明理由，最后单独一行严格写成"
            "“答案：X”，其中X是选项字母。"
        )
    return f"{instruction}\n\n{question}\n{option_lines}"


def normalize_four_choice(dataset, subject, split, row, row_index=None):
    question = str(row_value(row, "question", "Question")).strip()
    choices = [
        {"label": label, "text": str(row_value(row, label)).strip()}
        for label in "ABCD"
    ]
    answer = str(row_value(row, "answer", "Answer")).strip().upper()
    if answer not in {choice["label"] for choice in choices}:
        raise ValueError(f"invalid answer {answer!r}")
    source_id = row.get("id", row.get("ID"))
    if source_id in (None, ""):
        source_id = row_index if row_index is not None else "unknown"
    return {
        "id": f"{dataset}:{subject}:{source_id}",
        "dataset": dataset,
        "subject": subject,
        "split": split,
        "language": "zh",
        "question": question,
        "choices": choices,
        "answer": answer,
        "prompt": format_prompt(question, choices, "zh"),
    }


def normalize_arc(row, split):
    question = str(row_value(row, "question")).strip()
    raw_choices = row_value(row, "choices")
    texts = list(raw_choices["text"])
    raw_labels = [str(value).strip() for value in raw_choices["label"]]
    raw_answer = str(row_value(row, "answerKey")).strip()
    if raw_answer not in raw_labels:
        raise ValueError(f"ARC answer {raw_answer!r} is absent from labels {raw_labels!r}")
    if not 2 <= len(texts) <= 26:
        raise ValueError(f"unsupported ARC option count: {len(texts)}")
    canonical_labels = [chr(ord("A") + index) for index in range(len(texts))]
    answer = canonical_labels[raw_labels.index(raw_answer)]
    choices = [
        {"label": label, "text": str(text).strip()}
        for label, text in zip(canonical_labels, texts)
    ]
    source_id = row.get("id", "unknown")
    return {
        "id": f"arc_easy:science:{source_id}",
        "dataset": "arc_easy",
        "subject": "grade_school_science",
        "split": split,
        "language": "en",
        "question": question,
        "choices": choices,
        "answer": answer,
        "prompt": format_prompt(question, choices, "en"),
    }


def load_subject_rows(dataset_name, subjects, split, dataset_label, cache_dir=None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("缺少 datasets，请先执行 python -m pip install -r requirements.txt") from exc

    rows_by_subject = OrderedDict()
    failures = []
    for subject in subjects:
        dataset = None
        direct_errors = []
        # Read the repository's published parquet directly. This avoids Hub
        # metadata lookup and never executes dataset repository Python code.
        for url in subject_parquet_urls(dataset_name, subject, split, None):
            try:
                print(f"[{dataset_label}] downloading {url}", flush=True)
                dataset = load_dataset(
                    "parquet",
                    data_files={split: url},
                    split=split,
                    cache_dir=cache_dir,
                )
                break
            except Exception as exc:
                direct_errors.append(f"{url}: {type(exc).__name__}: {exc}")

        hub_error = None
        if dataset is None:
            try:
                print(
                    f"[{dataset_label}] {subject}: direct parquet failed; trying Hub",
                    flush=True,
                )
                dataset = load_dataset(
                    dataset_name,
                    subject,
                    split=split,
                    cache_dir=cache_dir,
                    trust_remote_code=False,
                )
            except Exception as exc:
                hub_error = exc
        if dataset is None:
            failures.append(
                f"{dataset_label}/{subject}: direct={' | '.join(direct_errors)}; "
                f"Hub={type(hub_error).__name__}: {hub_error}"
            )
            print(
                f"[{dataset_label}] {subject}: LOAD FAILED through direct URL and Hub",
                flush=True,
            )
            continue

        normalized = []
        for row_index, row in enumerate(dataset):
            try:
                normalized.append(
                    normalize_four_choice(
                        dataset_label, subject, split, row, row_index=row_index
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(f"{dataset_label}/{subject}: skipped row: {exc}")
        rows_by_subject[subject] = normalized
        print(f"[{dataset_label}] {subject}: {len(normalized)} rows", flush=True)
    return rows_by_subject, failures


def stratified_take(rows_by_subject, count, seed):
    if count <= 0:
        return []
    rng = random.Random(seed)
    pools = OrderedDict()
    for subject, rows in rows_by_subject.items():
        rows = list(rows)
        rng.shuffle(rows)
        pools[subject] = rows
    available = sum(len(rows) for rows in pools.values())
    if available < count:
        raise RuntimeError(f"requested {count} rows but only {available} are available")

    selected = []
    cursors = {subject: 0 for subject in pools}
    while len(selected) < count:
        progressed = False
        for subject, rows in pools.items():
            cursor = cursors[subject]
            if cursor < len(rows):
                selected.append(rows[cursor])
                cursors[subject] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def parquet_urls(dataset_name, subset, split, endpoint=None, revision="main"):
    filename = f"{split}-00000-of-00001.parquet"
    relative = f"datasets/{dataset_name}/resolve/{revision}/{subset}/{filename}"
    endpoints = []
    if endpoint:
        endpoints.append(endpoint.rstrip("/"))
    endpoints.append("https://huggingface.co")
    return list(dict.fromkeys(f"{base}/{relative}" for base in endpoints))


def subject_parquet_urls(dataset_name, subject, split, endpoint=None):
    revision = (
        CMMLU_PARQUET_REVISION if dataset_name == "lmlmcat/cmmlu" else "main"
    )
    return parquet_urls(dataset_name, subject, split, endpoint, revision)


def arc_parquet_urls(split, endpoint=None):
    return parquet_urls("allenai/ai2_arc", "ARC-Easy", split, endpoint)


def require_available(rows_by_subject, requested, dataset_label, failures):
    available = sum(len(rows) for rows in rows_by_subject.values())
    if available >= requested:
        return
    relevant = [item for item in failures if item.startswith(f"{dataset_label}/")]
    details = "\n".join(f"- {item}" for item in relevant[:3])
    raise RuntimeError(
        f"{dataset_label} requested {requested} rows but only {available} loaded.\n"
        f"First load errors:\n{details or '- no detailed loader error'}"
    )


def load_arc_rows(split, cache_dir=None, data_file=None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("缺少 datasets，请先执行 python -m pip install -r requirements.txt") from exc
    if data_file:
        source = str(Path(data_file).expanduser().resolve()) if Path(data_file).expanduser().exists() else data_file
        print(f"[arc_easy] loading parquet: {source}", flush=True)
        dataset = load_dataset(
            "parquet", data_files={split: source}, split=split, cache_dir=cache_dir
        )
    else:
        direct_errors = []
        dataset = None
        for url in arc_parquet_urls(split, None):
            try:
                print(f"[arc_easy] downloading {url}", flush=True)
                dataset = load_dataset(
                    "parquet",
                    data_files={split: url},
                    split=split,
                    cache_dir=cache_dir,
                )
                break
            except Exception as exc:
                direct_errors.append(f"{url}: {type(exc).__name__}: {exc}")
        hub_error = None
        if dataset is None:
            try:
                print("[arc_easy] direct parquet failed; trying Hub", flush=True)
                dataset = load_dataset(
                    "allenai/ai2_arc",
                    "ARC-Easy",
                    split=split,
                    cache_dir=cache_dir,
                    trust_remote_code=False,
                )
            except Exception as exc:
                hub_error = exc
        if dataset is None:
            details = "\n".join(direct_errors)
            raise RuntimeError(
                "ARC-Easy无法通过Parquet直链或Hub加载。可先手动下载validation "
                "Parquet，再通过--arc_data_file传入。\n"
                f"{details}\nHub={type(hub_error).__name__}: {hub_error}"
            ) from hub_error
    rows = []
    failures = []
    for row in dataset:
        try:
            rows.append(normalize_arc(row, split))
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"arc_easy: skipped row: {exc}")
    print(f"[arc_easy] grade_school_science: {len(rows)} rows", flush=True)
    return rows, failures


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="准备300题基础知识选择题评测集")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "dataset" / "eval_knowledge_basic_300.jsonl"),
    )
    parser.add_argument("--ceval_count", default=100, type=int)
    parser.add_argument("--cmmlu_count", default=100, type=int)
    parser.add_argument("--arc_count", default=100, type=int)
    parser.add_argument("--ceval_split", default="val")
    parser.add_argument("--cmmlu_split", default="test")
    parser.add_argument("--arc_split", default="validation")
    parser.add_argument(
        "--arc_data_file",
        default=None,
        help="可选的ARC-Easy本地Parquet路径或URL；用于Hub不可访问时",
    )
    parser.add_argument("--seed", default=20260715, type=int)
    parser.add_argument("--cache_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.ceval_count, args.cmmlu_count, args.arc_count) < 0:
        raise ValueError("各数据集题数不能为负数")

    failures = []
    ceval_by_subject = OrderedDict()
    if args.ceval_count:
        ceval_by_subject, errors = load_subject_rows(
            "ceval/ceval-exam",
            CEVAL_SUBJECTS,
            args.ceval_split,
            "ceval",
            args.cache_dir,
        )
        failures.extend(errors)
        require_available(
            ceval_by_subject, args.ceval_count, "ceval", failures
        )
    cmmlu_by_subject = OrderedDict()
    if args.cmmlu_count:
        cmmlu_by_subject, errors = load_subject_rows(
            "lmlmcat/cmmlu",
            CMMLU_SUBJECTS,
            args.cmmlu_split,
            "cmmlu",
            args.cache_dir,
        )
        failures.extend(errors)
        require_available(
            cmmlu_by_subject, args.cmmlu_count, "cmmlu", failures
        )
    arc_rows = []
    if args.arc_count:
        arc_rows, errors = load_arc_rows(
            args.arc_split, args.cache_dir, args.arc_data_file
        )
        failures.extend(errors)

    rows = []
    rows.extend(stratified_take(ceval_by_subject, args.ceval_count, args.seed + 1))
    rows.extend(stratified_take(cmmlu_by_subject, args.cmmlu_count, args.seed + 2))
    rng = random.Random(args.seed + 3)
    rng.shuffle(arc_rows)
    if len(arc_rows) < args.arc_count:
        raise RuntimeError(
            f"requested {args.arc_count} ARC rows but only {len(arc_rows)} are available"
        )
    rows.extend(arc_rows[: args.arc_count])

    output = Path(args.output).expanduser().resolve()
    write_jsonl(output, rows)
    manifest = {
        "output": str(output),
        "seed": args.seed,
        "counts": {
            dataset: sum(row["dataset"] == dataset for row in rows)
            for dataset in ("ceval", "cmmlu", "arc_easy")
        },
        "subjects": {
            dataset: sorted(
                {row["subject"] for row in rows if row["dataset"] == dataset}
            )
            for dataset in ("ceval", "cmmlu", "arc_easy")
        },
        "failures": failures,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
