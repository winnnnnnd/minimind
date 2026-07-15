"""Create deterministic stratified train/validation/pilot Agent-OPD splits."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split verified Agent-OPD sweet prompts")
    parser.add_argument(
        "--input",
        default="dataset/opd_agent_sweet_selection/teacher_pass_student_fail.jsonl",
    )
    parser.add_argument(
        "--output_dir", default="dataset/opd_agent_sweet_selection/splits"
    )
    parser.add_argument("--train_count", type=int, default=1000)
    parser.add_argument("--validation_count", type=int, default=200)
    parser.add_argument("--pilot_count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stratum(row):
    metadata = row.get("metadata", {})
    position = "first" if metadata.get("math_tool_position") == 0 else "not_first"
    return (
        metadata.get("category", "unknown"),
        metadata.get("pattern", "unknown"),
        position,
    )


def allocate_counts(groups, total):
    available = {key: len(rows) for key, rows in groups.items()}
    population = sum(available.values())
    if not 0 <= total <= population:
        raise ValueError(f"cannot allocate {total} rows from a population of {population}")
    if total == 0:
        return {key: 0 for key in groups}

    raw = {key: total * size / population for key, size in available.items()}
    allocation = {key: min(size, int(raw[key])) for key, size in available.items()}
    remaining = total - sum(allocation.values())
    while remaining:
        candidates = [key for key in groups if allocation[key] < available[key]]
        if not candidates:
            raise RuntimeError("stratified allocation exhausted before reaching requested size")
        candidates.sort(
            key=lambda key: (raw[key] - allocation[key], available[key], str(key)),
            reverse=True,
        )
        for key in candidates:
            if remaining == 0:
                break
            allocation[key] += 1
            remaining -= 1
    return allocation


def stratified_take(rows, count, rng):
    groups = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)
    allocation = allocate_counts(groups, count)
    selected, remaining = [], []
    for key, group_rows in groups.items():
        take = allocation[key]
        selected.extend(group_rows[:take])
        remaining.extend(group_rows[take:])
    rng.shuffle(selected)
    rng.shuffle(remaining)
    return selected, remaining


def summarize(rows):
    categories = Counter()
    patterns = Counter()
    positions = Counter()
    for row in rows:
        metadata = row.get("metadata", {})
        categories[metadata.get("category", "unknown")] += 1
        patterns[metadata.get("pattern", "unknown")] += 1
        positions["first" if metadata.get("math_tool_position") == 0 else "not_first"] += 1
    return {
        "total": len(rows),
        "category": dict(categories),
        "pattern": dict(patterns),
        "math_tool_position": dict(positions),
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if min(args.train_count, args.validation_count, args.pilot_count) < 0:
        raise ValueError("split counts must be non-negative")
    if args.pilot_count > args.train_count:
        raise ValueError("--pilot_count cannot exceed --train_count")

    input_path = Path(args.input).expanduser().resolve()
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = args.train_count + args.validation_count
    if len(rows) < required:
        raise ValueError(f"need at least {required} verified rows, found {len(rows)}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": output_dir / f"opd_train_{args.train_count}.jsonl",
        "validation": output_dir / f"opd_validation_{args.validation_count}.jsonl",
        "pilot": output_dir / f"opd_pilot_{args.pilot_count}.jsonl",
        "report": output_dir / "split_report.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "split output already exists; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )

    rng = random.Random(args.seed)
    validation_rows, non_validation_rows = stratified_take(rows, args.validation_count, rng)
    train_rows, unused_rows = stratified_take(non_validation_rows, args.train_count, rng)
    pilot_rows, _ = stratified_take(train_rows, args.pilot_count, rng)

    write_jsonl(output_paths["train"], train_rows)
    write_jsonl(output_paths["validation"], validation_rows)
    write_jsonl(output_paths["pilot"], pilot_rows)
    report = {
        "input": str(input_path),
        "seed": args.seed,
        "input_total": len(rows),
        "train": summarize(train_rows),
        "validation": summarize(validation_rows),
        "pilot": summarize(pilot_rows),
        "unused": summarize(unused_rows),
        "pilot_is_subset_of_train": True,
        "validation_is_disjoint": True,
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }
    output_paths["report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
