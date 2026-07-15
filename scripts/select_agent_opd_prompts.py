"""Measure and select Agent-strong/Base-weak prompts for Agent OPD.

The default cascade first evaluates the Base student, then runs the Agent only
on Base failures and stops when enough verified sweet-spot rows are collected.
Thus an 8k candidate reservoir is not automatically an 8k x 2 inference job.
Models are always loaded sequentially to keep peak memory low.
"""

import argparse
import gc
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.eval_agent_math import load_model_and_tokenizer, resolve_device  # noqa: E402
from trainer.tool_utils import validate_gt_in_text  # noqa: E402
from trainer.train_agent import execute_tool, parse_tool_calls  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Select teacher-aligned prompts for Agent OPD")
    parser.add_argument("--data_path", default=str(REPO_ROOT / "dataset" / "agent_rl_math.jsonl"))
    parser.add_argument("--student_model", default=str(REPO_ROOT / "minimind-3"))
    parser.add_argument("--teacher_model", default=str(REPO_ROOT / "model_files" / "agent_768.pth"))
    parser.add_argument("--native_tokenizer", default=str(REPO_ROOT / "model"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "dataset" / "opd_math_selection"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=2000, help="0 means all remaining rows")
    parser.add_argument("--gt_count", type=int, default=1,
                        help="Only keep rows with exactly this many GT values; 0 disables filtering")
    parser.add_argument("--both_pass_ratio", type=float, default=0.10,
                        help="Keep at most this many both-pass rows per teacher-only row")
    parser.add_argument(
        "--screen_mode", choices=["cascade", "full"], default="cascade",
        help="cascade=Base全测后Agent仅测Base失败项；full=两模型测全部",
    )
    parser.add_argument(
        "--target_teacher_only", type=int, default=1000,
        help="cascade模式收集到多少条Agent对/Base错后停止Agent；0表示测完候选子集",
    )
    parser.add_argument("--required_tool", choices=["calculate_math", "any", "none"],
                        default="calculate_math")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0/mps")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--do_sample", default=0, type=int, choices=[0, 1])
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--progress_interval", default=20, type=int)
    parser.add_argument("--reuse_cache", action="store_true",
                        help="Reuse or resume matching per-model result files in output_dir")
    return parser.parse_args()


def read_samples(path, offset, limit, gt_count):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index < offset:
                continue
            if limit > 0 and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            conversations = row.get("conversations")
            if not isinstance(conversations, list) or len(conversations) < 2:
                raise ValueError(f"row {index} has no usable conversations")
            tools = None
            messages = []
            for message in conversations[:-1]:
                message = dict(message)
                if message.get("role") == "system" and message.get("tools"):
                    tools = message["tools"]
                    if isinstance(tools, str):
                        tools = json.loads(tools)
                messages.append(message)
            gt = row.get("gt", [])
            if not isinstance(gt, list):
                gt = [gt]
            if gt_count > 0 and len(gt) != gt_count:
                continue
            rows.append({
                "source_index": index,
                "row": row,
                "messages": messages,
                "tools": tools,
                "gt": gt,
            })
    if not rows:
        raise ValueError("No rows selected; check --data_path/--offset/--limit")
    return rows


def parse_arguments(call):
    raw = call.get("arguments", {}) if isinstance(call, dict) else {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    return raw if isinstance(raw, dict) else {}


@torch.inference_mode()
def generate_turn(model, tokenizer, messages, tools, args, device):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
        open_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(args.do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_args.update(temperature=args.temperature, top_p=args.top_p)
    generated = model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"], **generation_args
    )
    completion_ids = generated[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def run_sample(model, tokenizer, sample, args, device):
    messages = [dict(message) for message in sample["messages"]]
    tools = sample["tools"]
    final_output = ""
    called_tools = []
    unfinished = False

    if not tools and args.required_tool != "none":
        return {
            "passed": False,
            "reason": "no_tools_in_prompt",
            "called_tools": [],
            "verified_gt": [],
            "final_output": "",
        }

    for turn in range(args.max_turns):
        final_output = generate_turn(model, tokenizer, messages, tools, args, device)
        calls = parse_tool_calls(final_output)
        if not calls:
            break
        called_tools.extend(call.get("name", "") for call in calls)
        if turn == args.max_turns - 1:
            unfinished = True
            break
        messages.append({"role": "assistant", "content": final_output})
        for call in calls:
            name = call.get("name", "")
            result = execute_tool(name, parse_arguments(call))
            if result is None:
                result = {"error": f"tool execution failed: {name}"}
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)[:2048]})

    verified = validate_gt_in_text(final_output, sample["gt"]) if sample["gt"] else set()
    gt_passed = bool(sample["gt"]) and len(verified) == len(sample["gt"])
    if args.required_tool == "calculate_math":
        tool_passed = "calculate_math" in called_tools
    elif args.required_tool == "any":
        tool_passed = bool(called_tools)
    else:
        tool_passed = True
    passed = gt_passed and tool_passed and not unfinished
    if unfinished:
        reason = "unfinished"
    elif not tool_passed:
        reason = "required_tool_missing"
    elif not gt_passed:
        reason = "ground_truth_missing"
    else:
        reason = "passed"
    return {
        "passed": passed,
        "reason": reason,
        "called_tools": called_tools,
        "verified_gt": sorted(str(value) for value in verified),
        "final_output": final_output,
    }


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unload_model(model, device):
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def cache_metadata(source, args):
    source_path = Path(source).expanduser()
    resolved_source = str(source_path.resolve()) if source_path.exists() else str(source)
    data_path = Path(args.data_path).expanduser().resolve()
    data_stat = data_path.stat()
    return {
        "source": resolved_source,
        "data_path": str(data_path),
        "data_size": data_stat.st_size,
        "data_mtime_ns": data_stat.st_mtime_ns,
        "required_tool": args.required_tool,
        "max_new_tokens": args.max_new_tokens,
        "max_turns": args.max_turns,
        "do_sample": bool(args.do_sample),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "gt_count": args.gt_count,
        "screen_mode": args.screen_mode,
    }


def load_cached_results(path, metadata_path, expected_metadata, samples, allow_prefix=False):
    if not path.exists() or not metadata_path.exists():
        return None
    actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if actual_metadata != expected_metadata:
        raise ValueError(
            f"Cache metadata {metadata_path} does not match this evaluation; "
            "rerun without --reuse_cache"
        )
    results = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_indices = [sample["source_index"] for sample in samples]
    actual_indices = [result.get("source_index") for result in results]
    valid_indices = (
        actual_indices == expected_indices[:len(actual_indices)]
        if allow_prefix
        else actual_indices == expected_indices
    )
    if len(actual_indices) > len(expected_indices) or not valid_indices:
        raise ValueError(f"Cache {path} does not match the currently selected source rows")
    return results


def evaluate_source(source, label, samples, args, device, output_dir, stop_after_passes=0):
    cache_path = output_dir / f"{label}_results.jsonl"
    metadata_path = output_dir / f"{label}_results.meta.json"
    metadata = cache_metadata(source, args)
    results = []
    if args.reuse_cache:
        cached = load_cached_results(
            cache_path,
            metadata_path,
            metadata,
            samples,
            allow_prefix=True,
        )
        if cached is not None:
            results = cached
            complete = len(results) == len(samples)
            target_reached = (
                stop_after_passes > 0
                and sum(item["passed"] for item in results) >= stop_after_passes
            )
            if complete or target_reached:
                print(f"[{label}] reused {len(results)} cached results from {cache_path}")
                return results
            print(f"[{label}] resuming after {len(results)} cached results from {cache_path}")

    model, tokenizer, source_type = load_model_and_tokenizer(source, args, device)
    print(f"[{label}] loaded {source_type}: {source}")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    passed_count = sum(item["passed"] for item in results)
    try:
        cache_mode = "a" if results else "w"
        with cache_path.open(cache_mode, encoding="utf-8") as cache_file:
            for position, sample in enumerate(samples[len(results):], start=len(results) + 1):
                seed_everything(args.seed + sample["source_index"])
                result = run_sample(model, tokenizer, sample, args, device)
                result = {"source_index": sample["source_index"], **result}
                results.append(result)
                passed_count += int(result["passed"])
                cache_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                cache_file.flush()
                if position % args.progress_interval == 0 or position == len(samples):
                    print(
                        f"[{label}] {position}/{len(samples)} | pass={passed_count}/{position} "
                        f"({passed_count / position:.2%})"
                    )
                if (
                    stop_after_passes > 0
                    and passed_count >= stop_after_passes
                ):
                    print(
                        f"[{label}] early stop: collected {stop_after_passes} verified "
                        f"Agent-pass/Base-fail rows after {position} Agent evaluations"
                    )
                    break
    finally:
        unload_model(model, device)
        del tokenizer
    return results


def write_rows(path, samples):
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample["row"], ensure_ascii=False) + "\n")


def summarize_rows(samples):
    metadata = [sample["row"].get("metadata", {}) for sample in samples]
    return {
        "category": dict(Counter(item.get("category", "unknown") for item in metadata)),
        "pattern": dict(Counter(item.get("pattern", "unknown") for item in metadata)),
        "math_tool_position": dict(
            Counter(
                "first" if item.get("math_tool_position") == 0 else "not_first"
                for item in metadata
            )
        ),
    }


def main():
    args = parse_args()
    if args.offset < 0 or args.limit < 0 or args.gt_count < 0:
        raise ValueError("--offset, --limit and --gt_count must be non-negative")
    if args.max_turns < 1 or args.max_new_tokens < 1:
        raise ValueError("--max_turns and --max_new_tokens must be positive")
    if args.progress_interval < 1:
        raise ValueError("--progress_interval must be positive")
    if args.both_pass_ratio < 0:
        raise ValueError("--both_pass_ratio must be non-negative")
    if args.target_teacher_only < 0:
        raise ValueError("--target_teacher_only must be non-negative")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_samples(args.data_path, args.offset, args.limit, args.gt_count)
    device = resolve_device(args.device)

    student_results = evaluate_source(
        args.student_model, "student", samples, args, device, output_dir
    )
    if args.screen_mode == "cascade":
        teacher_candidates = [
            sample
            for sample, student in zip(samples, student_results)
            if not student["passed"]
        ]
        teacher_results = (
            evaluate_source(
                args.teacher_model,
                "teacher",
                teacher_candidates,
                args,
                device,
                output_dir,
                stop_after_passes=args.target_teacher_only,
            )
            if teacher_candidates
            else []
        )
        buckets = {
            "teacher_pass_student_fail": [],
            "both_pass": [],
            "student_pass_teacher_fail": [],
            "both_fail": [],
        }
        for sample, teacher in zip(teacher_candidates, teacher_results):
            bucket = "teacher_pass_student_fail" if teacher["passed"] else "both_fail"
            buckets[bucket].append(sample)
        not_evaluated = {
            "student_pass_not_teacher_evaluated": [
                sample
                for sample, student in zip(samples, student_results)
                if student["passed"]
            ],
            "student_fail_not_teacher_evaluated": teacher_candidates[len(teacher_results):],
        }
    else:
        teacher_results = evaluate_source(
            args.teacher_model, "teacher", samples, args, device, output_dir
        )
        buckets = {
            "teacher_pass_student_fail": [],
            "both_pass": [],
            "student_pass_teacher_fail": [],
            "both_fail": [],
        }
        for sample, student, teacher in zip(samples, student_results, teacher_results):
            if teacher["passed"] and not student["passed"]:
                bucket = "teacher_pass_student_fail"
            elif teacher["passed"] and student["passed"]:
                bucket = "both_pass"
            elif student["passed"] and not teacher["passed"]:
                bucket = "student_pass_teacher_fail"
            else:
                bucket = "both_fail"
            buckets[bucket].append(sample)
        not_evaluated = {}

    for name, rows in buckets.items():
        write_rows(output_dir / f"{name}.jsonl", rows)
    for name, rows in not_evaluated.items():
        write_rows(output_dir / f"{name}.jsonl", rows)

    hard_rows = list(buckets["teacher_pass_student_fail"])
    easy_limit = min(len(buckets["both_pass"]), round(len(hard_rows) * args.both_pass_ratio))
    rng = random.Random(args.seed)
    easy_rows = rng.sample(buckets["both_pass"], easy_limit) if easy_limit else []
    train_rows = hard_rows + easy_rows
    rng.shuffle(train_rows)
    write_rows(output_dir / "opd_train.jsonl", train_rows)

    report = {
        "data_path": str(Path(args.data_path).expanduser().resolve()),
        "offset": args.offset,
        "limit": args.limit,
        "gt_count": args.gt_count,
        "evaluated": len(samples),
        "required_tool": args.required_tool,
        "screen_mode": args.screen_mode,
        "target_teacher_only": args.target_teacher_only,
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "inference_counts": {
            "student": len(student_results),
            "teacher": len(teacher_results),
            "total": len(student_results) + len(teacher_results),
            "full_two_model_cost": len(samples) * 2,
        },
        "counts": {name: len(rows) for name, rows in buckets.items()},
        "not_evaluated_counts": {name: len(rows) for name, rows in not_evaluated.items()},
        "sweet_spot_distribution": summarize_rows(hard_rows),
        "opd_train": {
            "total": len(train_rows),
            "teacher_pass_student_fail": len(hard_rows),
            "sampled_both_pass": len(easy_rows),
            "both_pass_ratio": args.both_pass_ratio,
        },
    }
    (output_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not train_rows:
        print("WARNING: opd_train.jsonl is empty; do not start training before inspecting model outputs")


if __name__ == "__main__":
    main()
