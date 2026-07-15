"""Compare two or more MiniMind weights on math ToolUse tasks.

By default the evaluator compares the local Transformers-format ``minimind-3``
directory with ``model_files/agent_768.pth``. Models are loaded sequentially to
keep peak memory low.
"""

import argparse
import ast
import gc
import json
import operator
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


TOOLS = {
    "calculate_math": {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "计算数学表达式的结果，支持加减乘除和幂运算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如123+456、2**10",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    "unit_converter": {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "进行单位换算",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    "get_current_weather": {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
    "get_current_time": {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}},
                "required": [],
            },
        },
    },
    "get_exchange_rate": {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "查询两种货币之间的汇率",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["from_currency", "to_currency"],
            },
        },
    },
    "translate_text": {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "将文本翻译成目标语言",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_language": {"type": "string"},
                },
                "required": ["text", "target_language"],
            },
        },
    },
}

DISTRACTOR_NAMES = [
    "unit_converter",
    "get_current_weather",
    "get_current_time",
    "get_exchange_rate",
    "translate_text",
]

LEGACY_EXPRESSIONS = [
    "(94)-35",
    "3**2",
    "(29)+64",
    "(20**3)*((198)/11)",
    "10**2",
    "(4**3)+(20**2)",
    "(12)*48+(47-45)",
    "59*48",
    "3**2",
    "14**3",
    "(72)*(91)",
    "180/(12)",
    "14-(19)+(289/17)",
    "5**3",
    "(2**3)-64*(13)",
    "17**2",
    "11**2",
    "72+10",
    "(84)-60",
    "(348/(12))-(28)*(8)",
]

PROMPT_TEMPLATES = [
    "请使用合适的工具计算：{expression}",
    "Calculate {expression} with the appropriate tool.",
    "帮我算一下{expression}等于多少",
    "What is {expression}? Please use a tool.",
]

BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def normalize_number(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def safe_calculate(expression):
    """Evaluate a short arithmetic expression without Python ``eval``."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 256:
        raise ValueError("表达式为空或过长")
    expression = (
        expression.replace("^", "**")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("（", "(")
        .replace("）", ")")
    )
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("表达式过于复杂")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
            return UNARY_OPS[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("指数过大")
            result = BIN_OPS[type(node.op)](left, right)
            if abs(result) > 1e18:
                raise ValueError("计算结果过大")
            return result
        raise ValueError(f"不支持的表达式节点: {type(node).__name__}")

    return normalize_number(visit(tree))


def generate_expression(rng, difficulty):
    """Generate an integer-result expression with deterministic randomness."""
    if difficulty == "mixed":
        difficulty = rng.choices(["easy", "medium", "hard"], weights=[35, 40, 25], k=1)[0]

    if difficulty == "easy":
        pattern = rng.randrange(5)
        if pattern == 0:
            return f"{rng.randint(1, 9999)}+{rng.randint(1, 9999)}"
        if pattern == 1:
            return f"{rng.randint(1, 9999)}-{rng.randint(1, 9999)}"
        if pattern == 2:
            return f"{rng.randint(2, 9999)}*{rng.randint(2, 9999)}"
        if pattern == 3:
            return f"{rng.randint(2, 20)}**{rng.randint(2, 3)}"
        quotient, divisor = rng.randint(1, 9999), rng.randint(2, 50)
        return f"{quotient * divisor}/{divisor}"

    if difficulty == "medium":
        pattern = rng.randrange(6)
        a, b, c, d = (rng.randint(1, 999) for _ in range(4))
        if pattern == 0:
            return f"({a}+{b})*{rng.randint(2, 99)}"
        if pattern == 1:
            return f"({a}-{b})+{c}*{rng.randint(2, 99)}"
        if pattern == 2:
            return f"({rng.randint(2, 30)}**2)+({rng.randint(2, 30)}**2)"
        if pattern == 3:
            return f"{a}*{b}+({c}-{d})"
        if pattern == 4:
            high, low = max(c, d), min(c, d)
            return f"({a}+{b})*({high}-{low})"
        return f"({a}*{b})-({rng.randint(2, 99)}**2)"

    pattern = rng.randrange(6)
    a, b, c, d, e = (rng.randint(2, 99) for _ in range(5))
    quotient, divisor = rng.randint(2, 999), rng.randint(2, 30)
    numerator = quotient * divisor
    if pattern == 0:
        return f"({a}**{rng.randint(2, 3)})*(({numerator})/{divisor})"
    if pattern == 1:
        return f"({a}**2)+({b}**2)-({c}*{d})"
    if pattern == 2:
        return f"({a}*{b}+({c}-{d}))*{e}"
    if pattern == 3:
        return f"(({numerator}/{divisor})-({a}*{b}))+{c}"
    if pattern == 4:
        high, low = max(c, d), min(c, d)
        return f"({a}+{b})*({high}-{low})+({e}**2)"
    return f"(({a}+{b})*{c})-(({numerator})/{divisor})"


def read_expressions(path):
    """Read raw expressions or JSONL objects containing an ``expression`` field."""
    expressions = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                try:
                    expression = json.loads(line)["expression"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_number} 不是合法的 expression JSONL") from exc
            else:
                expression = line
            safe_calculate(expression)
            expressions.append(expression)
    if not expressions:
        raise ValueError(f"题目文件为空: {path}")
    return expressions


def build_expression_set(num_cases, case_seed, difficulty, cases_file=None):
    if cases_file:
        return read_expressions(cases_file)[:num_cases]

    expressions = list(LEGACY_EXPRESSIONS[:num_cases])
    if len(expressions) >= num_cases:
        return expressions

    rng = random.Random(case_seed)
    seen = set(expressions)
    while len(expressions) < num_cases:
        expression = generate_expression(rng, difficulty)
        if expression in seen:
            continue
        result = safe_calculate(expression)
        if not isinstance(result, (int, float)) or not float(result).is_integer():
            continue
        seen.add(expression)
        expressions.append(expression)
    return expressions


def parse_tool_calls(text):
    calls = []
    for match in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            call = json.loads(match.strip())
            if isinstance(call.get("function"), dict):
                call = call["function"]
            calls.append(call)
        except (AttributeError, json.JSONDecodeError):
            continue
    return calls


def parse_arguments(call):
    arguments = call.get("arguments", {}) if isinstance(call, dict) else {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return arguments if isinstance(arguments, dict) else {}


def execute_tool(call):
    name = call.get("name", "") if isinstance(call, dict) else ""
    arguments = parse_arguments(call)
    try:
        if name == "calculate_math":
            return {"result": str(safe_calculate(str(arguments.get("expression", ""))))}
        if name == "get_current_time":
            return {
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": arguments.get("timezone", "Asia/Shanghai"),
            }
        if name == "get_current_weather":
            return {"city": arguments.get("location", ""), "temperature": "22°C", "condition": "晴"}
        if name == "get_exchange_rate":
            return {
                "from": arguments.get("from_currency", ""),
                "to": arguments.get("to_currency", ""),
                "rate": 7.15,
            }
        if name == "translate_text":
            return {"translated_text": "hello world"}
        if name == "unit_converter":
            return {"result": round(float(arguments.get("value", 0)) * 0.621371, 4)}
        return {"error": f"未知工具: {name}"}
    except Exception as exc:
        return {"error": f"工具执行失败: {str(exc)[:120]}"}


def extract_prediction(text):
    if not text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).replace(",", "")
    numbers = NUMBER_PATTERN.findall(text)
    if not numbers:
        return None
    value = float(numbers[-1])
    return normalize_number(value)


def numbers_equal(prediction, ground_truth):
    if prediction is None:
        return False
    return abs(float(prediction) - float(ground_truth)) < 1e-6


def resolve_device(device):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype, device):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return torch.float32 if device == "cpu" else torch.float16


def load_native_model(checkpoint, args, device):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到权重文件: {checkpoint}")
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = MiniMindForCausalLM(config)
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=resolve_dtype(args.dtype, device)).eval()
    return model


def load_model_and_tokenizer(source, args, device):
    """Load either a native ``.pth`` checkpoint or a Transformers model."""
    source_path = Path(source).expanduser()
    if source_path.suffix.lower() == ".pth":
        tokenizer = AutoTokenizer.from_pretrained(args.native_tokenizer)
        model = load_native_model(source_path, args, device)
        return model, tokenizer, "native-pth"

    model_source = str(source_path.resolve()) if source_path.exists() else source
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        dtype=resolve_dtype(args.dtype, device),
    )
    model = model.to(device).eval()
    return model, tokenizer, "transformers"


@torch.inference_mode()
def generate(model, tokenizer, messages, tools, args, device):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
        open_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(args.do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generate_kwargs.update(temperature=args.temperature, top_p=args.top_p)
    generated_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        **generate_kwargs,
    )
    completion_ids = generated_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def build_case(index, expression):
    distractor = DISTRACTOR_NAMES[index % len(DISTRACTOR_NAMES)]
    names = ["calculate_math", distractor]
    if index % 2:
        names.reverse()
    return {
        "expression": expression,
        "prompt": PROMPT_TEMPLATES[index % len(PROMPT_TEMPLATES)].format(expression=expression),
        "gt": safe_calculate(expression),
        "tools": [TOOLS[name] for name in names],
    }


def run_case(model, tokenizer, case, args, device):
    messages = [{"role": "user", "content": case["prompt"]}]
    final_content = ""
    used_math_tool = False
    total_tool_calls = 0
    turns = []

    for turn_index in range(1, args.max_turns + 1):
        content = generate(model, tokenizer, messages, case["tools"], args, device)
        final_content = content
        calls = parse_tool_calls(content)
        call_records = []
        if not calls:
            turns.append({
                "turn": turn_index,
                "raw_response": content,
                "parsed_tool_calls": call_records,
            })
            break
        total_tool_calls += len(calls)
        messages.append({"role": "assistant", "content": content})
        for call in calls:
            used_math_tool = used_math_tool or call.get("name") == "calculate_math"
            result = execute_tool(call)
            call_records.append({
                "name": call.get("name", ""),
                "arguments": parse_arguments(call),
                "raw_call": call,
                "tool_result": result,
            })
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
        turns.append({
            "turn": turn_index,
            "raw_response": content,
            "parsed_tool_calls": call_records,
        })

    prediction = extract_prediction(final_content)
    answer_correct = numbers_equal(prediction, case["gt"])
    passed = answer_correct and (used_math_tool or not args.require_tool_call)
    return {
        "passed": passed,
        "answer_correct": answer_correct,
        "prediction": prediction,
        "used_math_tool": used_math_tool,
        "tool_calls": total_tool_calls,
        "raw_response": final_content,
        "turns": turns,
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


def safe_filename(value):
    filename = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._")
    return filename or "model"


def evaluate_model(source, label, section, cases, args, device, output_dir):
    print(f"[{section}] {args.model_name} ({label})", flush=True)
    model, tokenizer, source_type = load_model_and_tokenizer(source, args, device)
    results = []
    result_path = output_dir / f"{safe_filename(label)}.jsonl"
    try:
        with result_path.open("w", encoding="utf-8") as result_file:
            for index, case in enumerate(cases, 1):
                seed_everything(args.seed + index)
                result = run_case(model, tokenizer, case, args, device)
                record = {
                    "case_id": f"case_{index:04d}",
                    "case_index": index,
                    "label": label,
                    "model_source": str(source),
                    "model_source_type": source_type,
                    "expression": case["expression"],
                    "prompt": case["prompt"],
                    "ground_truth": case["gt"],
                    "tools": case["tools"],
                    **result,
                }
                results.append(record)
                result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                result_file.flush()
                mark = "✅" if result["passed"] else "❌"
                prediction = "None" if result["prediction"] is None else result["prediction"]
                print(
                    f"[{label}] {index}/{len(cases)} | {mark} | {case['expression']} "
                    f"| gt={case['gt']} | pred={prediction}",
                    flush=True,
                )
    finally:
        unload_model(model, device)
        del tokenizer
    print(f"[{label}] case records: {result_path}", flush=True)
    print(flush=True)
    return results, result_path


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind Transformers/.pth 数学 ToolUse 对比评测")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[str(REPO_ROOT / "minimind-3"), str(REPO_ROOT / "model_files" / "agent_768.pth")],
        metavar="MODEL",
        help="至少两个模型来源；支持 Transformers 目录/模型 ID 和原生 .pth",
    )
    parser.add_argument(
        "--labels", nargs="+", default=["full_sft", "agent"], metavar="LABEL"
    )
    parser.add_argument(
        "--native_tokenizer",
        default=str(REPO_ROOT / "model"),
        help="加载原生 .pth 时使用的 tokenizer 目录",
    )
    parser.add_argument("--model_name", default="minimind-3", help="报告中显示的模型名称")
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0/mps")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--max_turns", default=3, type=int)
    parser.add_argument("--num_cases", default=100, type=int, help="评测题数；前20题为固定基准，之后按种子生成")
    parser.add_argument("--case_seed", default=20260714, type=int, help="动态题目生成种子")
    parser.add_argument(
        "--difficulty",
        default="mixed",
        choices=["easy", "medium", "hard", "mixed"],
        help="第21题后的动态题目难度",
    )
    parser.add_argument("--cases_file", default=None, help="可选：每行一个表达式，或含 expression 字段的 JSONL")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--do_sample", default=0, type=int, choices=[0, 1])
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument(
        "--require_tool_call",
        default=1,
        type=int,
        choices=[0, 1],
        help="1=必须调用 calculate_math 且答案正确；0=只判断最终答案",
    )
    parser.add_argument("--show_tool_stats", default=0, type=int, choices=[0, 1])
    parser.add_argument(
        "--output_dir",
        default=None,
        help="逐case JSONL和summary.json保存目录；默认按时间写入evals/agent_math_results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_cases < 1:
        raise ValueError("--num_cases 必须大于 0")
    if len(args.models) < 2:
        raise ValueError("--models 至少需要两个模型")
    if len(args.models) != len(args.labels):
        raise ValueError("--models 与 --labels 数量必须一致")
    if len(set(args.labels)) != len(args.labels):
        raise ValueError("--labels must be unique so result filenames do not collide")
    device = resolve_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT / "evals" / "agent_math_results" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    expressions = build_expression_set(
        args.num_cases,
        args.case_seed,
        args.difficulty,
        args.cases_file,
    )
    cases = [build_case(i, expression) for i, expression in enumerate(expressions)]
    cases_path = output_dir / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as cases_file:
        for index, case in enumerate(cases, 1):
            cases_file.write(json.dumps({
                "case_id": f"case_{index:04d}",
                "case_index": index,
                **case,
            }, ensure_ascii=False) + "\n")

    all_results = {}
    result_paths = {}
    for model_index, (source, label) in enumerate(zip(args.models, args.labels), start=1):
        results, result_path = evaluate_model(
            source, label, f"M{model_index}", cases, args, device, output_dir
        )
        all_results[label] = results
        result_paths[label] = result_path

    comparison_counts = {
        "all_pass": 0,
        "all_fail": 0,
        "partial_pass": 0,
        **{f"{label}_only": 0 for label in args.labels},
    }
    paired_path = output_dir / "paired_results.jsonl"
    with paired_path.open("w", encoding="utf-8") as paired_file:
        for index, case in enumerate(cases):
            rows = {label: all_results[label][index] for label in args.labels}
            passed_labels = [label for label, row in rows.items() if row["passed"]]
            failed_labels = [label for label in args.labels if label not in passed_labels]
            if len(passed_labels) == len(args.labels):
                outcome, winner = "all_pass", "all"
            elif not passed_labels:
                outcome, winner = "all_fail", "neither"
            else:
                outcome = "partial_pass"
                winner = passed_labels[0] if len(passed_labels) == 1 else "multiple"
            comparison_counts[outcome] += 1
            if len(passed_labels) == 1:
                comparison_counts[f"{passed_labels[0]}_only"] += 1
            first = rows[args.labels[0]]
            paired_record = {
                "case_id": first["case_id"],
                "case_index": first["case_index"],
                "expression": case["expression"],
                "prompt": case["prompt"],
                "ground_truth": case["gt"],
                "tools": case["tools"],
                "outcome": outcome,
                "winner": winner,
                "passed_labels": passed_labels,
                "failed_labels": failed_labels,
                "models": {
                    label: {
                        key: first[key]
                        for key in (
                            "passed", "answer_correct", "prediction", "used_math_tool",
                            "tool_calls", "raw_response", "turns",
                        )
                    }
                    for label, first in rows.items()
                },
            }
            paired_file.write(json.dumps(paired_record, ensure_ascii=False) + "\n")

    print("=" * 60, flush=True)
    for label in args.labels:
        results = all_results[label]
        correct = sum(result["passed"] for result in results)
        print(f"{label}: {correct}/{len(results)} = {correct / len(results):.2%}", flush=True)

    if args.show_tool_stats:
        print("\nToolUse:", flush=True)
        for label in args.labels:
            results = all_results[label]
            used = sum(result["used_math_tool"] for result in results)
            answer_correct = sum(result["answer_correct"] for result in results)
            print(
                f"{label}: calculate_math={used}/{len(results)}, "
                f"answer_correct={answer_correct}/{len(results)}",
                flush=True,
            )

    summary = {
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "cases_file": str(cases_path),
        "paired_results_file": str(paired_path),
        "comparison_counts": comparison_counts,
        "evaluation_config": {
            "models": args.models,
            "labels": args.labels,
            "native_tokenizer": args.native_tokenizer,
            "device": device,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "max_turns": args.max_turns,
            "num_cases": args.num_cases,
            "case_seed": args.case_seed,
            "difficulty": args.difficulty,
            "cases_file": args.cases_file,
            "generation_seed": args.seed,
            "do_sample": bool(args.do_sample),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "require_tool_call": bool(args.require_tool_call),
        },
        "models": {},
    }
    for label in args.labels:
        results = all_results[label]
        passed = sum(result["passed"] for result in results)
        answer_correct = sum(result["answer_correct"] for result in results)
        used_math_tool = sum(result["used_math_tool"] for result in results)
        summary["models"][label] = {
            "result_file": str(result_paths[label]),
            "passed": passed,
            "total": len(results),
            "accuracy": passed / len(results),
            "answer_correct": answer_correct,
            "used_math_tool": used_math_tool,
        }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Cases: {cases_path}", flush=True)
    print(f"Paired results: {paired_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
