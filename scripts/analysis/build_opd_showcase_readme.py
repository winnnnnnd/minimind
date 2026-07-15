"""Build the personal Issue #804 showcase README from lightweight result files."""

import argparse
import html
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MATH_DIR = REPO_ROOT / "results" / "opd_issue804" / "math_tooluse_20"
QA_PATH = REPO_ROOT / "results" / "opd_issue804" / "readme_qa" / "generations.json"

DISPLAY_LABELS = {
    "full_sft": "A · Base / full_sft",
    "opd_gkd": "B · OPD-GKD",
    "agent": "C · Agent / Teacher",
}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def format_number(value):
    if value is None:
        return "None"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def render_math_cases():
    blocks = []
    for label in ("full_sft", "opd_gkd", "agent"):
        rows = load_jsonl(MATH_DIR / f"{label}.jsonl") if (MATH_DIR / f"{label}.jsonl").exists() else None
        if rows is None:
            paired = load_jsonl(MATH_DIR / "paired_results.jsonl")
            rows = []
            for record in paired:
                model = record["models"][label]
                rows.append(
                    {
                        "case_index": record["case_index"],
                        "expression": record["expression"],
                        "ground_truth": record["ground_truth"],
                        "prediction": model["prediction"],
                        "passed": model["passed"],
                    }
                )
        lines = [f"[{DISPLAY_LABELS[label]}]"]
        for row in rows:
            mark = "✅" if row["passed"] else "❌"
            lines.append(
                f"[{label}] {row['case_index']}/20 | {mark} | {row['expression']} | "
                f"gt={format_number(row['ground_truth'])} | pred={format_number(row['prediction'])}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_qa_cases():
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    answers = {run["label"]: run["answers"] for run in data["model_runs"]}
    blocks = []
    for index, case in enumerate(data["cases"], 1):
        sections = [
            f'<details open><summary><strong>{index}. [Q] {html.escape(case["prompt"])}</strong></summary>',
            "",
        ]
        for label in ("full_sft", "opd_gkd", "agent"):
            sections.extend(
                [
                    f"**[{DISPLAY_LABELS[label]}]**",
                    "",
                    f"<pre>{html.escape(answers[label][case['id']])}</pre>",
                    "",
                ]
            )
        sections.append("</details>")
        blocks.append("\n".join(sections))
    return "\n\n".join(blocks)


def build_readme():
    math_cases = render_math_cases()
    qa_cases = render_qa_cases()
    return f"""# MiniMind Issue #804：On-Policy Distillation（GKD / OPD）实验记录

> 这是个人 fork 的**实验展示分支**，用于保存可复现命令、轻量结果和实验分析；准备提交上游的干净实现位于 [`feat/804-on-policy-distillation`](https://github.com/winnnnnnd/minimind/tree/feat/804-on-policy-distillation)，该分支不包含本 README、模型、数据、checkpoint、日志或评测产物。

[Issue #804](https://github.com/jingyaogong/minimind/issues/804) · [正式 SwanLab 训练记录](https://swanlab.cn/@lacuson/MiniMind-OPD/runs/sm6x3b85cc3re7qgncc6o) · [GKD 论文](https://arxiv.org/abs/2306.13649) · [TRL GKDTrainer](https://huggingface.co/docs/trl/main/en/gkd_trainer)

## 结论先行

我使用 `minimind-3 (full_sft)` 作为 Student、`agent_768.pth` 作为冻结 Teacher，在 200 条 Agent-pass/Base-fail 数学 ToolUse prompt 上完成了一次纯 on-policy GKD 训练。干净实现采用论文定义的完整词表 generalized JSD，正式运行完成 200/200 个 micro-batch，没有出现 NaN、负散度、轨迹对齐失败或 checkpoint 加载错误。

在原作者 README 使用的固定 20 道轻 Agent 题上：

| 模型 | 严格成功率 | `calculate_math` 调用率 | 相对 Base |
|---|---:|---:|---:|
| Base / full_sft | 11/20 = 55.0% | 15/20 = 75.0% | — |
| **OPD-GKD** | **13/20 = 65.0%** | **19/20 = 95.0%** | **准确率 +10.0pp，调用率 +20.0pp** |
| Agent / Teacher | 17/20 = 85.0% | 20/20 = 100.0% | Teacher 上界参考 |

这个小样本复验说明：规范 GKD 实现能够训练，也能把 Teacher 的部分工具调用行为迁移给 Base；但 OPD 仍未追平 Teacher，而且 7 道开放问答没有显示出明确的通用知识提升，因此不应把它描述为“整体能力无损增强”。

![Issue #804 light Agent comparison](assets/opd_issue804/issue804_light_agent_comparison.png)

## 1. 实现与论文的对应关系

干净贡献分支实现 [Agarwal et al., ICLR 2024](https://arxiv.org/abs/2306.13649) 的 Generalized Knowledge Distillation：

$$
\\mathcal{{L}}_{{GKD}}=\\beta\\,\\mathrm{{KL}}(p_T\\parallel m)+(1-\\beta)\\,\\mathrm{{KL}}(p_S\\parallel m),\\qquad
m=\\beta p_T+(1-\\beta)p_S.
$$

- `lmbda=1` 使用 Student 自生成 completion 做纯 on-policy distillation，`lmbda=0` 使用数据集固定 completion，中间值按 batch 选择分支。
- `beta=0` 和 `beta=1` 分别退化为 forward KL 与 reverse KL，本次正式实验使用 `beta=0.5`。
- Teacher 全程冻结；只在有效 assistant completion token 上计算 loss，prompt、padding 和首个 EOS 后的 token 都被 mask。
- token divergence 使用完整词表，不使用早期实验中的 top-k surrogate、规则 reward、verifier、工具执行或 Base-reference KL。
- 训练器支持 `.pth` 与 Transformers 模型目录，并沿用仓库的 CPU、CUDA、MPS、DDP、checkpoint 和 SwanLab 设施。

## 2. 正式训练过程

| 配置 | 数值 |
|---|---|
| Student / Teacher | `minimind-3` / `agent_768.pth` |
| 训练 prompt | 200 条经实测验证的 Agent-pass/Base-fail ToolUse prompt |
| `lmbda` / `beta` | `1.0` / `0.5` |
| batch / gradient accumulation | `1` / `4` |
| micro-batch / optimizer step | `200` / `50` |
| learning rate | `1e-6`，cosine 衰减至 `1e-7` |
| rollout | temperature `1.0`，top-p `1.0`，top-k `0` |
| sequence / generation length | `1536` / `128` |
| device | Apple M4 Max，MPS，float16 |

![GKD training curves](assets/opd_issue804/issue804_gkd_training_curves.png)

最后一个 micro-batch 的 GKD loss 为 `0.000883`，forward KL 为 `0.003528`，reverse KL 为 `0.003766`，Teacher/Student Top-1 agreement 为 `1.0000`。图中淡线保留每一步原始值，实线使用 EMA(0.9)：第 100 步附近确实存在由不同 on-policy 序列带来的尖峰，但各项指标保持有限并重新下降，不能把尖峰本身误判成数值发散。生成长度在 1–128 token 之间变化也符合 on-policy prompt 难度和停止位置不同的预期。

完整交互曲线、配置和环境信息见 [SwanLab run `sm6x3b85cc3re7qgncc6o`](https://swanlab.cn/@lacuson/MiniMind-OPD/runs/sm6x3b85cc3re7qgncc6o)。轻量配置快照见 [`experiments/opd_issue804/training_config.json`](experiments/opd_issue804/training_config.json)。

## 3. 测试2：轻 Agent 任务对比

这里完全复用原 README 的 20 个固定数学表达式、工具定义和题目顺序；三个模型使用同一 tokenizer、`max_new_tokens=256`、`max_turns=3`、greedy decoding 和 seed 42。严格成功条件是：模型实际调用 `calculate_math`，并在工具交互结束后给出正确最终答案。

```text
{math_cases}

============================================================
full_sft: 11/20 = 55.00%
opd_gkd: 13/20 = 65.00%
agent: 17/20 = 85.00%

ToolUse:
full_sft: calculate_math=15/20, answer_correct=11/20
opd_gkd: calculate_math=19/20, answer_correct=13/20
agent: calculate_math=20/20, answer_correct=17/20
```

### 测试2总结与 case 分析

OPD 相对 Base 净增加 2 道成功题：第 4、5、14 题由错转对，第 15 题由对转错，因此提升并非简单记忆全部题目，也不是每个 case 单调改善。明显收益集中在幂运算与工具路由：OPD 将数学工具覆盖率从 75% 提高到 95%，符合训练数据针对 Base 工具路由短板筛选的预期。

仍需诚实指出两个限制：第一，20 题的 10pp 只对应 2 道题，统计方差较大；第二，OPD 在第 1、6、10、15、16、20 题仍会错误追加工具调用、错误提取参数或不能正确终止，尚未完整继承 Teacher 的 85% 能力。所有逐 turn 原始回复均保留在 [`paired_results.jsonl`](results/opd_issue804/math_tooluse_20/paired_results.jsonl)，没有只挑选正面 case。

## 4. 测试3：原作者问答形式

下面复用原 README 的 7 个问题，并使用同一确定性生成设置展示完整原始回复。这里是定性 case study，不把 7 题包装成通用 benchmark；本次自动执行环境没有读取个人 DeepSeek API key，因此没有为这 7 题追加新的 Judge 分数。

{qa_cases}

### 测试3总结与分析

- Base 与 OPD 都能生成可运行、能保留重复元素的快速排序代码；OPD 没有继承 Teacher 在代码题上“只讲概念、不输出代码”的明显退化。
- OPD 在“长江”问题开头从 Base 的“长江是中国首都”修正为“中国最长的河流”，但随后又生成了 `3.8万公里` 等严重幻觉，不能判定为完整正确。
- Base 与 OPD 都没有回答“牛顿提出万有引力”，也都没有正确解释海水盐分的来源和积累；说明此次任务定向蒸馏没有解决基础知识短板。
- Agent 在多数开放问答中出现大量“共舞”等模板化无意义文本；OPD 回复仍接近 Base 的语言形态，说明 200 条训练没有把 Teacher 的通用退化整体复制过来，但这只是定性观察。
- 三个模型都没有遵守 20 字摘要约束，因此指令遵循仍是共同短板。

完整生成记录见 [`results/opd_issue804/readme_qa/generations.json`](results/opd_issue804/readme_qa/generations.json)。

## 5. 更大样本的探索阶段实验

以下曲线来自**早期任务定向 OPD 实验管线**，其中包含 Base-reference KL，并非干净 PR 中的 generalized GKD 默认实现；它们用于说明任务甜点区、训练步数和能力权衡，不作为最终实现的同配置复验结果。

| 模型 | 500题数学严格准确率 | 数学工具调用率 | 200题通用问答分数 |
|---|---:|---:|---:|
| Base | 45.60% | 85.60% | 15.14 |
| OPD40 | **63.60%** | 99.40% | 15.09 |
| OPD80 | 63.40% | **99.80%** | 15.07 |
| OPD120 | 60.00% | 98.60% | 14.64 |
| OPD160 | 60.20% | 99.20% | 14.54 |
| OPD200 | 58.60% | 99.20% | 14.29 |
| Agent | 60.80% | 99.20% | 2.73 |

![Exploratory 500-case math curve](assets/opd_issue804/exploratory_math_500.png)

![Exploratory 200-case general QA curve](assets/opd_issue804/exploratory_general_qa_200.png)

![Exploratory capability tradeoff](assets/opd_issue804/exploratory_capability_tradeoff.png)

探索结果显示，40–80 个 micro-batch 已进入该任务的甜点区；继续训练并没有持续提高数学准确率，通用问答分数反而缓慢下降。这也是最终 PR 选择提供通用 GKD 机制、而不把任务筛选器、reference KL 或特定 early-stop 规则写死进训练器的原因。汇总数字见 [`exploratory_metrics.json`](experiments/opd_issue804/exploratory_metrics.json)。

## 6. 复现命令

### 训练

```bash
python trainer/train_opd.py \\
  --data_path /path/to/prompts.jsonl \\
  --student_model /path/to/minimind-3 \\
  --teacher_model /path/to/agent_768.pth \\
  --tokenizer_path /path/to/shared-tokenizer \\
  --student_use_moe 0 \\
  --teacher_use_moe 0 \\
  --batch_size 1 \\
  --accumulation_steps 4 \\
  --max_train_steps 200 \\
  --learning_rate 1e-6 \\
  --lmbda 1.0 \\
  --beta 0.5 \\
  --temperature 1.0 \\
  --distill_temperature 1.0 \\
  --rollout_top_p 1.0 \\
  --rollout_top_k 0 \\
  --max_seq_len 1536 \\
  --max_gen_len 128 \\
  --device mps \\
  --dtype float16 \\
  --use_wandb
```

CUDA 环境将最后两项替换为 `--device cuda:0 --dtype bfloat16`；多卡训练可按仓库现有 trainer 使用 `torchrun`。

### 固定 20 题 ToolUse 横评

```bash
python scripts/eval_agent_math.py \\
  --models /path/to/minimind-3 /path/to/opd_768.pth /path/to/agent_768.pth \\
  --labels full_sft opd_gkd agent \\
  --native_tokenizer /path/to/shared-tokenizer \\
  --device mps \\
  --dtype float16 \\
  --max_new_tokens 256 \\
  --max_turns 3 \\
  --num_cases 20 \\
  --seed 42 \\
  --do_sample 0 \\
  --require_tool_call 1 \\
  --show_tool_stats 1 \\
  --output_dir evals/issue804_readme_20
```

### 7 道问答与 DeepSeek Judge

```bash
export DEEPSEEK_API_KEY='...'
python scripts/eval_general_qa_deepseek.py \\
  --questions_file experiments/opd_issue804/readme_qa_cases.jsonl \\
  --models /path/to/minimind-3 /path/to/opd_768.pth /path/to/agent_768.pth \\
  --labels full_sft opd_gkd agent \\
  --native_tokenizer /path/to/shared-tokenizer \\
  --device mps \\
  --dtype float16 \\
  --do_sample 0 \\
  --output_dir evals/issue804_readme_qa
```

## 7. 验证范围与边界

- 单元测试与真实 MPS on-policy/off-policy 训练均已通过；正式保存的 63,912,192 参数 checkpoint 已严格重载。
- 当前机器没有 CUDA，因此 CUDA/DDP 是代码兼容路径和仓库范式对齐，不能冒充为本次实际硬件验证。
- 固定 20 题用于和原 README 直观对照，不等价于统计稳定 benchmark；更大样本结果来自早期实验实现，已经在标题和表格中单独标明。
- 训练 prompt 是为 Teacher-strong/Base-weak ToolUse 区域筛选的，因此收益不能外推到所有数学、Agent 或知识问答任务。
- 模型、训练数据和 checkpoint 均不放入展示分支；上游 PR 只提交通用训练器、测试与必要文档。

## 8. 分支职责

| 分支 | 用途 | 内容 |
|---|---|---|
| `feat/804-on-policy-distillation` | 向官方仓库提交 PR | 核心实现、单元测试、中文/英文使用文档 |
| `agent-opd-showcase` | 个人 fork 默认展示分支 | 本 README、轻量结果、图片和评测脚本 |
| 本地 `agentic_opd` | 长期实验工作区 | 数据、checkpoint、调试脚本、完整日志和历史实验，不直接提交上游 |

这种拆分保证维护者能查看效果证据和完整上下文，同时上游 diff 仍保持最小、可审查和无个人产物。
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(REPO_ROOT / "README.md"))
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.write_text(build_readme(), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
