# MiniMind Agent OPD：任务定向实验代码

> 这是个人 fork 的实验代码分支，用于公开 Issue #804 前期的 Agent ToolUse 数据构造、任务定向 OPD、Base-reference KL、评测和可视化代码。它不是准备合入官方仓库的最小 PR，也不等同于论文中的通用 GKD 实现。

- 干净 PR 实现：[`feat/804-on-policy-distillation`](https://github.com/winnnnnnd/minimind/tree/feat/804-on-policy-distillation)
- 完整实验报告：[`agent-opd-showcase`](https://github.com/winnnnnnd/minimind/tree/agent-opd-showcase)
- 正式 GKD SwanLab：[issue804-gkd-opd-pilot200-mps-beta050](https://swanlab.cn/@lacuson/MiniMind-OPD/runs/sm6x3b85cc3re7qgncc6o)
- 详细执行手册：[docs/agent_opd_runbook.md](docs/agent_opd_runbook.md)

## 实验目标

`agent_768.pth` 在数学 ToolUse 上明显强于 `minimind-3 (full_sft)`，但开放问答能力退化严重。本实验尝试只在“Agent 能做对、Base 做错”的 prompt 上蒸馏 Agent 行为，同时使用冻结的初始 Base 约束 Student，降低定向后训练造成的通用能力遗忘。

实验损失为：

$$
\mathcal{L}=\mathcal{L}_{\text{OPD-Agent}}(x_{\text{sweet}})
+\lambda_{\text{ref}}D_{\mathrm{KL}}(\pi_{\text{student}}\parallel\pi_{\text{base}}).
$$

- `OPD-Agent`：Student 在线生成多轮工具轨迹，冻结 Agent Teacher 在 Student 实际访问的 token 状态上提供蒸馏信号。
- `sweet prompts`：通过真实模型推理筛出的 Agent-pass/Base-fail 样本，而不是只按表达式模板静态猜测难度。
- `Base reference KL`：在通用 SFT replay 上约束当前 Student 不要偏离冻结初始 Base。
- `reference_replay_ratio`：只在部分 batch 执行 Base replay，减少第三个模型带来的训练成本。

这套损失是任务定制实验：它包含 top-k 候选近似、工具执行、结果校验、Base replay 和特定数据筛选；干净 PR 分支则只保留论文规范、任务无关的完整词表 generalized GKD。

## 已观测效果

| 模型 | 500题数学严格准确率 | `calculate_math` 调用率 | 200题通用问答分数 |
|---|---:|---:|---:|
| Base | 45.60% | 85.60% | 15.14 |
| **OPD40** | **63.60%** | 99.40% | 15.09 |
| OPD80 | 63.40% | **99.80%** | 15.07 |
| OPD120 | 60.00% | 98.60% | 14.64 |
| OPD160 | 60.20% | 99.20% | 14.54 |
| OPD200 | 58.60% | 99.20% | 14.29 |
| Agent Teacher | 60.80% | 99.20% | 2.73 |

最佳 checkpoint `OPD40` 相对 Base 的数学严格准确率提升 18.00 个百分点，通用问答只下降 0.05；继续训练到 200 个 micro-batch 后两项指标均回落，说明 reference KL 可以减缓遗忘，但仍需 checkpoint selection 和 early stopping。

上述数字、图表和逐 case 分析见个人展示分支；本分支不提交模型、数据集、checkpoint、完整日志或评测输出。

## 代码结构

| 路径 | 作用 |
|---|---|
| `trainer/train_opd.py` | 多轮 Agent OPD、工具观察增量 token 拼接、Base replay、checkpoint 和 SwanLab |
| `trainer/opd_utils.py` | sampled-K1/top-k/full-vocabulary 蒸馏代理、mask 和 reference reverse KL |
| `trainer/tool_utils.py` | 数学 Ground Truth 的边界安全校验 |
| `trainer/rollout_engine.py` | Torch/SGLang rollout 的 top-p、top-k 和可选 log-prob 控制 |
| `dataset/lm_dataset.py` | `ReferenceReplayDataset` 与仓库原有数据集 |
| `scripts/generate_agent_opd_candidates.py` | 程序化生成 8,000 条数学 ToolUse 候选 |
| `scripts/select_agent_opd_prompts.py` | Base-first、Agent-second 级联实测筛选 |
| `scripts/split_agent_opd_data.py` | 分层切分 train/validation/pilot |
| `scripts/eval_agent_math.py` | 多模型数学 ToolUse 规则评测与原始回答保存 |
| `scripts/eval_general_qa_deepseek.py` | 多模型开放问答与匿名 DeepSeek Judge |
| `scripts/eval_knowledge_benchmark.py` | 选择题知识评测与随机基线 |
| `scripts/analysis/` | case 导出、问答生成、benchmark 准备和图表 |
| `tests/` | 实验 loss、工具校验、评测解析和 rollout 参数测试 |

## 环境与外部文件

代码基于官方 MiniMind master。运行完整实验还需要自行准备以下未提交内容：

```text
minimind-3/                         # Base / Student Transformers目录
model_files/agent_768.pth           # Agent Teacher
dataset/sft_t2t_mini.jsonl          # Base-reference replay，可替换为兼容SFT JSONL
```

Apple Silicon 使用：

```bash
--device mps --dtype float16
```

NVIDIA Ampere 或更新架构建议使用：

```bash
--device cuda:0 --dtype bfloat16
```

##  OPD + Base reference KL

```bash
python trainer/train_opd.py \
  --data_path dataset/opd_agent_sweet_selection/splits/opd_pilot_200.jsonl \
  --student_model minimind-3 \
  --teacher_model model_files/agent_768.pth \
  --reference_model minimind-3 \
  --reference_data_path dataset/sft_t2t_mini.jsonl \
  --lambda_ref 0.10 \
  --reference_replay_ratio 0.25 \
  --reference_batch_size 1 \
  --reference_max_len 512 \
  --reference_max_samples 10000 \
  --reference_sample_stride 10 \
  --student_tokenizer minimind-3 \
  --teacher_tokenizer model \
  --save_dir out/opd_agent_ref_pilot \
  --checkpoint_dir checkpoints/opd_agent_ref_pilot \
  --save_weight opd_agent_ref \
  --batch_size 1 \
  --num_workers 0 \
  --accumulation_steps 4 \
  --max_train_steps 200 \
  --save_interval 40 \
  --keep_step_checkpoints \
  --learning_rate 1e-6 \
  --max_gen_len 256 \
  --max_total_len 2048 \
  --max_turns 3 \
  --num_generations 1 \
  --rollout_temperature 1.0 \
  --rollout_top_p 1.0 \
  --rollout_top_k 0 \
  --rollout_alignment_retries 3 \
  --distill_top_k 16 \
  --top_k_strategy only_stu \
  --weight_mode student_p \
  --thinking_ratio 0 \
  --device mps \
  --dtype float16 \
  --log_interval 1 \
  --metrics_ema_decay 0.95 \
  --use_wandb \
  --wandb_project MiniMind-Agent-OPD \
  --wandb_run_name agent-opd-pilot200-mps-refkl010 \
  --wandb_mode cloud \
  --wandb_logdir swanlog
```

## 三模型数学 ToolUse 评测

```bash
python scripts/eval_agent_math.py \
  --models minimind-3 out/opd_agent_ref_pilot/opd_agent_ref_768.pth model_files/agent_768.pth \
  --labels base opd_pilot200 agent \
  --native_tokenizer model \
  --num_cases 500 \
  --difficulty mixed \
  --require_tool_call 1 \
  --show_tool_stats 1 \
  --device mps \
  --dtype float16 \
  --output_dir evals/agent_math_results/opd_pilot200_three_way
```

##  通用能力回归评测

```bash
export DEEPSEEK_API_KEY='你的密钥'
python scripts/eval_general_qa_deepseek.py \
  --models minimind-3 out/opd_agent_ref_pilot/opd_agent_ref_768.pth model_files/agent_768.pth \
  --labels base opd_pilot200 agent \
  --native_tokenizer model \
  --device mps \
  --dtype float16 \
  --output_dir evals/general_qa_results/opd_pilot200_three_way
```
