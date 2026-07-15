# MiniMind Agent OPD：任务定向实验代码

> 这是个人 fork 的实验代码分支，用于公开 Issue #804 前期的 Agent ToolUse 数据构造、甜点区筛选、任务定向 OPD、Base-reference KL、评测和可视化代码。它不是准备合入官方仓库的最小 PR，也不等同于论文中的通用 GKD 实现。

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

## 本分支验证

迁移到独立 worktree 后执行了完整语法检查和测试：

```text
Ran 45 tests in 0.057s
OK
```

覆盖范围包括 top-k 候选策略、sampled-K1、完整词表反向 KL 梯度、Base-reference KL、assistant mask、空 mask、增量 token 后缀、reference replay 调度、工具数值边界、知识选择题解析、随机基线和通用问答数据生成。测试只保存在个人实验分支，不进入上游 PR。

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

## 1. 基础检查

```bash
python -m py_compile \
  dataset/lm_dataset.py \
  trainer/opd_utils.py \
  trainer/tool_utils.py \
  trainer/train_opd.py \
  scripts/generate_agent_opd_candidates.py \
  scripts/select_agent_opd_prompts.py

python -m unittest discover -s tests -p 'test_*utils.py' -v
```

## 2. 生成候选数据

```bash
python scripts/generate_agent_opd_candidates.py \
  --num_candidates 8000 \
  --output dataset/opd_agent_candidates/agent_math_candidates_8k.jsonl
```

这一步不加载模型，也不调用外部 LLM；所有表达式都可以由规则程序计算 Ground Truth。

## 3. 级联筛选甜点区

```bash
python scripts/select_agent_opd_prompts.py \
  --data_path dataset/opd_agent_candidates/agent_math_candidates_8k.jsonl \
  --gt_count 1 \
  --limit 3000 \
  --screen_mode cascade \
  --target_teacher_only 1200 \
  --both_pass_ratio 0 \
  --output_dir dataset/opd_agent_sweet_selection \
  --device mps \
  --dtype float16 \
  --progress_interval 20
```

`cascade` 先运行 Base，只将 Base 失败的题交给 Agent，从而减少不必要的双模型全量推理；扩大 `--limit` 时可加入 `--reuse_cache` 续筛。

## 4. 固定切分

```bash
python scripts/split_agent_opd_data.py \
  --input dataset/opd_agent_sweet_selection/teacher_pass_student_fail.jsonl \
  --output_dir dataset/opd_agent_sweet_selection/splits \
  --train_count 1000 \
  --validation_count 200 \
  --pilot_count 200 \
  --seed 42
```

## 5. OPD + Base reference KL pilot

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

## 6. 三模型数学 ToolUse 评测

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

## 7. 通用能力回归评测

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

DeepSeek 只作为匿名 Judge，不参与 SFT 或 OPD 训练。

## 重要边界

- 该实验分支为了研究特定 MiniMind Agent ToolUse 场景，包含明显的任务工程，不应被当作通用 GKD 标准答案。
- 甜点数据来自 Teacher 与 Base 的实测差异，若更换权重必须重新筛选。
- 500 题规则评测要求实际调用 `calculate_math` 且最终答案正确，不只是文本里碰巧出现 Ground Truth。
- Base-reference KL 只在 replay batch 上产生非零曲线，规律性尖峰是稀疏回放设计，而不是必然的数值故障。
- `loss/opd` 是蒸馏代理，不要求像监督交叉熵一样单调下降；应结合 reverse KL、工具成功率、通用回归和 checkpoint 横评判断。
- 本分支不包含任何模型、私有 API key、训练数据、checkpoint 或完整实验日志。

## 与上游 PR 的关系

| 分支 | 定位 |
|---|---|
| `feat/804-on-policy-distillation` | 仅提交任务无关的 `train_opd.py`、`opd_utils.py` 和必要 `rollout_engine.py` 修改 |
| `agent-opd-showcase` | 默认展示分支，保存实验报告、图表和轻量结果 |
| `agent-opd-experiments` | 当前分支，保存可供复用的任务定向实验代码 |

如果只想审阅或使用论文规范的 generalized GKD，请使用干净 PR 分支；如果想复现实验中的数据筛选、Base KL、工具交互和评测流程，再使用本分支。
