# MiniMind Agent OPD 实验执行手册

本实验固定使用：

- 学生初始模型：`minimind-3`（full_sft，保留其通用问答能力）
- 教师模型：`model_files/agent_768.pth`（提供更强的 Agent/数学 ToolUse 分布）
- OPD 数据：从 8,000 条程序化候选中实测筛出的 Agent 对、Base 错甜点样本
- 保护数据：`dataset/sft_t2t_mini.jsonl` 的固定抽样上下文，仅用于冻结 Base reference KL
- 第一阶段不做 DeepSeek SFT，不做冷启动；DeepSeek 只用于训练后的通用能力评测

以下命令都从仓库根目录执行。每一步成功后再进入下一步。

## 设备参数

- Apple Silicon（M1/M2/M3/M4）：使用 `--device mps --dtype float16`。学生保持 FP32 训练，冻结教师使用 FP16，避免依赖不同 PyTorch 版本中不一致的 MPS autocast。
- NVIDIA Ampere 或更新架构：优先使用 `--device cuda:0 --dtype bfloat16`。
- 不支持 BF16 的 NVIDIA GPU：使用 `--device cuda:0 --dtype float16`，脚本会启用 GradScaler。
- `--device auto` 会依次选择 CUDA、MPS、CPU；为了实验记录可复现，正式命令建议显式指定设备。

## 1. 语法与 OPD loss 单元测试

```bash
python -m py_compile dataset/lm_dataset.py trainer/opd_utils.py trainer/tool_utils.py trainer/train_opd.py scripts/generate_agent_opd_candidates.py scripts/select_agent_opd_prompts.py
python -m unittest discover -s tests -p 'test_*utils.py' -v
```

第一条检查新增/修改脚本能否被 Python 正常解析。第二条除原 OPD loss 外，还检查精确 Base reverse KL 的方向、mask 和空 mask 数值稳定性。

## 2. 生成 8,000 条候选（零模型推理）

```bash
python scripts/generate_agent_opd_candidates.py \
  --num_candidates 8000 \
  --output dataset/opd_agent_candidates/agent_math_candidates_8k.jsonl
```

该步骤只做可校验的表达式与 prompt 组合，不加载模型，也不调用 DeepSeek。默认配额为：路由/干扰工具 3,000 条、拿到结果后结束 1,800 条、简单运算与乘方 2,000 条、已观察到可能有正迁移的复杂模板探针 1,200 条。候选被固定 seed 打乱，因此前 2,000 条也是混合分布。manifest 会记录实际类别、模板和数学工具位置分布。

8,000 是候选池，不是必须全部实测或全部训练的数据量。

## 3. 级联实测筛选甜点样本

先用真实输出目录跑 20 条链路 smoke；后续会复用这 20 条缓存，不会浪费推理：

```bash
python scripts/select_agent_opd_prompts.py \
  --data_path dataset/opd_agent_candidates/agent_math_candidates_8k.jsonl \
  --gt_count 1 \
  --limit 20 \
  --screen_mode cascade \
  --target_teacher_only 1000 \
  --both_pass_ratio 0 \
  --output_dir dataset/opd_agent_sweet_selection \
  --device mps \
  --dtype float16 \
  --progress_interval 20
```

链路正常后扩到前 2,000 条：

```bash
python scripts/select_agent_opd_prompts.py \
  --data_path dataset/opd_agent_candidates/agent_math_candidates_8k.jsonl \
  --gt_count 1 \
  --limit 2000 \
  --screen_mode cascade \
  --target_teacher_only 1000 \
  --both_pass_ratio 0 \
  --output_dir dataset/opd_agent_sweet_selection \
  --device mps \
  --dtype float16 \
  --progress_interval 20 \
  --reuse_cache
```

`cascade` 不会让两个模型各跑全部候选。它先让 Base 评测当前切片，只把 Base 失败项交给 Agent；Agent 收集到 1,000 条“Agent 对、Base 错”后立即停止。报告中的 `inference_counts` 会给出实际 Base/Agent 推理数及相对双模型全量评测节省量。

如果 2,000 条不足以得到目标数量，扩大到 4,000 条并复用、续写已有缓存：

```bash
python scripts/select_agent_opd_prompts.py \
  --data_path dataset/opd_agent_candidates/agent_math_candidates_8k.jsonl \
  --gt_count 1 \
  --limit 4000 \
  --screen_mode cascade \
  --target_teacher_only 1000 \
  --both_pass_ratio 0 \
  --output_dir dataset/opd_agent_sweet_selection \
  --device mps \
  --dtype float16 \
  --progress_interval 20 \
  --reuse_cache
```

仍不足时依次把 `--limit` 调到 6,000、8,000，保持同一个输出目录并使用 `--reuse_cache`。只有前一切片不够才扩大，不应一开始就双模型跑完 8,000 条。

最终训练集为：

```text
dataset/opd_agent_sweet_selection/opd_train.jsonl
```

级联模式下它只含实测的“Agent 通过、Base 失败”样本，不加入双方通过样本；通用能力保护由独立 Base KL 路径承担。`both_fail.jsonl` 是 Base 失败后 Agent 也失败的已测样本，不进入训练。Base 已通过的候选不会再浪费 Agent 推理。

查看分桶报告：

```bash
python -m json.tool dataset/opd_agent_sweet_selection/selection_report.json
```

报告中的 `sweet_spot_distribution` 会显示最终甜点数据按类别、表达式模板和数学工具位置的分布。

## 4. OPD + Base reference KL 四批 smoke test

在训练前，先把1,200条已验证甜点数据固定分成1,000训练、200验证，并从训练集取200条pilot：

```bash
python scripts/split_agent_opd_data.py \
  --input dataset/opd_agent_sweet_selection/teacher_pass_student_fail.jsonl \
  --output_dir dataset/opd_agent_sweet_selection/splits \
  --train_count 1000 \
  --validation_count 200 \
  --pilot_count 200 \
  --seed 42
```

切分按类别、表达式模板和数学工具位置联合分层。pilot是训练集的子集，验证集与两者完全隔离。

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
  --save_dir out/opd_agent_ref_smoke \
  --checkpoint_dir checkpoints/opd_agent_ref_smoke \
  --save_weight opd_agent_ref_smoke \
  --batch_size 1 \
  --accumulation_steps 1 \
  --max_train_steps 4 \
  --save_interval 4 \
  --learning_rate 1e-6 \
  --max_gen_len 256 \
  --max_total_len 2048 \
  --max_turns 3 \
  --distill_top_k 16 \
  --top_k_strategy only_stu \
  --weight_mode student_p \
  --thinking_ratio 0 \
  --device mps \
  --dtype float16 \
  --debug_mode \
  --debug_interval 1
```

这个命令严格加载三个独立模型：可训练学生、冻结 Agent、冻结初始 Base。OPD 每批都执行；Base replay 每四批执行一次，因此 smoke 必须至少四批。回放步的实际权重是 `0.10 / 0.25 = 0.40`，长期按全部 OPD batch 平均后正好是 `lambda_ref=0.10`。日志中 `Ref KL` 后的 `*` 表示本批执行了回放，`-` 表示本批未回放。

Base replay 从 1.6GB 通用文件按固定步长抽 10,000 条上下文，不把整份文件读入内存。它不是 SFT：没有使用数据里的回答做交叉熵，而是让当前学生在 assistant token 位置不要偏离冻结 Base 的完整输出分布。

## 5. 200 批 OPD + Base KL pilot

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

有效 batch 为 4，共最多 50 次 optimizer update，并在第 40、80、120、160、200 批保存独立权重。冻结 Base KL 已作为主要的能力保持项；低学习率、短 pilot 和 checkpoint 早停仍然保留。

多轮工具轨迹采用增量token构建：模型原始采样的action token始终保留；程序只从“关闭assistant”和“追加工具结果”两个标准模板渲染中提取新增后缀，并将该后缀以`mask=0`追加。这样不会因同一文本存在不同BPE切分（例如`["Ġ3", "6"]`与`["Ġ36"]`）而改写on-policy轨迹。`--rollout_alignment_retries`只处理两个标准模板自身无法保持前缀的异常，不再用于规避采样token重新编码差异。

仓库统一使用 SwanLab 提供训练可视化，`--use_wandb`只是沿用现有trainer参数名。标量会按以下面板自动生成曲线：

- `loss/`：OPD代理loss、EMA、加权Base保护项和当前随机目标；
- `distill/`：Agent reverse KL、top-k重合率和top-1一致率；
- `reference/`：仅在真实回放步记录Base KL，避免用假零值制造锯齿图；
- `rollout/`：验证成功率、数学工具调用率、未正常结束率和EMA；
- `rollout/alignment_retry_rate`：多轮模板严格对齐失败后成功重采样的比例；偶发重试可接受，持续偏高说明需要检查模板；
- `optimizer/`：学习率、optimizer update与裁剪前梯度范数；
- `system/`：单步耗时与预计剩余时间。

`loss/opd`是policy-gradient surrogate，不要求像交叉熵一样单调下降。判断OPD是否有效应优先查看`distill/reverse_kl_ema`、`rollout/verified_success_rate_ema`以及稀疏的`reference/reverse_kl_ema`。

## 6. 三模型数学ToolUse评测

使用完全相同的500道表达式，对比原始Base、OPD pilot最终权重和Agent教师：

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

这个评测在固定的前20题之外按固定seed动态生成表达式。若最终模型不理想，可将OPD文件名依次替换为`step000040`、`step000080`、`step000120`、`step000160`和`step000200`做checkpoint选择；每次使用不同`output_dir`。

## 7. 三模型通用能力回归评测

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

DeepSeek在这里仅是三个匿名回答的盲评裁判，不产生训练回答。只有数学ToolUse明显提高且通用分数没有不可接受下降的checkpoint才进入扩大训练阶段。若数学提高但通用能力明显下降，再增加full_sft reference KL或缩短训练，而不是用DeepSeek做冷启动。
