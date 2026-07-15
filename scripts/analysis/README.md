# Analysis helpers

这里集中放置不参与训练、只用于评测结果整理和人工 case 分析的辅助脚本。

## Agent Math case 导出

`export_agent_math_cases.py` 读取 `eval_agent_math.py` 生成的
`paired_results.jsonl`，默认只导出：

- `agent_math_case_analysis.xlsx`：只有一个“Cases”工作表，一行对应一个 case，
  全部数据行按照四类结果分别着色。

XLSX 颜色约定：

- 绿色：都对；
- 蓝色：仅 Agent 对；
- 黄色：仅 Full-SFT 对；
- 红色：都错。

安装 XLSX 导出所需的可选依赖：

```bash
python -m pip install -r scripts/analysis/requirements.txt
```

导出当前 500 道评测结果：

```bash
python scripts/analysis/export_agent_math_cases.py \
  --input evals/agent_math_results/repro_500_case_analysis/paired_results.jsonl \
  --format xlsx
```

脚本仍支持 `--format csv`，但 CSV 格式不能保存颜色。
