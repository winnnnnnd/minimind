"""Render lightweight figures for the Issue #804 OPD/GKD showcase report."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from swanlab.data.porter import DataPorter


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWAN_RUN = REPO_ROOT / "swanlog" / "run-20260715_211107-sm6x3b85cc3re7qgncc6o"
DEFAULT_MATH_SUMMARY = (
    REPO_ROOT
    / "results"
    / "opd_issue804"
    / "math_tooluse_500"
    / "summary.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "evals" / "agent_opd_summary" / "figures"

FONT_LIGHT = Path("/System/Library/Fonts/STHeiti Light.ttc")
FONT_MEDIUM = Path("/System/Library/Fonts/STHeiti Medium.ttc")

BG = "#F5F7FB"
PANEL = "#FFFFFF"
TEXT = "#172033"
MUTED = "#687386"
GRID = "#DDE3EC"
BLUE = "#2563EB"
ORANGE = "#F97316"
TEAL = "#0F9D8A"
PURPLE = "#7C3AED"
RED = "#DC4C64"


def font(size, medium=False):
    return ImageFont.truetype(str(FONT_MEDIUM if medium else FONT_LIGHT), size)


def text_size(draw, value, fnt):
    box = draw.textbbox((0, 0), str(value), font=fnt)
    return box[2] - box[0], box[3] - box[1]


def centered_text(draw, x, y, value, fnt, fill=TEXT):
    width, height = text_size(draw, value, fnt)
    draw.text((x - width / 2, y - height / 2), str(value), font=fnt, fill=fill)


def right_text(draw, x, y, value, fnt, fill=MUTED):
    width, height = text_size(draw, value, fnt)
    draw.text((x - width, y - height / 2), str(value), font=fnt, fill=fill)


def panel(draw, box):
    draw.rounded_rectangle(box, radius=22, fill=PANEL, outline=GRID, width=2)


def ema(values, decay=0.9):
    output = []
    running = None
    for value in values:
        running = value if running is None else decay * running + (1 - decay) * value
        output.append(running)
    return output


def load_swan_metrics(run_dir):
    values = defaultdict(list)
    with DataPorter().open_for_sync(str(run_dir)) as porter:
        porter.parse()
        for scalar in porter._scalars:
            if not scalar.key.startswith("__swanlab__"):
                values[scalar.key].append((int(scalar.step), float(scalar.metric["data"])))
    return {key: sorted(rows) for key, rows in values.items()}


def line_chart(draw, box, title, series, y_min, y_max, y_ticks, y_format="{:.2f}"):
    x0, y0, x1, y1 = box
    draw.text((x0 + 26, y0 + 20), title, font=font(26, True), fill=TEXT)
    px0, py0, px1, py1 = x0 + 80, y0 + 76, x1 - 28, y1 - 90
    for tick in y_ticks:
        y = py1 - (tick - y_min) / (y_max - y_min) * (py1 - py0)
        draw.line((px0, y, px1, y), fill=GRID, width=2)
        right_text(draw, px0 - 12, y, y_format.format(tick), font(16))
    draw.line((px0, py0, px0, py1), fill=MUTED, width=2)
    draw.line((px0, py1, px1, py1), fill=MUTED, width=2)
    for step in (1, 50, 100, 150, 200):
        x = px0 + (step - 1) / 199 * (px1 - px0)
        centered_text(draw, x, py1 + 28, step, font(15), MUTED)

    for item in series:
        rows = item["rows"]
        steps = [step for step, _ in rows]
        raw = [value for _, value in rows]
        values = ema(raw, item.get("ema", 0.9)) if item.get("smooth", True) else raw
        points = [
            (
                px0 + (step - 1) / 199 * (px1 - px0),
                py1 - (max(y_min, min(y_max, value)) - y_min) / (y_max - y_min) * (py1 - py0),
            )
            for step, value in zip(steps, values)
        ]
        if item.get("show_raw", False):
            raw_points = [
                (
                    px0 + (step - 1) / 199 * (px1 - px0),
                    py1 - (max(y_min, min(y_max, value)) - y_min) / (y_max - y_min) * (py1 - py0),
                )
                for step, value in rows
            ]
            draw.line(raw_points, fill=item["raw_color"], width=2)
        draw.line(points, fill=item["color"], width=4, joint="curve")

    legend_x = x0 + 28
    legend_y = y1 - 24
    for item in series:
        draw.line((legend_x, legend_y, legend_x + 28, legend_y), fill=item["color"], width=4)
        draw.text((legend_x + 36, legend_y - 11), item["name"], font=font(15), fill=TEXT)
        legend_x += 48 + text_size(draw, item["name"], font(15))[0]


def render_training(metrics, output):
    image = Image.new("RGB", (1800, 1120), BG)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((62, 42), "Issue #804：规范 GKD / OPD 训练过程", font=font(42, True), fill=TEXT)
    draw.text(
        (62, 98),
        "200 个 on-policy micro-batches｜β=0.5 generalized JSD｜MPS float16｜实线为 EMA(0.9)，淡线为逐步原值",
        font=font(21),
        fill=MUTED,
    )
    boxes = [(55, 150, 875, 615), (925, 150, 1745, 615), (55, 650, 875, 1065), (925, 650, 1745, 1065)]
    for box in boxes:
        panel(draw, box)

    line_chart(
        draw,
        boxes[0],
        "GKD loss",
        [{"name": "loss/gkd", "rows": metrics["loss/gkd"], "color": BLUE, "raw_color": (37, 99, 235, 55), "show_raw": True}],
        0,
        0.045,
        [0, 0.01, 0.02, 0.03, 0.04],
        "{:.2f}",
    )
    line_chart(
        draw,
        boxes[1],
        "Teacher–Student token divergence",
        [
            {"name": "Forward KL", "rows": metrics["distill/forward_kl"], "color": PURPLE, "raw_color": (124, 58, 237, 45), "show_raw": True},
            {"name": "Reverse KL", "rows": metrics["distill/reverse_kl"], "color": ORANGE, "raw_color": (249, 115, 22, 35), "show_raw": True},
        ],
        0,
        0.24,
        [0, 0.06, 0.12, 0.18, 0.24],
        "{:.2f}",
    )
    line_chart(
        draw,
        boxes[2],
        "Teacher–Student Top-1 agreement",
        [{"name": "Top-1 agreement", "rows": metrics["distill/top1_agreement"], "color": TEAL, "raw_color": (15, 157, 138, 55), "show_raw": True}],
        0.75,
        1.01,
        [0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
        "{:.2f}",
    )
    line_chart(
        draw,
        boxes[3],
        "On-policy completion length",
        [{"name": "generated tokens", "rows": metrics["rollout/avg_completion_length"], "color": RED, "raw_color": (220, 76, 100, 55), "show_raw": True}],
        0,
        130,
        [0, 32, 64, 96, 128],
        "{:.0f}",
    )
    image.save(output, quality=95)


def bar_group(draw, box, title, labels, values, colors, y_max, suffix="%"):
    x0, y0, x1, y1 = box
    draw.text((x0 + 30, y0 + 24), title, font=font(28, True), fill=TEXT)
    # Leave a dedicated header band for the title and delta badge so values
    # near 100% do not overlap the annotation.
    px0, py0, px1, py1 = x0 + 72, y0 + 140, x1 - 36, y1 - 95
    for tick in range(0, int(y_max) + 1, 20):
        y = py1 - tick / y_max * (py1 - py0)
        draw.line((px0, y, px1, y), fill=GRID, width=2)
        right_text(draw, px0 - 12, y, f"{tick}{suffix}", font(16))
    width = (px1 - px0) / len(labels)
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        cx = px0 + width * (index + 0.5)
        bar_w = width * 0.48
        top = py1 - value / y_max * (py1 - py0)
        draw.rounded_rectangle((cx - bar_w / 2, top, cx + bar_w / 2, py1), radius=12, fill=color)
        centered_text(draw, cx, top - 28, f"{value:.1f}{suffix}", font(22, True), color)
        centered_text(draw, cx, py1 + 38, label, font(19, True), TEXT)


def render_math(summary, output):
    image = Image.new("RGB", (1800, 930), BG)
    draw = ImageDraw.Draw(image)
    models = summary["models"]
    total = models["full_sft"]["total"]
    draw.text((62, 42), f"{total} 题数学 ToolUse：Base / OPD-GKD / Agent 对比", font=font(42, True), fill=TEXT)
    draw.text(
        (62, 98),
        "前 20 题复用原 README，后续题由固定 seed 生成｜相同工具与 greedy 解码｜严格成功需工具调用且答案正确",
        font=font(22),
        fill=MUTED,
    )
    left = (55, 155, 875, 865)
    right = (925, 155, 1745, 865)
    panel(draw, left)
    panel(draw, right)
    keys = ["full_sft", "opd_gkd", "agent"]
    labels = ["Base", "OPD-GKD", "Agent"]
    colors = ["#7A8598", BLUE, ORANGE]
    bar_group(draw, left, "严格成功率", labels, [models[k]["accuracy"] * 100 for k in keys], colors, 100)
    bar_group(draw, right, "calculate_math 调用率", labels, [models[k]["used_math_tool"] / models[k]["total"] * 100 for k in keys], colors, 100)
    accuracy_gain = (models["opd_gkd"]["accuracy"] - models["full_sft"]["accuracy"]) * 100
    tool_gain = (
        models["opd_gkd"]["used_math_tool"] / models["opd_gkd"]["total"]
        - models["full_sft"]["used_math_tool"] / models["full_sft"]["total"]
    ) * 100
    draw.rounded_rectangle((465, 185, 775, 235), radius=20, fill="#DBEAFE")
    centered_text(draw, 620, 210, f"OPD vs Base  {accuracy_gain:+.1f} pp", font(20, True), BLUE)
    draw.rounded_rectangle((1335, 185, 1645, 235), radius=20, fill="#DBEAFE")
    centered_text(draw, 1490, 210, f"OPD vs Base  {tool_gain:+.1f} pp", font(20, True), BLUE)
    image.save(output, quality=95)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swan_run", default=str(DEFAULT_SWAN_RUN))
    parser.add_argument("--math_summary", default=str(DEFAULT_MATH_SUMMARY))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_swan_metrics(Path(args.swan_run).expanduser().resolve())
    summary = json.loads(Path(args.math_summary).read_text(encoding="utf-8"))
    training_path = output_dir / "issue804_gkd_training_curves.png"
    math_path = output_dir / "issue804_gkd_math_tooluse_500.png"
    render_training(metrics, training_path)
    render_math(summary, math_path)
    print(json.dumps({"training": str(training_path), "math": str(math_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
