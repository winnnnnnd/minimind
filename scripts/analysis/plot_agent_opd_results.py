"""Render the Agent-OPD math and general-QA evaluation results as PNG charts."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATH_CURVE = (
    REPO_ROOT / "evals" / "agent_math_results" / "opd_learning_curve_40_160" / "summary.json"
)
DEFAULT_MATH_THREE_WAY = (
    REPO_ROOT / "evals" / "agent_math_results" / "opd_pilot200_three_way" / "summary.json"
)
DEFAULT_GENERAL = (
    REPO_ROOT
    / "evals"
    / "general_qa_results"
    / "generated_qa_200_learning_curve_with_agent"
    / "results.json"
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
GRAY = "#7A8598"
LIGHT_BLUE = "#DBEAFE"


def font(size, medium=False):
    path = FONT_MEDIUM if medium else FONT_LIGHT
    return ImageFont.truetype(str(path), size)


def text_size(draw, text, fnt):
    box = draw.textbbox((0, 0), str(text), font=fnt)
    return box[2] - box[0], box[3] - box[1]


def centered_text(draw, xy, text, fnt, fill=TEXT):
    width, height = text_size(draw, text, fnt)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=fnt, fill=fill)


def right_text(draw, xy, text, fnt, fill=TEXT):
    width, height = text_size(draw, text, fnt)
    draw.text((xy[0] - width, xy[1] - height / 2), text, font=fnt, fill=fill)


def load_results(math_curve_path, math_three_way_path, general_path):
    math_curve = json.loads(Path(math_curve_path).read_text(encoding="utf-8"))
    math_three = json.loads(Path(math_three_way_path).read_text(encoding="utf-8"))
    general = json.loads(Path(general_path).read_text(encoding="utf-8"))

    labels = ["base", "opd40", "opd80", "opd120", "opd160", "opd200", "agent"]
    math = {
        "base": math_three["models"]["base"],
        "opd40": math_curve["models"]["opd40"],
        "opd80": math_curve["models"]["opd80"],
        "opd120": math_curve["models"]["opd120"],
        "opd160": math_curve["models"]["opd160"],
        "opd200": math_three["models"]["opd_pilot200"],
        "agent": math_three["models"]["agent"],
    }
    aggregates = general["aggregates"]
    qa = {label: aggregates[label] for label in labels}
    return labels, math, qa


def new_canvas(width=1800, height=1050, title=None, subtitle=None):
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    if title:
        draw.text((70, 48), title, font=font(42, True), fill=TEXT)
    if subtitle:
        draw.text((70, 105), subtitle, font=font(23), fill=MUTED)
    return image, draw


def panel(draw, box):
    draw.rounded_rectangle(box, radius=24, fill=PANEL, outline=GRID, width=2)


def line_chart(
    draw,
    box,
    x_labels,
    series,
    y_min,
    y_max,
    y_ticks,
    title,
    y_suffix="",
    annotation_suffix="",
    legend=True,
):
    x0, y0, x1, y1 = box
    draw.text((x0 + 30, y0 + 24), title, font=font(28, True), fill=TEXT)
    plot = (x0 + 90, y0 + 95, x1 - 35, y1 - 125)
    px0, py0, px1, py1 = plot
    for tick in y_ticks:
        y = py1 - (tick - y_min) / (y_max - y_min) * (py1 - py0)
        draw.line((px0, y, px1, y), fill=GRID, width=2)
        right_text(draw, (px0 - 14, y), f"{tick:g}{y_suffix}", font(18), MUTED)
    draw.line((px0, py0, px0, py1), fill=MUTED, width=2)
    draw.line((px0, py1, px1, py1), fill=MUTED, width=2)
    x_positions = [
        px0 + index * (px1 - px0) / max(1, len(x_labels) - 1)
        for index in range(len(x_labels))
    ]
    for x, label in zip(x_positions, x_labels):
        centered_text(draw, (x, py1 + 34), label, font(18), MUTED)

    for item in series:
        values = item["values"]
        color = item["color"]
        points = [
            (
                x,
                py1 - (value - y_min) / (y_max - y_min) * (py1 - py0),
            )
            for x, value in zip(x_positions, values)
        ]
        if item.get("reference"):
            y = points[0][1]
            for dash_x in range(int(px0), int(px1), 22):
                draw.line((dash_x, y, min(dash_x + 12, px1), y), fill=color, width=3)
            label_x = px0 + 18 if item.get("reference_label_side") == "left" else px1 - 190
            draw.text(
                (label_x, y - 34),
                f"{item['name']} {values[0]:.2f}{annotation_suffix}",
                font=font(18),
                fill=color,
            )
            continue
        draw.line(points, fill=color, width=5, joint="curve")
        for index, ((x, y), value) in enumerate(zip(points, values)):
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=PANEL, outline=color, width=4)
            label_y = y - 34 if index % 2 == 0 else y + 16
            centered_text(
                draw,
                (x, label_y),
                f"{value:.2f}{annotation_suffix}",
                font(17, True),
                color,
            )
    if legend:
        legend_x = x0 + 30
        legend_y = y1 - 34
        for item in series:
            draw.line((legend_x, legend_y, legend_x + 34, legend_y), fill=item["color"], width=5)
            draw.text((legend_x + 44, legend_y - 13), item["name"], font=font(18), fill=TEXT)
            legend_x += 44 + text_size(draw, item["name"], font(18))[0] + 38


def render_math_chart(labels, math, output):
    image, draw = new_canvas(
        title="数学工具调用能力",
        subtitle="500道 ToolUse 数学题｜必须正确调用 calculate_math 且最终答案正确",
    )
    panel(draw, (55, 155, 880, 985))
    panel(draw, (920, 155, 1745, 985))
    curve_labels = ["Base", "40", "80", "120", "160", "200"]
    curve_keys = ["base", "opd40", "opd80", "opd120", "opd160", "opd200"]
    accuracy = [math[key]["accuracy"] * 100 for key in curve_keys]
    tool_call = [math[key]["used_math_tool"] / math[key]["total"] * 100 for key in curve_keys]
    line_chart(
        draw,
        (55, 155, 880, 985),
        curve_labels,
        [
            {"name": "Base → OPD", "values": accuracy, "color": BLUE},
            {
                "name": "Agent参考",
                "values": [math["agent"]["accuracy"] * 100] * len(curve_keys),
                "color": ORANGE,
                "reference": True,
                "reference_label_side": "left",
            },
        ],
        40,
        68,
        [40, 45, 50, 55, 60, 65],
        "规则准确率",
        "%",
        "%",
    )
    line_chart(
        draw,
        (920, 155, 1745, 985),
        curve_labels,
        [
            {"name": "Base → OPD", "values": tool_call, "color": TEAL},
            {
                "name": "Agent参考",
                "values": [
                    math["agent"]["used_math_tool"] / math["agent"]["total"] * 100
                ]
                * len(curve_keys),
                "color": ORANGE,
                "reference": True,
                "reference_label_side": "left",
            },
        ],
        82,
        101,
        [84, 88, 92, 96, 100],
        "calculate_math 调用率",
        "%",
        "%",
    )
    image.save(output, quality=95)


def render_general_chart(labels, qa, output):
    image, draw = new_canvas(
        title="通用回答能力",
        subtitle="200道开放式通用问答｜DeepSeek Judge｜非代码题归一化总分",
    )
    panel(draw, (55, 155, 1745, 985))
    curve_keys = ["base", "opd40", "opd80", "opd120", "opd160", "opd200"]
    curve_labels = ["Base", "40", "80", "120", "160", "200"]
    values = [qa[key]["normalized_total_avg_100"] for key in curve_keys]
    line_chart(
        draw,
        (55, 155, 1745, 985),
        curve_labels,
        [
            {"name": "Base → OPD", "values": values, "color": PURPLE},
            {
                "name": "Agent参考",
                "values": [qa["agent"]["normalized_total_avg_100"]] * len(curve_keys),
                "color": RED,
                "reference": True,
            },
        ],
        0,
        18,
        [0, 3, 6, 9, 12, 15, 18],
        "归一化总分 / 100",
        "",
        "",
    )
    image.save(output, quality=95)


def render_tradeoff(labels, math, qa, output):
    image, draw = new_canvas(
        title="能力权衡：数学工具调用 vs 通用回答",
        subtitle="右上角代表两类能力同时更好；虚线连接 Base 与不同 OPD 训练步数",
    )
    panel(draw, (55, 155, 1745, 985))
    x0, y0, x1, y1 = (170, 245, 1660, 875)
    x_min, x_max = 0, 17
    y_min, y_max = 42, 66
    for tick in [0, 3, 6, 9, 12, 15]:
        x = x0 + (tick - x_min) / (x_max - x_min) * (x1 - x0)
        draw.line((x, y0, x, y1), fill=GRID, width=2)
        centered_text(draw, (x, y1 + 34), str(tick), font(18), MUTED)
    for tick in [45, 50, 55, 60, 65]:
        y = y1 - (tick - y_min) / (y_max - y_min) * (y1 - y0)
        draw.line((x0, y, x1, y), fill=GRID, width=2)
        right_text(draw, (x0 - 16, y), f"{tick}%", font(18), MUTED)
    draw.line((x0, y0, x0, y1), fill=MUTED, width=2)
    draw.line((x0, y1, x1, y1), fill=MUTED, width=2)
    centered_text(draw, ((x0 + x1) / 2, y1 + 82), "通用回答归一化总分 / 100", font(22), TEXT)
    draw.text((x0, y0 - 45), "数学规则准确率 ↑", font=font(22), fill=TEXT)

    def coord(key):
        xv = qa[key]["normalized_total_avg_100"]
        yv = math[key]["accuracy"] * 100
        return (
            x0 + (xv - x_min) / (x_max - x_min) * (x1 - x0),
            y1 - (yv - y_min) / (y_max - y_min) * (y1 - y0),
        )

    curve_keys = ["base", "opd40", "opd80", "opd120", "opd160", "opd200"]
    curve_points = [coord(key) for key in curve_keys]
    draw.line(curve_points, fill=BLUE, width=4, joint="curve")
    display = {
        "base": "Base",
        "opd40": "OPD40",
        "opd80": "OPD80",
        "opd120": "OPD120",
        "opd160": "OPD160",
        "opd200": "OPD200",
        "agent": "Agent",
    }
    offsets = {
        "base": (12, 8),
        "opd40": (-52, -42),
        "opd80": (12, -18),
        "opd120": (-62, 20),
        "opd160": (15, 12),
        "opd200": (15, 18),
        "agent": (15, -20),
    }
    for key in labels:
        x, y = coord(key)
        color = RED if key == "agent" else BLUE
        radius = 11 if key in {"base", "agent", "opd40"} else 8
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=PANEL, outline=color, width=5)
        dx, dy = offsets[key]
        draw.text((x + dx, y + dy), display[key], font=font(19, True), fill=color)
    image.save(output, quality=95)


def heat_color(value, maximum=30):
    ratio = max(0.0, min(1.0, value / maximum))
    start = (238, 242, 248)
    end = (37, 99, 235)
    return tuple(round(start[i] + ratio * (end[i] - start[i])) for i in range(3))


def render_heatmap(labels, qa, output):
    image, draw = new_canvas(
        title="通用回答分类表现",
        subtitle="单元格为各类别归一化总分 / 100；颜色越深，得分越高",
    )
    panel(draw, (55, 155, 1745, 985))
    categories = list(qa["base"]["by_category"])
    left, top = 255, 260
    cell_w, cell_h = 225, 82
    for col, category in enumerate(categories):
        centered_text(
            draw,
            (left + col * cell_w + cell_w / 2, top - 48),
            category,
            font(19, True),
            TEXT,
        )
    display = {
        "base": "Base",
        "opd40": "OPD40",
        "opd80": "OPD80",
        "opd120": "OPD120",
        "opd160": "OPD160",
        "opd200": "OPD200",
        "agent": "Agent",
    }
    for row, key in enumerate(labels):
        y = top + row * cell_h
        right_text(draw, (left - 25, y + cell_h / 2), display[key], font(21, True), TEXT)
        for col, category in enumerate(categories):
            value = qa[key]["by_category"][category]["normalized_total_avg_100"]
            x = left + col * cell_w
            fill = heat_color(value)
            draw.rounded_rectangle(
                (x + 5, y + 5, x + cell_w - 5, y + cell_h - 5),
                radius=12,
                fill=fill,
            )
            ratio = value / 30
            value_color = PANEL if ratio > 0.55 else TEXT
            centered_text(draw, (x + cell_w / 2, y + cell_h / 2), f"{value:.2f}", font(22, True), value_color)
    image.save(output, quality=95)


def render_dashboard(math_path, general_path, tradeoff_path, heatmap_path, output):
    image = Image.new("RGB", (2100, 1370), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 38), "MiniMind Agent OPD 评测总览", font=font(46, True), fill=TEXT)
    draw.text(
        (70, 98),
        "数学 ToolUse：500题规则准确率｜通用回答：200题 DeepSeek Judge",
        font=font(24),
        fill=MUTED,
    )
    sources = [
        Image.open(math_path).convert("RGB").crop((35, 135, 1765, 1015)),
        Image.open(general_path).convert("RGB").crop((35, 135, 1765, 1015)),
        Image.open(tradeoff_path).convert("RGB").crop((35, 135, 1765, 1015)),
        Image.open(heatmap_path).convert("RGB").crop((35, 135, 1765, 1015)),
    ]
    positions = [(55, 155), (1075, 155), (55, 745), (1075, 745)]
    panel_size = (970, 550)
    for source, (x, y) in zip(sources, positions):
        source.thumbnail(panel_size, Image.Resampling.LANCZOS)
        tile = Image.new("RGB", panel_size, PANEL)
        tile.paste(source, ((panel_size[0] - source.width) // 2, (panel_size[1] - source.height) // 2))
        image.paste(tile, (x, y))
        draw.rounded_rectangle((x, y, x + panel_size[0], y + panel_size[1]), radius=20, outline=GRID, width=2)
    image.save(output, quality=95)


def parse_args():
    parser = argparse.ArgumentParser(description="绘制Agent OPD数学与通用能力评测图")
    parser.add_argument("--math_curve", default=str(DEFAULT_MATH_CURVE))
    parser.add_argument("--math_three_way", default=str(DEFAULT_MATH_THREE_WAY))
    parser.add_argument("--general_results", default=str(DEFAULT_GENERAL))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels, math, qa = load_results(
        args.math_curve, args.math_three_way, args.general_results
    )
    math_path = output_dir / "math_tooluse_performance.png"
    general_path = output_dir / "general_qa_performance.png"
    tradeoff_path = output_dir / "capability_tradeoff.png"
    heatmap_path = output_dir / "general_qa_category_heatmap.png"
    dashboard_path = output_dir / "agent_opd_evaluation_dashboard.png"
    render_math_chart(labels, math, math_path)
    render_general_chart(labels, qa, general_path)
    render_tradeoff(labels, math, qa, tradeoff_path)
    render_heatmap(labels, qa, heatmap_path)
    render_dashboard(math_path, general_path, tradeoff_path, heatmap_path, dashboard_path)
    print(json.dumps({
        "output_dir": str(output_dir),
        "files": [str(path) for path in (dashboard_path, math_path, general_path, tradeoff_path, heatmap_path)],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
