#!/usr/bin/env python3
"""Convert .drawio flowcharts to PNG previews using matplotlib.

Usage:
    python scripts/drawio_to_png.py docs/assets/flowcharts/*.drawio
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def parse_style(style: str) -> dict:
    result = {}
    if not style:
        return result
    for part in style.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
        else:
            result[part] = True
    return result


def set_chinese_font():
    """Use a font that supports CJK characters on macOS."""
    preferred = ["Arial Unicode MS", "Noto Sans CJK SC", "PingFang SC", "Heiti SC"]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return plt.rcParams["font.family"][0] if plt.rcParams["font.family"] else "sans-serif"


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    if len(hex_color) == 6:
        return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (0.9, 0.9, 0.9)


def parse_drawio(path: Path) -> tuple:
    tree = ET.parse(path)
    root = tree.getroot()

    # Find mxGraphModel -> root
    graph_model = root.find(".//{*}mxGraphModel")
    if graph_model is None:
        raise ValueError(f"No mxGraphModel found in {path}")

    root_cell = graph_model.find(".//{*}root")
    if root_cell is None:
        raise ValueError(f"No root found in {path}")

    vertices = {}
    edges = []

    for cell in root_cell.findall("{*}mxCell"):
        cell_id = cell.get("id")
        if cell_id in ("0", "1"):
            continue

        value = cell.get("value", "")
        value = value.replace("&xa;", "\n").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

        geom = cell.find("{*}mxGeometry")
        if geom is None:
            continue

        if cell.get("vertex") == "1":
            x = float(geom.get("x", 0))
            y = float(geom.get("y", 0))
            w = float(geom.get("width", 100))
            h = float(geom.get("height", 50))
            vertices[cell_id] = {
                "id": cell_id,
                "value": value,
                "style": parse_style(cell.get("style", "")),
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            }
        elif cell.get("edge") == "1":
            edges.append(
                {
                    "id": cell_id,
                    "source": cell.get("source"),
                    "target": cell.get("target"),
                    "value": value,
                    "style": parse_style(cell.get("style", "")),
                }
            )

    return vertices, edges


def get_shape(style: dict) -> str:
    if "rhombus" in style:
        return "diamond"
    if "ellipse" in style:
        return "ellipse"
    if "shape" in style:
        return style["shape"]
    return "rect"


def draw_node(ax, node):
    x, y, w, h = node["x"], node["y"], node["width"], node["height"]
    style = node["style"]
    shape = get_shape(style)
    fill = hex_to_rgb(style.get("fillColor", "#ffffff"))
    stroke = hex_to_rgb(style.get("strokeColor", "#333333"))

    if shape == "diamond":
        cx, cy = x + w / 2, y + h / 2
        poly = plt.Polygon(
            [(cx, y), (x + w, cy), (cx, y + h), (x, cy)],
            closed=True,
            facecolor=fill,
            edgecolor=stroke,
            linewidth=1.5,
        )
        ax.add_patch(poly)
    elif shape == "ellipse":
        ellipse = mpatches.Ellipse(
            (x + w / 2, y + h / 2), w, h, facecolor=fill, edgecolor=stroke, linewidth=1.5
        )
        ax.add_patch(ellipse)
    else:
        rounded = "rounded" in style
        if rounded:
            box = mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=8",
                facecolor=fill,
                edgecolor=stroke,
                linewidth=1.5,
            )
        else:
            box = mpatches.Rectangle((x, y), w, h, facecolor=fill, edgecolor=stroke, linewidth=1.5)
        ax.add_patch(box)

    # Text
    if node["value"]:
        ax.text(
            x + w / 2,
            y + h / 2,
            node["value"],
            ha="center",
            va="center",
            fontsize=7,
            wrap=True,
            color="#333333",
            linespacing=1.1,
        )


def get_anchor(node, target_x, target_y):
    """Return intersection point of node rectangle toward target."""
    cx = node["x"] + node["width"] / 2
    cy = node["y"] + node["height"] / 2
    dx = target_x - cx
    dy = target_y - cy

    if dx == 0 and dy == 0:
        return cx, cy

    half_w = node["width"] / 2
    half_h = node["height"] / 2

    ux = half_w / abs(dx) if dx != 0 else float("inf")
    uy = half_h / abs(dy) if dy != 0 else float("inf")
    u = min(ux, uy)

    return cx + dx * u, cy + dy * u


def draw_edge(ax, edge, vertices):
    src = vertices.get(edge["source"])
    tgt = vertices.get(edge["target"])
    if src is None or tgt is None:
        return

    src_cx = src["x"] + src["width"] / 2
    src_cy = src["y"] + src["height"] / 2
    tgt_cx = tgt["x"] + tgt["width"] / 2
    tgt_cy = tgt["y"] + tgt["height"] / 2

    x1, y1 = get_anchor(src, tgt_cx, tgt_cy)
    x2, y2 = get_anchor(tgt, src_cx, src_cy)

    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="#666666", lw=1.2),
        zorder=1,
    )

    if edge["value"]:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mid_x,
            mid_y,
            edge["value"],
            ha="center",
            va="bottom",
            fontsize=6,
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
        )


def render_drawio(input_path: Path, output_path: Path, dpi: int = 150):
    vertices, edges = parse_drawio(input_path)

    if not vertices:
        raise ValueError(f"No vertices found in {input_path}")

    xs = [v["x"] for v in vertices.values()] + [v["x"] + v["width"] for v in vertices.values()]
    ys = [v["y"] for v in vertices.values()] + [v["y"] + v["height"] for v in vertices.values()]

    margin = 60
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin

    width = max(6, (max_x - min_x) / 120)
    height = max(6, (max_y - min_y) / 120)

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(max_y, min_y)  # y grows downward in drawio
    ax.set_aspect("equal")
    ax.axis("off")

    for edge in edges:
        draw_edge(ax, edge, vertices)

    for node in vertices.values():
        draw_node(ax, node)

    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    set_chinese_font()
    parser = argparse.ArgumentParser(description="Convert .drawio files to PNG previews")
    parser.add_argument("files", nargs="+", type=Path, help="Input .drawio files")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/flowcharts/previews"))
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for file in args.files:
        if not file.exists():
            print(f"Skipping missing file: {file}")
            continue
        output = args.output_dir / f"{file.stem}.png"
        try:
            render_drawio(file, output, args.dpi)
        except Exception as e:
            print(f"Failed to render {file}: {e}")


if __name__ == "__main__":
    main()
