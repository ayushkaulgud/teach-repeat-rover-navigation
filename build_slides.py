"""Build the class-style project update as an editable PowerPoint deck.

The 10 × 5.625 inch layout, cream canvas, teal accents, and plain typography
follow the team's original Google Slides presentation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from src import robot_motion as sim


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
METRICS = json.loads((RESULTS / "metrics.json").read_text())
METHODS = METRICS["methods"]
OUT = ROOT / "docs" / "Teach_and_Repeat_Project_Update_Oct_7_2026.pptx"

CREAM = "FBF9F0"
TEAL = "27A69A"
DEEP_TEAL = "147B73"
BLACK = "101010"
GREY = "555555"
LIGHT_GREY = "DDDCD3"
PALE_TEAL = "DFF0ED"
RED = "C5554C"
GOLD = "B7822F"
BLUE = "337DA5"


def color(code):
    return RGBColor.from_string(code)


def box(slide, x, y, w, h, fill, line=None, round_corner=False):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if round_corner else MSO_SHAPE.RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color(fill)
    if line:
        shape.line.color.rgb = color(line)
    else:
        shape.line.fill.background()
    return shape


def label(slide, value, x, y, w, h, pt=17, ink=BLACK, bold=False,
          align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE, font="Arial"):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.01)
    tf.margin_top = tf.margin_bottom = Inches(0.01)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.text = value
    p.alignment = align
    p.font.name = font
    p.font.size = Pt(pt)
    p.font.bold = bold
    p.font.color.rgb = color(ink)
    return shape


def slide(title=None):
    page = deck.slides.add_slide(deck.slide_layouts[6])
    page.background.fill.solid()
    page.background.fill.fore_color.rgb = color(CREAM)
    box(page, 0, 5.57, 10, 0.055, TEAL)
    if title:
        label(page, title, 0.45, 0.22, 9.1, 0.55, 28, BLACK)
    return page


def line(slide, x1, y1, x2, y2, ink=LIGHT_GREY, width=1):
    shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                       Inches(x2), Inches(y2))
    shape.line.color.rgb = color(ink)
    shape.line.width = Pt(width)
    return shape


def figure(slide, path, x, y, w, h):
    # All generated plots use the intended slide aspect ratio.
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def plot_map(ax):
    ax.set_facecolor("#fbf9f0")
    for ox, oy, radius in sim.OBSTACLES:
        ax.add_patch(plt.Circle((ox, oy), radius, color="#bb6d64", alpha=0.80, zorder=2))
        ax.add_patch(plt.Circle((ox, oy), radius + sim.ROVER_RADIUS + sim.SAFETY_MARGIN,
                                edgecolor="#bd786f", facecolor="none", lw=0.9, ls="--"))
    ax.plot(sim.DEMO[:, 0], sim.DEMO[:, 1], "--", color="#3e6661", lw=1.35, label="Taught route")
    ax.set(xlim=(0, 10), ylim=(0, 7))
    ax.set_aspect("equal")
    ax.grid(color="#9daaa4", alpha=0.28, lw=0.6)
    ax.tick_params(labelsize=8, colors="#444444")
    for spine in ax.spines.values():
        spine.set_color("#888888")


def create_figures():
    RESULTS.mkdir(exist_ok=True)
    start = (2.2, 3.35, math.radians(-35))
    styles = [
        ("Waypoints", sim.waypoint_path, "#c5554c"),
        ("Local reconnection", sim.reconnect_path, "#148a7d"),
        ("Classical DMP", lambda p: sim.dmp_path(p, False), "#337da5"),
        ("Obstacle-aware DMP", lambda p: sim.dmp_path(p, True), "#b7822f"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.3), constrained_layout=True, facecolor="#fbf9f0")
    for ax, (name, planner, ink) in zip(axes.flat, styles):
        path = planner(start)
        accepted = path is not None and sim.inside(path) and float(np.min(sim.clearance(path))) >= sim.SAFETY_MARGIN
        outcome, trace = sim.execute(path, start) if accepted else ("rejected", np.array([start[:2]]))
        plot_map(ax)
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], color=ink, lw=2.0)
        if accepted:
            ax.plot(trace[:, 0], trace[:, 1], color="#232323", lw=0.9, alpha=0.8)
        ax.scatter([start[0]], [start[1]], marker="*", s=72, color=ink, zorder=5)
        ax.set_title(f"{name}  —  {outcome.replace('_', ' ').title()}", fontsize=12, loc="left", pad=6)
    fig.savefig(RESULTS / "four_methods_class_style.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained", facecolor="#fbf9f0")
    plot_map(ax)
    path = sim.reconnect_path(start)
    ax.plot(path[:, 0], path[:, 1], color="#147e74", lw=2.4, label="Local reconnection")
    ax.scatter([start[0]], [start[1]], marker="*", s=100, color="#147e74", zorder=5)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.savefig(RESULTS / "route_example_class_style.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    icp = sim.icp_demo()
    fig, ax = plt.subplots(figsize=(7.3, 4.5), layout="constrained", facecolor="#fbf9f0")
    ax.set_facecolor("#fbf9f0")
    ax.scatter(icp["map"][:, 0], icp["map"][:, 1], s=20, color="#222222", label="Recorded map")
    ax.scatter(icp["scan"][:, 0], icp["scan"][:, 1], s=24, marker="x", color="#c5554c", label="Current scan")
    ax.scatter(icp["corrected"][:, 0], icp["corrected"][:, 1], s=13, color="#27a69a", label="After ICP")
    ax.set(xlim=(0, 10), ylim=(0, 7), xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal")
    ax.grid(color="#9daaa4", alpha=0.25)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.savefig(RESULTS / "icp_class_style.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


create_figures()
deck = Presentation()
deck.slide_width = Inches(10)
deck.slide_height = Inches(5.625)

# 1: The same two-tone cover as the original class deck.
page = deck.slides.add_slide(deck.slide_layouts[6])
page.background.fill.solid()
page.background.fill.fore_color.rgb = color(BLACK)
box(page, 0, 0, 10, 1.91, TEAL)
label(page, "Teach & Repeat Rover Navigation", 0.43, 0.63, 9.2, 0.72, 31, CREAM)
label(page, "Project update  |  Initial simulation results", 0.43, 2.33, 9.1, 0.63, 27, CREAM)
line(page, 0.48, 3.32, 1.18, 3.32, CREAM, 2.2)
label(page, "Ayush Kaulgud  ·  Irenov Joel  ·  Tapisar Singh", 0.48, 3.71, 8.8, 0.44, 19, CREAM, True)
label(page, "October 7, 2026", 0.48, 4.27, 6.0, 0.31, 15, CREAM)

# 2: Context with a concrete image instead of an abstract card.
page = slide("Where the project stands")
label(page, "The task", 0.48, 1.04, 3.0, 0.36, 18, BLACK, True)
label(page, "Start the rover away from a taught route and recover to the same goal without hitting an obstacle.",
      0.48, 1.45, 4.35, 1.18, 18)
label(page, "What works today", 0.48, 2.79, 3.8, 0.36, 18, BLACK, True)
label(page, "• One safe demonstration\n• Four recovery planners\n• Path and runtime safety checks\n• A separate ICP alignment demo",
      0.51, 3.14, 4.3, 1.69, 16)
figure(page, RESULTS / "route_example_class_style.png", 5.04, 1.17, 4.48, 3.86)
label(page, "A* reconnection from one displaced start. Dashed line is the taught route.",
      5.18, 5.06, 4.35, 0.27, 10, GREY)

# 3: Workflow follows the original deck's teal process boxes.
page = slide("From demonstration to evaluation")
steps = [("Teach route", 0.40), ("Build map", 2.42), ("Estimate pose", 4.44),
         ("Generate path", 6.46), ("Check + run", 8.48)]
for title, x in steps:
    box(page, x, 1.38, 1.38, 0.95, TEAL, DEEP_TEAL, True)
    label(page, title, x + 0.08, 1.59, 1.22, 0.52, 15, BLACK, align=PP_ALIGN.CENTER)
for x in (1.80, 3.82, 5.84, 7.86):
    line(page, x, 1.85, x + 0.52, 1.85, DEEP_TEAL, 1.5)
    label(page, "›", x + 0.29, 1.62, 0.20, 0.42, 20, DEEP_TEAL)
label(page, "Four candidate paths", 0.52, 2.83, 4.0, 0.41, 19, BLACK, True)
label(page, "Direct waypoints  ·  Local A* reconnection\nClassical DMP  ·  Obstacle-aware DMP",
      0.52, 3.28, 8.85, 0.95, 19)
label(page, "Current navigation runs use exact pose; ICP is demonstrated separately on a synthetic scan.",
      0.52, 4.59, 8.92, 0.43, 13, GREY)

# 4: Four methods as actual trajectories, with no decorative containers.
page = slide("One offset start, four outcomes")
figure(page, RESULTS / "four_methods_class_style.png", 0.35, 0.96, 9.32, 4.15)
label(page, "Start: (2.2, 3.35) m, −35°. Red circles are obstacles; dotted rings include rover radius + safety margin.",
      0.49, 5.19, 9.12, 0.26, 10.5, GREY)

# 5: A simple editable bar chart. Every bar is grounded in metrics.json.
page = slide("Completion across 60 starts per method")
names = [("Direct waypoints", "Waypoints"), ("Local reconnection", "Local reconnection"),
         ("Classical DMP", "Classical DMP"), ("Obstacle-aware DMP", "Obstacle-aware DMP")]
for i, (display, key) in enumerate(names):
    y = 1.27 + i * 0.82
    value = METHODS[key]["completion_pct"]
    label(page, display, 0.48, y, 2.60, 0.38, 17)
    box(page, 3.08, y + 0.07, 4.95, 0.26, "E2E1D9")
    box(page, 3.08, y + 0.07, 4.95 * value / 100, 0.26, TEAL)
    label(page, f"{value:.1f}%", 8.19, y - 0.01, 1.02, 0.39, 19, BLACK, True)
label(page, "A* local reconnection completed the most starts in this map.",
      0.48, 4.77, 8.95, 0.38, 17, BLACK, True)
label(page, "Across all 240 trials: 0 collisions; unsafe plans were rejected or stopped.",
      0.48, 5.15, 9.0, 0.22, 10.5, GREY)

# 6: The complete set of metrics from the original slideshow.
page = slide("The other measurements matter too")
headers = ["Method", "Rejected", "Stopped", "Min. clear.", "Goal err.", "Plan time"]
xs = [0.49, 3.02, 4.25, 5.43, 6.87, 8.17]
ws = [2.46, 1.16, 1.12, 1.37, 1.23, 1.35]
box(page, 0.41, 1.09, 9.15, 0.46, PALE_TEAL)
for x, w, h in zip(xs, ws, headers):
    label(page, h, x, 1.18, w, 0.24, 12, BLACK, True)
for i, (display, key) in enumerate(names):
    y = 1.62 + i * 0.70
    m = METHODS[key]
    if i % 2:
        box(page, 0.41, y, 9.15, 0.65, "F1F1E9")
    vals = [display, f'{m["rejection_pct"]:.1f}%', f'{m["safety_stop_pct"]:.1f}%',
            f'{m["minimum_clearance_m"]:.3f} m', f'{m["median_goal_error_m"]:.3f} m',
            f'{m["median_planning_ms"]:.2f} ms']
    for x, w, value in zip(xs, ws, vals):
        label(page, value, x, y + 0.15, w, 0.34, 12.5)
    line(page, 0.41, y + 0.65, 9.56, y + 0.65, LIGHT_GREY, 0.5)
label(page, "All methods had 0% observed collision rate. Goal error is the median of completed runs.",
      0.49, 4.56, 9.0, 0.29, 12, GREY)
label(page, "Minimum clearance includes sampled safety stops; planning time varies by computer.",
      0.49, 4.90, 9.0, 0.29, 12, GREY)

# 7: ICP is shown clearly as a separate prototype.
page = slide("ICP scan alignment: separate prototype")
figure(page, RESULTS / "icp_class_style.png", 0.45, 1.05, 5.50, 4.08)
label(page, "55 synthetic terrain landmarks", 6.05, 1.32, 3.38, 0.44, 17, BLACK, True)
label(page, f'Alignment RMSE   {METRICS["icp"]["registration_rmse_m"]:.3f} m',
      6.05, 2.03, 3.49, 0.36, 17)
label(page, f'Translation error   {METRICS["icp"]["translation_error_m"]:.4f} m',
      6.05, 2.64, 3.49, 0.36, 17)
label(page, f'Rotation error   {METRICS["icp"]["rotation_error_deg"]:.3f}°',
      6.05, 3.25, 3.49, 0.36, 17)
label(page, "Navigation results on the previous slides still use exact rover pose.",
      6.05, 4.13, 3.28, 0.68, 13, GREY)

# 8: Conservative interpretation and credible next work.
page = slide("What we can say so far")
label(page, "1", 0.49, 1.29, 0.38, 0.48, 24, TEAL, True)
label(page, "Local A* reconnection is strongest in this single map: 76.7% completion.",
      0.93, 1.31, 8.52, 0.62, 17)
label(page, "2", 0.49, 2.19, 0.38, 0.48, 24, TEAL, True)
label(page, "Obstacle coupling raises DMP completion from 35.0% to 53.3%.",
      0.93, 2.21, 8.52, 0.62, 17)
label(page, "3", 0.49, 3.09, 0.38, 0.48, 24, TEAL, True)
label(page, "No collisions occurred with safety checks enabled, but some paths stopped early.",
      0.93, 3.11, 8.52, 0.62, 17)
line(page, 0.49, 4.11, 9.47, 4.11, TEAL, 1.2)
label(page, "Next: integrate ICP with noisy odometry; test more maps and pose offsets; tune controller tracking.",
      0.49, 4.43, 8.92, 0.69, 16, BLACK, True)

# 9: A plain appendix supports questions in class.
page = slide("Method notes and sources")
label(page, "Trial design", 0.49, 1.14, 3.8, 0.33, 17, BLACK, True)
label(page, "60 starts = 4 x positions × 5 y positions × 3 headings. Static 10 × 7 m map, two circular obstacles, 0.16 m rover radius, 0.12 m margin. Shared bounded controller.",
      0.49, 1.49, 8.95, 1.08, 15)
label(page, "Reproduce", 0.49, 2.79, 3.8, 0.33, 17, BLACK, True)
label(page, "python src/robot_motion.py     python build_slides.py",
      0.49, 3.15, 8.95, 0.43, 15, BLACK, font="Consolas")
label(page, "Background references", 0.49, 3.89, 5.4, 0.33, 17, BLACK, True)
label(page, "Ijspeert et al. (2013) on DMPs; Hoffmann et al. (2009) on obstacle coupling; Besl & McKay (1992) on ICP. Sources listed in the original class presentation.",
      0.49, 4.24, 8.95, 0.95, 13.5)

OUT.parent.mkdir(exist_ok=True)
deck.save(OUT)
print(OUT)
