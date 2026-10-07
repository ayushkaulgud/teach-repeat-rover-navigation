"""Build the project update deck from measured rover_demo.py outputs."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DATA = json.loads((RESULTS / "metrics.json").read_text())
METHODS = DATA["methods"]
OUT = ROOT / "docs" / "Teach_and_Repeat_Project_Update_Oct_7_2026.pptx"

BG = "0B1523"
PANEL = "14263B"
PANEL2 = "1A3149"
WHITE = "EFF5FA"
MUTED = "AFC1D1"
CYAN = "65B8FF"
GREEN = "54D2A0"
GOLD = "FFC56E"
RED = "ED6A5A"
LAV = "AD99E7"


def rgb(hexcode):
    return RGBColor.from_string(hexcode)


def rect(slide, x, y, w, h, fill=PANEL, line=None, radius=True):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.fill.background() if line is None else None
    if line:
        shape.line.color.rgb = rgb(line)
    return shape


def text(slide, content, x, y, w, h, size=20, color=WHITE, bold=False,
         align=PP_ALIGN.LEFT, font="Aptos", valign=MSO_ANCHOR.MIDDLE):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0.02)
    frame.margin_top = frame.margin_bottom = Inches(0.02)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.text = content
    paragraph.alignment = align
    paragraph.font.name = font
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = rgb(color)
    return shape


def image(slide, path, x, y, w, h):
    with Image.open(path) as im:
        iw, ih = im.size
    ratio = min(w / iw, h / ih)
    ww, hh = iw * ratio, ih * ratio
    slide.shapes.add_picture(str(path), Inches(x + (w - ww) / 2), Inches(y + (h - hh) / 2),
                             width=Inches(ww), height=Inches(hh))


def base(title=None, section=None):
    slide = pres.slides.add_slide(pres.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(BG)
    rect(slide, 0, 0, 0.10, 7.5, CYAN, radius=False)
    if section:
        text(slide, section.upper(), 0.53, 0.24, 4.2, 0.28, 10, CYAN, True)
    if title:
        text(slide, title, 0.53, 0.63, 12.2, 0.6, 27, WHITE, True)
    text(slide, "TEACH & REPEAT  /  PRELIMINARY SIMULATION", 0.55, 7.15, 8.3, 0.19, 8.5, MUTED)
    text(slide, f"{len(pres.slides):02d}", 12.42, 7.12, 0.4, 0.23, 9, MUTED, align=PP_ALIGN.RIGHT)
    return slide


def metric_card(slide, x, y, w, value, label, color=CYAN):
    rect(slide, x, y, w, 1.25, PANEL)
    text(slide, value, x + 0.20, y + 0.16, w - 0.4, 0.49, 28, color, True)
    text(slide, label, x + 0.20, y + 0.71, w - 0.4, 0.31, 12, MUTED)


pres = Presentation()
pres.slide_width = Inches(13.333)
pres.slide_height = Inches(7.5)

# 1 — cover
slide = base(section="Project update  |  October 7, 2026")
rect(slide, 0.55, 1.03, 0.12, 0.71, GOLD, radius=False)
text(slide, "Teach & Repeat\nRover Navigation", 0.82, 1.15, 5.5, 1.72, 34, WHITE, True)
text(slide, "Adapting one demonstrated route from a displaced start", 0.82, 3.10, 5.05, 0.94, 22, MUTED)
rect(slide, 0.82, 4.44, 4.8, 0.58, PANEL2)
text(slide, "FOUR METHODS  •  60 START POSES EACH", 1.04, 4.53, 4.35, 0.34, 13, GREEN, True)
text(slide, "Ayush Kaulgud  ·  Irenov Joel  ·  Tapisar Singh", 0.82, 5.55, 5.35, 0.45, 15, WHITE)
with Image.open(RESULTS / "four_methods.png") as im:
    im.crop((im.width // 2, im.height // 2, im.width, im.height)).save(RESULTS / "cover_demo.png")
rect(slide, 6.75, 1.15, 5.85, 4.95, PANEL)
image(slide, RESULTS / "cover_demo.png", 6.93, 1.38, 5.50, 4.50)
text(slide, "Obstacle-aware DMP  /  successful example", 7.15, 6.19, 5.35, 0.35, 12, GOLD)

# 2 — purpose and pipeline
slide = base("The question and our current prototype", "01  /  Problem")
rect(slide, 0.55, 1.43, 5.08, 4.99, PANEL)
text(slide, "Can the rover safely rejoin a taught route when it starts elsewhere?",
     0.89, 1.81, 4.38, 1.65, 25, WHITE, True)
text(slide, "Same route. Same map. Same feedback controller. Only the recovery trajectory changes.",
     0.89, 3.94, 4.28, 1.16, 17, MUTED)
rect(slide, 0.89, 5.54, 4.16, 0.52, PANEL2)
text(slide, "INITIAL SIMULATION  /  2-D STATIC MAP", 1.10, 5.62, 3.79, 0.30, 11, CYAN, True)
steps = [("01", "Teach", "Record a safe reference route"),
         ("02", "ICP prototype", "Synthetic registration, separate from trials"),
         ("03", "Recover", "Generate one of four paths"),
         ("04", "Check", "Reject unsafe plans; monitor margin"),
         ("05", "Repeat", "Follow with a shared controller")]
for i, (num, label, detail) in enumerate(steps):
    y = 1.45 + i * 0.99
    rect(slide, 6.02, y, 6.62, 0.82, PANEL)
    text(slide, num, 6.27, y + 0.15, 0.57, 0.47, 18, CYAN, True)
    text(slide, label, 6.93, y + 0.09, 1.74, 0.31, 16, WHITE, True)
    text(slide, detail, 8.51, y + 0.09, 3.77, 0.52, 13, MUTED)

# 3 — four planners
slide = base("Four recovery methods, one test harness", "02  /  Algorithms")
cards = [
    ("01", "Direct waypoints", "Connect to the nearest forward taught point, then follow stored waypoints.", RED),
    ("02", "Local reconnection", "Use grid A* to find a collision-free merge onto the demonstrated route.", GREEN),
    ("03", "Classical DMP", "Fit a discrete DMP to the demonstration; regenerate from the new start.", CYAN),
    ("04", "Obstacle-aware DMP", "Add a repulsive obstacle coupling term during DMP rollout.", GOLD),
]
for i, (num, name, detail, color) in enumerate(cards):
    x = 0.55 + (i % 2) * 6.16
    y = 1.45 + (i // 2) * 2.43
    rect(slide, x, y, 5.78, 2.12, PANEL)
    rect(slide, x, y, 0.08, 2.12, color, radius=False)
    text(slide, num, x + 0.28, y + 0.20, 0.64, 0.34, 14, color, True)
    text(slide, name, x + 0.90, y + 0.17, 4.60, 0.40, 21, WHITE, True)
    text(slide, detail, x + 0.30, y + 0.80, 5.12, 0.85, 16, MUTED)
rect(slide, 0.55, 6.39, 12.00, 0.39, PANEL2)
text(slide, "ICP is the shared localization component; it is not a fifth trajectory planner.",
     0.78, 6.45, 11.45, 0.24, 12, CYAN)

# 4 — actual trajectories
slide = base("One displaced start, four visibly different outcomes", "03  /  Demonstration")
rect(slide, 0.55, 1.33, 8.51, 5.56, PANEL)
image(slide, RESULTS / "four_methods.png", 0.70, 1.49, 8.21, 5.23)
rect(slide, 9.31, 1.33, 3.25, 5.56, PANEL)
text(slide, "EXAMPLE START", 9.61, 1.68, 2.60, 0.29, 11, CYAN, True)
text(slide, "(2.2, 3.35) m\n−35° heading", 9.61, 2.12, 2.55, 1.06, 22, WHITE, True)
text(slide, "Local reconnection and obstacle-aware DMP completed this run.",
     9.61, 3.60, 2.50, 1.20, 17, MUTED)
text(slide, "Dashed white = taught route\nColored = planned path\nCream = rover trace\nRed rings = safety buffer",
     9.61, 5.03, 2.65, 1.28, 12, MUTED)

# 5 — primary outcomes
slide = base("Initial outcome across 60 starts per method", "04  /  Results")
rect(slide, 0.55, 1.38, 8.62, 4.99, PANEL)
image(slide, RESULTS / "outcomes.png", 0.76, 1.71, 8.20, 4.35)
metric_card(slide, 9.42, 1.38, 3.15, f'{METHODS["Local reconnection"]["completion_pct"]:.1f}%',
            "Highest completion: A*", GREEN)
metric_card(slide, 9.42, 2.81, 3.15,
            f'+{METHODS["Obstacle-aware DMP"]["completion_pct"] - METHODS["Classical DMP"]["completion_pct"]:.1f} pp',
            "Coupling vs. classical DMP", GOLD)
metric_card(slide, 9.42, 4.24, 3.15, "0%", "Observed collision rate", CYAN)
text(slide, "Safety stops are counted separately from path rejections.",
     0.75, 6.53, 10.80, 0.29, 12, MUTED)

# 6 — full metrics
slide = base("Metrics from the original project plan", "05  /  Measurements")
headers = ["METHOD", "COMPLETE", "REJECT", "STOP", "MIN CLR", "GOAL ERR", "PLAN TIME"]
widths = [2.40, 1.35, 1.13, 1.03, 1.52, 1.59, 1.70]
xs = [0.70]
for width in widths[:-1]:
    xs.append(xs[-1] + width)
rect(slide, 0.56, 1.45, 12.0, 0.72, PANEL2)
for x, width, header in zip(xs, widths, headers):
    text(slide, header, x, 1.63, width - 0.07, 0.28, 11, CYAN, True)
for idx, (name, values) in enumerate(METHODS.items()):
    y = 2.29 + idx * 0.93
    rect(slide, 0.56, y, 12.0, 0.77, PANEL if idx % 2 == 0 else "122237")
    fields = [name, f'{values["completion_pct"]:.1f}%', f'{values["rejection_pct"]:.1f}%',
              f'{values["safety_stop_pct"]:.1f}%', f'{values["minimum_clearance_m"]:.3f} m',
              f'{values["median_goal_error_m"]:.3f} m' if values["median_goal_error_m"] is not None else "—",
              f'{values["median_planning_ms"]:.2f} ms']
    for j, (x, width, value) in enumerate(zip(xs, widths, fields)):
        text(slide, value, x, y + 0.17, width - 0.08, 0.37, 13 if j == 0 else 12,
             WHITE if j != 1 else GREEN, j == 0)
text(slide, "Minimum clearance is measured on executed traces. Goal error is the median for completed runs.",
     0.76, 6.19, 11.80, 0.30, 12, MUTED)
text(slide, "Planning time depends on this laptop and is descriptive, not a hardware benchmark.",
     0.76, 6.55, 11.80, 0.30, 12, MUTED)

# 7 — ICP localization demonstration
slide = base("ICP localization proof of concept", "06  /  Localization")
rect(slide, 0.55, 1.34, 8.09, 5.45, PANEL)
image(slide, RESULTS / "icp_registration.png", 0.76, 1.53, 7.68, 5.03)
metric_card(slide, 8.90, 1.34, 3.66, f'{DATA["icp"]["registration_rmse_m"]:.3f} m',
            "Registration RMSE", GREEN)
metric_card(slide, 8.90, 2.77, 3.66, f'{DATA["icp"]["translation_error_m"]:.4f} m',
            "Translation error", CYAN)
metric_card(slide, 8.90, 4.20, 3.66, f'{DATA["icp"]["rotation_error_deg"]:.3f}°',
            "Rotation error", GOLD)
text(slide, "Synthetic 55-point scan; ICP is demonstrated separately from navigation trials.",
     9.10, 5.80, 3.12, 0.76, 12, MUTED)

# 8 — interpretation and work ahead
slide = base("What we learned and what comes next", "07  /  Update")
rect(slide, 0.55, 1.46, 5.78, 4.98, PANEL)
text(slide, "EARLY FINDINGS", 0.88, 1.75, 4.85, 0.31, 12, GREEN, True)
findings = [
    "A* reconnection completed the most starts in this map.",
    "Obstacle coupling improved DMP completion by 18.3 points.",
    "Runtime safety stops exposed controller corner-cutting.",
]
for i, item in enumerate(findings):
    y = 2.29 + i * 1.21
    text(slide, f"0{i+1}", 0.87, y, 0.57, 0.37, 15, GREEN, True)
    text(slide, item, 1.49, y - 0.03, 4.36, 0.83, 17, WHITE)
rect(slide, 6.58, 1.46, 5.98, 4.98, PANEL)
text(slide, "NEXT MILESTONES", 6.91, 1.75, 4.86, 0.31, 12, GOLD, True)
next_steps = [
    "Feed periodic ICP corrections into noisy odometry runs.",
    "Test more obstacle layouts and larger pose offsets.",
    "Tune controller and DMP coupling; report uncertainty.",
]
for i, item in enumerate(next_steps):
    y = 2.29 + i * 1.21
    text(slide, f"0{i+1}", 6.89, y, 0.57, 0.37, 15, GOLD, True)
    text(slide, item, 7.52, y - 0.03, 4.48, 0.83, 17, WHITE)

# 9 — transparent experiment details and source files
slide = base("Experiment details and reproducibility", "08  /  Appendix")
rect(slide, 0.55, 1.44, 5.87, 4.95, PANEL)
text(slide, "TRIAL DESIGN", 0.89, 1.76, 4.9, 0.35, 13, CYAN, True)
lines = [
    "60 starts = 4 x positions × 5 y positions × 3 headings",
    "10 × 7 m static map with two circular obstacles",
    "Rover radius 0.16 m; path margin 0.12 m",
    "Deterministic bounded unicycle controller",
    "Unsafe paths rejected before execution; margin monitored online",
]
for i, line in enumerate(lines):
    text(slide, "• " + line, 0.90, 2.24 + i * 0.73, 4.97, 0.63, 15, WHITE)
rect(slide, 6.67, 1.44, 5.90, 4.95, PANEL)
text(slide, "REPRODUCE", 7.00, 1.76, 5.00, 0.35, 13, GREEN, True)
text(slide, "python src/robot_motion.py\npython build_slides.py", 7.00, 2.22, 5.08, 0.87,
     18, WHITE, font="Cascadia Code")
text(slide, "The script exports trial-level CSV, summary JSON, and the figures shown here.",
     7.00, 3.45, 4.89, 1.05, 16, MUTED)
text(slide, "Sources: class proposal and literature review; Ijspeert et al. (2013), Hoffmann et al. (2009), Besl & McKay (1992).",
     7.00, 5.07, 4.97, 0.94, 12, MUTED)

OUT.parent.mkdir(exist_ok=True)
pres.save(OUT)
print(OUT)
