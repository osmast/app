"""
generate_multiwell_matrix.py
Generates a complete single-page A4 calibration chart featuring:
  3 Well Sizes (6-well, 24-well, 96-well)
  x 3 Patterns (Dots, Line Grid, Checkerboard)
  x 6 Pitches per category
  = 54 circular targets total.

Includes outer dashed cut rings, registration crosshairs, pitch labels,
and an aligned 50 mm caliper verification scale bar.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_FILENAME = "discorpy_multiwell_matrix.pdf"

# Standard ANSI/SLAS well diameters (mm)
DIA_6W = 34.8     # r = 17.4 mm
DIA_24W = 15.6    # r = 7.8 mm
DIA_96W = 6.4     # r = 3.2 mm

# 6 pitch steps per well size (mm)
PITCHES_6W = [2.00, 1.50, 1.25, 1.00, 0.75, 0.50]
PITCHES_24W = [1.50, 1.25, 1.00, 0.75, 0.50, 0.35]
PITCHES_96W = [1.00, 0.75, 0.50, 0.40, 0.30, 0.25]

# Page layout geometry (A4: 210 x 297 mm)
MARGIN_X = 12.0   # Left page margin
COL_W = 60.0      # Column width (60 * 3 = 180 mm)
GAP_X = 3.0       # Horizontal gutter between columns (12 + 60 + 3 + 60 + 3 + 60 + 12 = 210 mm)
HEADER_Y = 286.0  # Top header baseline
# ==============================================================================


def draw_header(c: canvas.Canvas):
    """Draws category titles and aligns the 50 mm verification ruler over Column 3."""
    titles = ["DOTS", "LINE GRID", "CHECKERBOARD"]
    c.setFont("Helvetica-Bold", 11)

    # Columns 1 & 2
    for i in range(2):
        col_x = MARGIN_X + i * (COL_W + GAP_X)
        c.drawString(col_x * mm, HEADER_Y * mm, titles[i])

    # Column 3 + Ruler
    col3_x = (MARGIN_X + 2 * (COL_W + GAP_X)) * mm
    c.drawString(col3_x, (HEADER_Y + 2.5) * mm, titles[2])

    ruler_y = (HEADER_Y - 1.2) * mm
    ruler_len = 50  # 50 mm scale bar

    c.setLineWidth(0.22 * mm)
    c.setStrokeColorRGB(0, 0, 0)
    c.line(col3_x, ruler_y, col3_x + ruler_len * mm, ruler_y)

    c.setLineWidth(0.12 * mm)
    c.setFont("Helvetica", 5.0)
    for k in range(ruler_len + 1):
        x = col3_x + k * mm
        if k % 10 == 0:
            tick = 2.8 * mm
            c.drawCentredString(x, ruler_y - tick - 2.0 * mm, str(k))
        elif k % 5 == 0:
            tick = 1.8 * mm
        else:
            tick = 0.9 * mm
        c.line(x, ruler_y, x, ruler_y - tick)


def draw_well_guides(c: canvas.Canvas, cx: float, cy: float, r: float, pitch: float):
    """Draws dashed cut rings, registration tick marks, and pitch labels."""
    c.saveState()
    # Dashed outer cutting boundary
    c.setStrokeColorRGB(0.65, 0.65, 0.65)
    c.setLineWidth(0.12 * mm)
    c.setDash(1.0 * mm, 1.0 * mm)
    c.circle(cx * mm, cy * mm, r * mm, stroke=1, fill=0)

    # Crosshairs (cut registration marks outside the perimeter)
    c.setStrokeColorRGB(0.0, 0.0, 0.0)
    c.setDash([])
    c.setLineWidth(0.18 * mm)
    t = 1.2 * mm
    c.line((cx - r - t) * mm, cy * mm, (cx - r) * mm, cy * mm)
    c.line((cx + r) * mm, cy * mm, (cx + r + t) * mm, cy * mm)
    c.line(cx * mm, (cy - r - t) * mm, cx * mm, (cy - r) * mm)
    c.line(cx * mm, (cy + r) * mm, cx * mm, (cy + r + t) * mm)

    # Pitch label above circle
    c.setFont("Helvetica-Bold", 4.5)
    c.setFillColorRGB(0.15, 0.15, 0.15)
    c.drawCentredString(cx * mm, (cy + r + 1.0) * mm, f"{pitch:.2f} mm")
    c.restoreState()


def draw_dots_circle(c: canvas.Canvas, cx: float, cy: float, r: float, pitch: float):
    draw_well_guides(c, cx, cy, r, pitch)
    c.saveState()
    clip_path = c.beginPath()
    clip_path.circle(cx * mm, cy * mm, r * mm)
    c.clipPath(clip_path, stroke=0, fill=0)

    dot_dia = 0.40 * pitch
    dot_r = (dot_dia / 2.0) * mm
    c.setFillColorRGB(0, 0, 0)

    steps = int(round((2 * r) / pitch)) + 2
    start_x = cx - (steps / 2.0) * pitch
    start_y = cy - (steps / 2.0) * pitch

    for col in range(steps):
        x = (start_x + (col + 0.5) * pitch) * mm
        for row in range(steps):
            y = (start_y + (row + 0.5) * pitch) * mm
            c.circle(x, y, dot_r, stroke=0, fill=1)
    c.restoreState()


def draw_lines_circle(c: canvas.Canvas, cx: float, cy: float, r: float, pitch: float):
    draw_well_guides(c, cx, cy, r, pitch)
    c.saveState()
    clip_path = c.beginPath()
    clip_path.circle(cx * mm, cy * mm, r * mm)
    c.clipPath(clip_path, stroke=0, fill=0)

    line_w = max(0.04, 0.15 * pitch)
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(line_w * mm)

    steps = int(round((2 * r) / pitch)) + 2
    start_x = cx - (steps / 2.0) * pitch
    start_y = cy - (steps / 2.0) * pitch

    for col in range(steps):
        x = (start_x + (col + 0.5) * pitch) * mm
        c.line(x, (cy - r) * mm, x, (cy + r) * mm)
    for row in range(steps):
        y = (start_y + (row + 0.5) * pitch) * mm
        c.line((cx - r) * mm, y, (cx + r) * mm, y)
    c.restoreState()


def draw_checker_circle(c: canvas.Canvas, cx: float, cy: float, r: float, pitch: float):
    draw_well_guides(c, cx, cy, r, pitch)
    c.saveState()
    clip_path = c.beginPath()
    clip_path.circle(cx * mm, cy * mm, r * mm)
    c.clipPath(clip_path, stroke=0, fill=0)

    c.setFillColorRGB(0, 0, 0)
    steps = int(round((2 * r) / pitch)) + 2
    start_x = cx - (steps / 2.0) * pitch
    start_y = cy - (steps / 2.0) * pitch
    p = pitch * mm

    for col in range(steps):
        for row in range(steps):
            if (col + row) % 2 == 0:
                c.rect((start_x + col * pitch) * mm, (start_y + row * pitch) * mm, p, p, stroke=0, fill=1)
    c.restoreState()


def main():
    c = canvas.Canvas(OUTPUT_FILENAME, pagesize=A4)
    c.setTitle("Discorpy Multi-Well Complete Matrix")

    draw_header(c)
    draw_fns = [draw_dots_circle, draw_lines_circle, draw_checker_circle]

    # ==========================================================================
    # SECTION 1: 6-WELL (Ø34.8 mm, r = 17.4 mm)
    # 6 pitches per column arranged as 2 columns of 3
    # ==========================================================================
    r6 = DIA_6W / 2.0
    sec1_y_base = HEADER_Y - 14.0
    c.setFont("Helvetica-Bold", 6.5)
    c.setFillColorRGB(0.1, 0.1, 0.1)

    for col_idx in range(3):
        col_x0 = MARGIN_X + col_idx * (COL_W + GAP_X)
        c.drawString(col_x0 * mm, sec1_y_base * mm, "6-WELL FORMAT (Ø34.8 mm)")

        # 2 columns x 3 rows of circles inside this 60 mm column
        for idx, pitch in enumerate(PITCHES_6W):
            sub_col = idx % 2
            sub_row = idx // 2
            cx = col_x0 + 15.0 + sub_col * 30.0
            cy = (sec1_y_base - 22.0) - sub_row * 40.5
            draw_fns[col_idx](c, cx, cy, r6, pitch)

    # ==========================================================================
    # SECTION 2: 24-WELL (Ø15.6 mm, r = 7.8 mm)
    # 6 pitches per column arranged as 3 columns of 2
    # ==========================================================================
    r24 = DIA_24W / 2.0
    sec2_y_base = sec1_y_base - 124.0

    for col_idx in range(3):
        col_x0 = MARGIN_X + col_idx * (COL_W + GAP_X)
        c.drawString(col_x0 * mm, sec2_y_base * mm, "24-WELL FORMAT (Ø15.6 mm)")

        for idx, pitch in enumerate(PITCHES_24W):
            sub_col = idx % 3
            sub_row = idx // 3
            cx = col_x0 + 10.0 + sub_col * 20.0
            cy = (sec2_y_base - 12.0) - sub_row * 20.0
            draw_fns[col_idx](c, cx, cy, r24, pitch)

    # ==========================================================================
    # SECTION 3: 96-WELL (Ø6.4 mm, r = 3.2 mm)
    # 6 pitches per column arranged as 3 columns of 2
    # ==========================================================================
    r96 = DIA_96W / 2.0
    sec3_y_base = sec2_y_base - 44.0

    for col_idx in range(3):
        col_x0 = MARGIN_X + col_idx * (COL_W + GAP_X)
        c.drawString(col_x0 * mm, sec3_y_base * mm, "96-WELL FORMAT (Ø6.4 mm)")

        for idx, pitch in enumerate(PITCHES_96W):
            sub_col = idx % 3
            sub_row = idx // 3
            cx = col_x0 + 10.0 + sub_col * 20.0
            cy = (sec3_y_base - 8.0) - sub_row * 10.5
            draw_fns[col_idx](c, cx, cy, r96, pitch)

    c.showPage()
    c.save()
    print(f"Generated complete multi-well matrix: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()