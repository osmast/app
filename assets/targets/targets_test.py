"""
generate_single_page_ladder_cyan.py
Generates a compact, single-page A4 test chart featuring all 3 Discorpy target categories
(Dots, Lines, Checkerboard) across 6 resolution steps (2.00 mm down to 0.50 mm).

Rendered strictly in pure CMYK Cyan (C:1.0, M:0.0, Y:0.0, K:0.0) so only the
cyan ink/toner channel is printed.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_FILENAME = "discorpy_single_page_ladder_cyan.pdf"

# Pitches for the 6 vertical steps (mm)
PITCHES = [2.00, 1.50, 1.25, 1.00, 0.75, 0.50]

# Rectangle tile geometry (mm)
TILE_W = 58.0         # Width of each rectangle
TILE_H = 37.0         # Height of each rectangle
GAP_X = 8.0           # Horizontal gutter between columns
LABEL_H = 3.0         # Height allocated for label above tile (mm)
GAP_Y = 5.0           # Gap between rows (mm)

# Page margins & alignment (A4: 210 x 297 mm)
MARGIN_X = 10.0       # 10 + 58 + 8 + 58 + 8 + 58 + 10 = 210 mm (exact fit)
HEADER_Y = 286.0      # Y position for top column titles & ruler
# ==============================================================================


def draw_cut_guides(c: canvas.Canvas, x0: float, y0: float, w: float, h: float, pitch: float):
    """Draws thin dashed boundary, corner cut marks, and heading directly ABOVE the tile in pure Cyan."""
    c.saveState()

    # 1. Heading ABOVE the tile (100% Cyan)
    c.setFont("Helvetica-Bold", 6.5)
    c.setFillColorCMYK(1.0, 0.0, 0.0, 0.0)
    c.drawString(x0 * mm, (y0 + h + 1.0) * mm, f"PITCH: {pitch:.2f} mm")

    # 2. Dashed outer cut rectangle (40% Cyan tint for visibility without bleed)
    c.setStrokeColorCMYK(0.40, 0.0, 0.0, 0.0)
    c.setLineWidth(0.12 * mm)
    c.setDash(1.2 * mm, 1.2 * mm)
    c.rect(x0 * mm, y0 * mm, w * mm, h * mm, stroke=1, fill=0)

    # 3. Outer corner registration ticks (100% Cyan)
    c.setStrokeColorCMYK(1.0, 0.0, 0.0, 0.0)
    c.setDash([])
    c.setLineWidth(0.18 * mm)
    t = 1.2 * mm
    c.line((x0 - t) * mm, y0 * mm, x0 * mm, y0 * mm)
    c.line(x0 * mm, (y0 - t) * mm, x0 * mm, y0 * mm)
    c.line((x0 + w) * mm, y0 * mm, (x0 + w + t) * mm, y0 * mm)
    c.line((x0 + w) * mm, (y0 - t) * mm, (x0 + w) * mm, y0 * mm)
    c.line((x0 - t) * mm, (y0 + h) * mm, x0 * mm, (y0 + h) * mm)
    c.line(x0 * mm, (y0 + h) * mm, x0 * mm, (y0 + h + t) * mm)
    c.line((x0 + w) * mm, (y0 + h) * mm, (x0 + w + t) * mm, (y0 + h) * mm)
    c.line((x0 + w) * mm, (y0 + h) * mm, (x0 + w) * mm, (y0 + h + t) * mm)

    c.restoreState()


def draw_dots(c: canvas.Canvas, x0: float, y0: float, pitch: float):
    draw_cut_guides(c, x0, y0, TILE_W, TILE_H, pitch)
    dot_dia = 0.40 * pitch
    r = (dot_dia / 2.0) * mm
    cols = int(round(TILE_W / pitch))
    rows = int(round(TILE_H / pitch))

    # 100% Cyan fill
    c.setFillColorCMYK(1.0, 0.0, 0.0, 0.0)
    for col in range(cols):
        cx = (x0 + 0.5 * pitch + col * pitch) * mm
        for row in range(rows):
            cy = (y0 + 0.5 * pitch + row * pitch) * mm
            c.circle(cx, cy, r, stroke=0, fill=1)


def draw_lines(c: canvas.Canvas, x0: float, y0: float, pitch: float):
    draw_cut_guides(c, x0, y0, TILE_W, TILE_H, pitch)
    line_w = max(0.04, 0.15 * pitch)
    cols = int(round(TILE_W / pitch))
    rows = int(round(TILE_H / pitch))

    # 100% Cyan stroke
    c.setStrokeColorCMYK(1.0, 0.0, 0.0, 0.0)
    c.setLineWidth(line_w * mm)
    for col in range(cols):
        x = (x0 + 0.5 * pitch + col * pitch) * mm
        c.line(x, y0 * mm, x, (y0 + TILE_H) * mm)
    for row in range(rows):
        y = (y0 + 0.5 * pitch + row * pitch) * mm
        c.line(x0 * mm, y, (x0 + TILE_W) * mm, y)


def draw_checker(c: canvas.Canvas, x0: float, y0: float, pitch: float):
    draw_cut_guides(c, x0, y0, TILE_W, TILE_H, pitch)
    cols = int(round(TILE_W / pitch))
    rows = int(round(TILE_H / pitch))
    p = pitch * mm

    # 100% Cyan fill
    c.setFillColorCMYK(1.0, 0.0, 0.0, 0.0)
    for col in range(cols):
        for row in range(rows):
            if (col + row) % 2 == 0:
                c.rect((x0 * mm) + col * p, (y0 * mm) + row * p, p, p, stroke=0, fill=1)


def draw_header(c: canvas.Canvas):
    """Draws column titles and aligns the verification ruler over Column 3 entirely in Cyan."""
    titles = ["DOTS", "LINE GRID", "CHECKERBOARD"]
    c.setFont("Helvetica-Bold", 10.5)
    c.setFillColorCMYK(1.0, 0.0, 0.0, 0.0)

    # Column 1 & 2 Titles
    for i in range(2):
        col_x = MARGIN_X + i * (TILE_W + GAP_X)
        c.drawString(col_x * mm, HEADER_Y * mm, titles[i])

    # Column 3 Title + Ruler
    col3_x = (MARGIN_X + 2 * (TILE_W + GAP_X)) * mm
    c.drawString(col3_x, (HEADER_Y + 2.5) * mm, titles[2])

    ruler_y = (HEADER_Y - 1.2) * mm
    ruler_len = int(TILE_W)  # 58 mm

    c.setLineWidth(0.22 * mm)
    c.setStrokeColorCMYK(1.0, 0.0, 0.0, 0.0)
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


def main():
    c = canvas.Canvas(OUTPUT_FILENAME, pagesize=A4)
    c.setTitle("Discorpy Single-Page Calibration Ladder (Cyan Only)")

    draw_header(c)

    draw_fns = [draw_dots, draw_lines, draw_checker]

    # Row 0 starts immediately under the ruler's lower number labels
    first_tile_top_y = HEADER_Y - 10.0
    row_step = TILE_H + LABEL_H + GAP_Y

    for row_idx, pitch in enumerate(PITCHES):
        tile_top_y = first_tile_top_y - (row_idx * row_step) - LABEL_H
        tile_bottom_y = tile_top_y - TILE_H
        for col_idx in range(3):
            x = MARGIN_X + col_idx * (TILE_W + GAP_X)
            draw_fns[col_idx](c, x, tile_bottom_y, pitch)

    c.showPage()
    c.save()
    print(f"Generated pure-cyan target PDF: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()