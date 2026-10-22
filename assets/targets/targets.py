"""
generate_targets.py
Generates an exact-dimension, multi-page vector PDF with Discorpy calibration targets:
  Page 1: Dots
  Page 2: Line Grid
  Page 3: Checkerboard

Every dimension (pitch, tile size, margins, line width) is defined in millimeters.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# ==============================================================================
# CONFIGURATION (Edit these parameters freely)
# ==============================================================================
OUTPUT_FILENAME = "1cm_calibration_print.pdf"

# Grid / Tile specs (all in mm)
PITCH = 10           # Grid pitch in mm (1.0 mm checkerboard, 1.0 mm dot spacing)
TILE_SIZE = 80.0      # Tile width & height (80 mm x 80 mm; multiple of PITCH)
DOT_DIAMETER = 0.4    # Diameter of dots for the dot target
LINE_WIDTH = 0.15     # Line thickness for the line grid target

# Page layout (2 columns x 3 rows = 6 tiles)
COLS = 2
ROWS = 3
GAP_X = 12.0          # Horizontal cutting margin between tiles (mm)
GAP_Y = 8.0           # Vertical cutting margin between tiles (mm)

# Page positioning
MARGIN_X = 18.0       # Left margin on A4 sheet (mm)
HEADER_Y = 280.0      # Top of sheet reference for header/scale bar (mm)
GRID_TOP_Y = 264.0    # Top edge of highest row of tiles (mm)
# ==============================================================================


def draw_header(c: canvas.Canvas, title: str):
    """Draws page title on the left and a metric ruler aligned with column 2."""
    # 1. Main Title (Left aligned above Column 1)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(MARGIN_X * mm, HEADER_Y * mm, title)

    # 2. Metric Verification Scale Bar (Exactly aligned with Column 2)
    ruler_x = (MARGIN_X + TILE_SIZE + GAP_X) * mm
    ruler_y = (HEADER_Y + 1.0) * mm
    scale_len = TILE_SIZE

    # Baseline
    c.setLineWidth(0.25 * mm)
    c.setStrokeColorRGB(0, 0, 0)
    c.line(ruler_x, ruler_y, ruler_x + scale_len * mm, ruler_y)

    # Ruler ticks and numbers (no title text)
    c.setLineWidth(0.12 * mm)
    c.setFont("Helvetica", 5.5)
    for k in range(int(scale_len) + 1):
        x = ruler_x + k * mm
        if k % 10 == 0:
            tick = 3.0 * mm
            c.drawCentredString(x, ruler_y - tick - 2.5 * mm, str(k))
        elif k % 5 == 0:
            tick = 2.0 * mm
        else:
            tick = 1.0 * mm
        c.line(x, ruler_y, x, ruler_y - tick)


def draw_cut_guides(c: canvas.Canvas, x0: float, y0: float, size: float):
    """Draws dashed boundary and corner registration cut marks."""
    c.saveState()
    # Dashed outer box
    c.setStrokeColorRGB(0.7, 0.7, 0.7)
    c.setLineWidth(0.12 * mm)
    c.setDash(1.5 * mm, 1.5 * mm)
    c.rect(x0 * mm, y0 * mm, size * mm, size * mm, stroke=1, fill=0)

    # Corner cut ticks
    c.setStrokeColorRGB(0.0, 0.0, 0.0)
    c.setDash([])
    c.setLineWidth(0.2 * mm)
    t = 1.5 * mm
    c.line((x0 - t) * mm, y0 * mm, x0 * mm, y0 * mm)
    c.line(x0 * mm, (y0 - t) * mm, x0 * mm, y0 * mm)
    c.line((x0 + size) * mm, y0 * mm, (x0 + size + t) * mm, y0 * mm)
    c.line((x0 + size) * mm, (y0 - t) * mm, (x0 + size) * mm, y0 * mm)
    c.line((x0 - t) * mm, (y0 + size) * mm, x0 * mm, (y0 + size) * mm)
    c.line(x0 * mm, (y0 + size) * mm, x0 * mm, (y0 + size + t) * mm)
    c.line((x0 + size) * mm, (y0 + size) * mm, (x0 + size + t) * mm, (y0 + size) * mm)
    c.line((x0 + size) * mm, (y0 + size) * mm, (x0 + size) * mm, (y0 + size + t) * mm)
    c.restoreState()


def draw_dots_tile(c: canvas.Canvas, x0: float, y0: float):
    draw_cut_guides(c, x0, y0, TILE_SIZE)
    r = (DOT_DIAMETER / 2.0) * mm
    c.setFillColorRGB(0, 0, 0)
    steps = int(TILE_SIZE / PITCH)
    for col in range(steps):
        cx = (x0 + 0.5 * PITCH + col * PITCH) * mm
        for row in range(steps):
            cy = (y0 + 0.5 * PITCH + row * PITCH) * mm
            c.circle(cx, cy, r, stroke=0, fill=1)


def draw_lines_tile(c: canvas.Canvas, x0: float, y0: float):
    draw_cut_guides(c, x0, y0, TILE_SIZE)
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(LINE_WIDTH * mm)
    steps = int(TILE_SIZE / PITCH)
    for i in range(steps):
        offset = (0.5 * PITCH + i * PITCH) * mm
        # Vertical line
        c.line((x0 * mm) + offset, y0 * mm, (x0 * mm) + offset, (y0 + TILE_SIZE) * mm)
        # Horizontal line
        c.line(x0 * mm, (y0 * mm) + offset, (x0 + TILE_SIZE) * mm, (y0 * mm) + offset)


def draw_checker_tile(c: canvas.Canvas, x0: float, y0: float):
    draw_cut_guides(c, x0, y0, TILE_SIZE)
    c.setFillColorRGB(0, 0, 0)
    steps = int(TILE_SIZE / PITCH)
    p = PITCH * mm
    for col in range(steps):
        for row in range(steps):
            if (col + row) % 2 == 0:
                c.rect((x0 * mm) + col * p, (y0 * mm) + row * p, p, p, stroke=0, fill=1)


def generate_page(c: canvas.Canvas, page_num: int, title: str, draw_fn):
    draw_header(c, f"{page_num}.  {title}")
    for col in range(COLS):
        for row in range(ROWS):
            x = MARGIN_X + col * (TILE_SIZE + GAP_X)
            y = GRID_TOP_Y - (row + 1) * TILE_SIZE - row * GAP_Y
            draw_fn(c, x, y)
    c.showPage()


def main():
    c = canvas.Canvas(OUTPUT_FILENAME, pagesize=A4)
    c.setTitle("Discorpy Calibration Targets")

    generate_page(c, 1, "DOTS", draw_dots_tile)
    generate_page(c, 2, "LINE GRID", draw_lines_tile)
    generate_page(c, 3, "CHECKERBOARD", draw_checker_tile)

    c.save()
    print(f"Generated {OUTPUT_FILENAME} successfully.")


if __name__ == "__main__":
    main()