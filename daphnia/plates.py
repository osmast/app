"""Multi-well plate geometry: a format defines the rows x columns and therefore
the well labels to iterate while capturing.

    6-well  2 x 3      A01 A02 A03 / B01 B02 B03
    24-well 4 x 6      A01 .. D06
    96-well 8 x 12     A01 .. H12

Treatment assignment on top of the geometry belongs to the plate builder; the
acquisition loop only needs the labels.
"""

#: format -> (rows, columns)
FORMATS = {
    "6": (2, 3),
    "24": (4, 6),
    "96": (8, 12),
}


def rows_columns(fmt: str):
    """(rows, columns) of a plate format; raises for unknown formats."""
    try:
        return FORMATS[str(fmt)]
    except KeyError:
        raise ValueError(f"unknown plate format {fmt!r}; expected one of {sorted(FORMATS)}")


def well_count(fmt: str) -> int:
    """Number of wells in a plate format."""
    rows, columns = rows_columns(fmt)
    return rows * columns


def row_labels(fmt: str) -> list:
    """Row letters, top to bottom ('A', 'B', ...)."""
    rows, _columns = rows_columns(fmt)
    return [chr(ord("A") + index) for index in range(rows)]


def well_labels(fmt: str) -> list:
    """Well labels in reading order, e.g. ``['A01', 'A02', ..., 'H12']``."""
    rows, columns = rows_columns(fmt)
    return [f"{chr(ord('A') + row)}{column:02d}"
            for row in range(rows)
            for column in range(1, columns + 1)]
