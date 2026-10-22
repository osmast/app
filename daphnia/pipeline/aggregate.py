"""Aggregate step: longitudinal series, dose-response and survival summaries.

Not implemented yet — the Visualize screen will consume these aggregations and
export them to CSV / Excel / Parquet.
"""


def growth_trajectory(conn, plate_id=None, **kwargs):
    """Return the L(t) growth trajectory per specimen (timepoint vs length).

    Raises:
        NotImplementedError: always, until measurements are being persisted.
    """
    raise NotImplementedError("Growth trajectories need persisted measurements first.")
