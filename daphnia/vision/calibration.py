"""Calibrate step: convert pixels into micrometres, and say how sure we are.

The scale is *decoupled* from the plates: each imaging session references a
verified calibration (``calibration_profiles.px_per_um``) instead of deriving its
scale from the well rim, which is impossible for 6-/24-well plates that are
larger than the microscope field of view.

A calibration is a **dated, immutable record**: recalibrating (a bumped adapter, a
different objective, a new session) adds a new record instead of editing the old
one, so every measurement keeps pointing at the exact scale - and the exact
uncertainty - it was taken under.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from daphnia import units

#: Click uncertainty of one landmark on a reference photo, in pixels. Two
#: landmarks are clicked, so the measured span carries twice this. Defined in
#: :mod:`daphnia.units`, which is also where the *total* budget of a length is
#: combined from it (`units.total_uncertainty_um`) - one constant, one owner.
CLICK_ERROR_PX = units.CLICK_ERROR_PX


@dataclass(frozen=True)
class OpticalProfile:
    """Runtime view of one row in ``calibration_profiles``: one calibration."""

    profile_id: str
    px_per_um: float
    uncertainty_pct: float = 0.5
    setup_key: str = ""          # the optical chain this calibration belongs to
    calibrated_at: str = ""      # when it was verified (naive UTC, from SQLite)
    label: str = ""              # the operator's own note for *this* calibration

    @classmethod
    def from_row(cls, row) -> "OpticalProfile":
        """Build a calibration from a ``calibration_profiles`` row."""
        keys = row.keys()
        uncertainty = row["uncertainty_pct"]
        return cls(
            profile_id=row["profile_id"],
            px_per_um=row["px_per_um"],
            uncertainty_pct=0.5 if uncertainty is None else uncertainty,
            setup_key=(row["setup_key"] if "setup_key" in keys else "") or "",
            calibrated_at=((row["last_verified_at"]
                            if "last_verified_at" in keys else "") or ""),
            label=((row["label"] if "label" in keys else "") or ""),
        )

    def to_um(self, pixels: float) -> float:
        """Convert a pixel distance to micrometres using this calibration."""
        if self.px_per_um <= 0:
            raise ValueError("px_per_um must be > 0")
        return pixels / self.px_per_um

    def uncertainty_um(self, um: float) -> float:
        """Absolute half-width of a length measured under this calibration."""
        return units.uncertainty_um(um, self.uncertainty_pct)

    def total_uncertainty_um(self, um: float, *, extra_um: float = 0.0) -> float:
        """The **whole** budget of one length under this calibration, in µm.

        This calibration's own ``+/-`` plus the two terms that come from measuring
        with it - where the landmarks were clicked and the pixel grid they landed
        on - plus ``extra_um`` for what the caller measured elsewhere (a model's
        error, a repeat's spread). See :func:`daphnia.units.total_uncertainty_um`
        and ``docs/accuracy.md`` §1; on this rig the click term dominates the
        calibration's own by more than an order of magnitude.
        """
        return units.total_uncertainty_um(um, self.px_per_um, self.uncertainty_pct,
                                          extra_um=extra_um)

    def range_um(self, um: float) -> tuple:
        """``(low, high)`` bounds of a length measured under this calibration."""
        half = self.uncertainty_um(um)
        return um - half, um + half


def px_per_um(detected_distance_px: float, actual_distance_um: float) -> float:
    """Scale factor from a reference measurement (e.g. a stage micrometer).

    Mirrors the generated column ``calibration_records.calculated_px_per_um``.
    """
    if actual_distance_um <= 0:
        raise ValueError("actual_distance_um must be > 0")
    return detected_distance_px / actual_distance_um


def relative_uncertainty_pct(distance_px: float,
                             click_error_px: float = CLICK_ERROR_PX) -> float:
    """Relative scale uncertainty of a two-click reference span, in percent.

    Each landmark is placed with roughly ``click_error_px`` of imprecision, so the
    span between them is uncertain by twice that. Relative to the span, that is
    the calibration's relative error - which is why clicking a *longer* span (ten
    pitches instead of one) makes the scale better, and gives a real number for
    "how accurate was this calibration".
    """
    if distance_px <= 0:
        raise ValueError("distance_px must be > 0")
    return 2.0 * click_error_px / distance_px * 100.0


def utc_now() -> datetime:
    """Naive UTC "now", the frame SQLite's ``datetime('now')`` lives in."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def calibration_age_hours(calibrated_at, now=None):
    """Hours since a calibration was verified, or ``None`` when unknown.

    SQLite timestamps are naive UTC, so a naive ``now`` is read as UTC too.
    """
    if not calibrated_at:
        return None
    try:
        stamp = datetime.fromisoformat(str(calibrated_at))
    except ValueError:
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
    reference = now or utc_now()
    if reference.tzinfo is not None:
        reference = reference.astimezone(timezone.utc).replace(tzinfo=None)
    return (reference - stamp).total_seconds() / 3600.0


def is_stale(calibrated_at, max_age_hours, now=None) -> bool:
    """True when a calibration is older than ``max_age_hours``.

    A phone clamped to an eyepiece can be bumped, which invalidates the scale
    without changing a single recorded value - so age is treated as a signal to
    re-verify, never as a formality.
    """
    if not max_age_hours or max_age_hours <= 0:
        return False
    age = calibration_age_hours(calibrated_at, now)
    return age is not None and age > max_age_hours

