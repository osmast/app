"""Length units for reporting: micrometres at the sensor, millimetres on paper.

The pipeline measures in pixels, converts to micrometres with the optical profile,
and reports both - ``length_um`` for machine precision, ``length_mm`` because that
is what a biologist reads. A measurement also carries the *uncertainty* of the
calibration it was taken under, so a reported length can be honest about how much
of it is method rather than animal.

## Two uncertainties, and they are not the same number

:func:`uncertainty_um` is the **scale's** relative error: how well the calibration
that turned pixels into micrometres was measured. It is what the app has always
reported, and on this rig it is the *smallest* term of the budget
(``docs/accuracy.md`` S1-S3).

:func:`total_uncertainty_um` is the **whole budget of one length**: the scale above,
plus where the two landmarks were placed (:data:`CLICK_ERROR_PX`), plus the pixel
grid the landmarks had to land on (:data:`QUANTISATION_PX`), plus anything the
caller knows and this module cannot derive (the pose model's own error, the spread
of a repeat). Measured on this rig the click alone is 73.4 µm against a 5.9 µm
scale term - i.e. the number on screen was 13x smaller than the budget. Both are
reported side by side so neither is mistaken for the other
(``docs/accuracy.md`` §1, ``memory-bank/mlContext.md`` law 5).
"""

import math

#: One micrometre, in millimetres.
MM_PER_UM = 0.001

#: Click uncertainty of one landmark, in pixels: how far a click can land from the
#: landmark it means. A length is *two* landmarks, so the span carries twice this -
#: which is why a longer reference span calibrates better
#: (``vision.calibration.relative_uncertainty_pct``, ``docs/accuracy.md`` S1/S6).
CLICK_ERROR_PX = 1.0

#: One endpoint's own quantisation, in pixels. A coordinate can only land on the
#: pixel grid, so it is uniform over ``+/- 0.5 px`` and its standard deviation is
#: ``1/sqrt(12)``. A length is the difference of two endpoints, so
#: ``sqrt(2)/sqrt(12) = 1/sqrt(6) ~ 0.408 px`` (``docs/accuracy.md`` S5).
QUANTISATION_PX = math.sqrt(2.0 / 12.0)


def to_mm(um: float) -> float:
    """Convert micrometres to millimetres."""
    return um * MM_PER_UM


def format_length(um: float, places: int = 1) -> str:
    """Human length carrying both units, e.g. ``'160.1 um (0.160 mm)'``."""
    return f"{um:.{places}f} um ({to_mm(um):.3f} mm)"


def uncertainty_um(um: float, uncertainty_pct: float) -> float:
    """Absolute half-width of a measurement, from a relative scale uncertainty.

    The scale error is *relative* (it comes from the calibration), so it scales
    with the measured length: ``160 um`` at ``1.4 %`` is ``+/- 2.2 um``.
    """
    return abs(um) * abs(uncertainty_pct) / 100.0


def format_uncertainty(um: float, uncertainty_pct: float, places: int = 1) -> str:
    """Length with its calibration uncertainty, e.g. ``'160.1 +/- 2.2 um'``."""
    return (f"{um:.{places}f} +/- "
            f"{uncertainty_um(um, uncertainty_pct):.{places}f} um")


def format_with_uncertainty(um: float, uncertainty_pct: float,
                            places: int = 1) -> str:
    """Uncertainty and both units, e.g. ``'160.1 +/- 2.2 um (0.160 mm)'``."""
    return (f"{format_uncertainty(um, uncertainty_pct, places)}"
            f" ({to_mm(um):.3f} mm)")


def quantisation_um(px_per_um: float, sigma_px: float = QUANTISATION_PX) -> float:
    """Absolute half-width the pixel grid alone puts on a length, in micrometres.

    ``sigma_px / S``. No new measurement is needed for it - ``S`` is already in
    every row - and it does **not** shrink with a better calibration, only with a
    finer scale: on the eyepiece chain (``S = 0.027264``) it is 15.0 µm, i.e. larger
    than the scale's own ``+/-`` beside it.
    """
    if px_per_um <= 0:
        raise ValueError("px_per_um must be > 0")
    return sigma_px / px_per_um


def click_uncertainty_um(px_per_um: float,
                         click_error_px: float = CLICK_ERROR_PX) -> float:
    """Absolute half-width of a two-click length, in micrometres.

    ``L * 2c/s``, and because ``s = L * S`` the length cancels:

        ``u_click = 2c / S``

    so this is a **constant** in micrometres however long the animal is. A longer
    animal is a longer *line* and a smaller relative error, which is why "measure
    it again more carefully" cannot be bought with size - only the calibration's
    route (a period instead of a click) removes the term.
    """
    if px_per_um <= 0:
        raise ValueError("px_per_um must be > 0")
    return 2.0 * click_error_px / px_per_um


def total_uncertainty_um(um: float, px_per_um: float, uncertainty_pct: float,
                         *, extra_um: float = 0.0,
                         click_error_px: float = CLICK_ERROR_PX) -> float:
    """The whole error budget of one length, in micrometres (1σ-ish, in quadrature).

        ``u_total = sqrt(u_scale^2 + u_click^2 + sigma_quant^2 + extra^2)``

    The first three terms are derived from what the row already carries - the
    length, the scale it was converted with, and the calibration's own relative
    error (``docs/accuracy.md`` §1). ``extra_um`` is where a caller puts the terms
    this module cannot derive: the pose model's own length error (S8, measured
    offline by ``scripts/train_rtmpose.py --evaluate``) and the standard error of a
    repeat (S7, ``vision.tracking``). **A human-placed measurement passes nothing**,
    because a human *is* the model there.

    ``click_error_px=0.0`` switches the click term **off**, which is what a length
    placed by the pose model needs: nobody clicked, so ``2c/S`` would be a number
    about a person who was not there. (At ``S = 0.01336`` that term alone is 150 µm
    - two thirds of the model's own 224 µm - so leaving it in would not be a small
    error.)

    What this deliberately leaves out, so nobody reads it as more than it is: the
    drift across the field (S4, a *position*, kept as its own line), the age of the
    calibration (S10, a step, not a σ), and the annotator's own variance (S6's
    second half, unmeasured). ``docs/accuracy.md`` §6 is that list.
    """
    return math.hypot(
        math.hypot(uncertainty_um(um, uncertainty_pct),
                   click_uncertainty_um(px_per_um, click_error_px)),
        math.hypot(quantisation_um(px_per_um), extra_um))


def format_total_uncertainty(um: float, px_per_um: float, uncertainty_pct: float,
                             *, extra_um: float = 0.0, total_um: float = None,
                             click_error_px: float = CLICK_ERROR_PX,
                             places: int = 1) -> str:
    """The total budget as text, e.g. ``'total +/- 75.1 um (3.8 %)'``.

    The share is printed because the budget is *relative*: the same rig gives
    ``+/- 75 um`` on a 2 mm adult (3.8 %) and on a 0.7 mm neonate (10.7 %), and a
    bare micrometre figure would read as the smaller one.

    ``total_um`` renders a budget that was **already computed and stored** (a row's
    own ``total_uncertainty_um``) instead of deriving it again, so a screen and an
    export cannot disagree about the same number. ``extra_um`` and
    ``click_error_px`` are passed through to :func:`total_uncertainty_um`, so the
    text and the number can never be built from different terms.
    """
    if total_um is None:
        total_um = total_uncertainty_um(um, px_per_um, uncertainty_pct,
                                        extra_um=extra_um,
                                        click_error_px=click_error_px)
    share = 100.0 * total_um / abs(um) if um else 0.0
    return f"total +/- {total_um:.{places}f} um ({share:.1f} %)"
