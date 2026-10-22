"""The seven high-level pipeline steps, in execution order.

This module documents the architecture in code so the orchestration layer has a
single, agreed vocabulary:

    Ingest -> Localize -> Calibrate -> Predict -> Derive -> Persist -> Aggregate
"""

from enum import Enum


class Step(str, Enum):
    """Names of the pipeline stages."""

    INGEST = "Ingest"        # grab frames from the MediaMTX stream
    LOCALIZE = "Localize"    # isolate the specimen inside the well
    CALIBRATE = "Calibrate"  # pixels -> micrometres via the optical profile
    PREDICT = "Predict"      # anatomical keypoints (eye apex, spine base)
    DERIVE = "Derive"        # biological measurements (body length)
    PERSIST = "Persist"      # write measurements / assets to SQLite
    AGGREGATE = "Aggregate"  # longitudinal series, dose-response, survival


#: Canonical execution order.
PIPELINE = (
    Step.INGEST,
    Step.LOCALIZE,
    Step.CALIBRATE,
    Step.PREDICT,
    Step.DERIVE,
    Step.PERSIST,
    Step.AGGREGATE,
)
