"""Computer-vision pipeline: Localize, Calibrate and Predict steps.

* :mod:`~daphnia.vision.detection`   — Localize: isolate the specimen in a well.
* :mod:`~daphnia.vision.calibration` — Calibrate: pixels -> micrometres.
* :mod:`~daphnia.vision.keypoints`   — Predict: eye apex / spine base keypoints.
* :mod:`~daphnia.vision.export`      — read a Dataset export (the corpus a model trains on).
* :mod:`~daphnia.vision.rtmdet`      — the RTMDet detector: train it, then run it live.

The last two are imported explicitly by their callers (and PyTorch is imported
inside them), so this package stays importable without the ML stack.
"""
