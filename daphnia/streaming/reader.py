"""Frame reader for RTSP streams (Ingest step).

Wraps OpenCV's FFmpeg backend in a :class:`QThread`. OpenCV is imported lazily so
the Qt application still starts when the CV dependencies are absent.

The phone publishes SRT; MediaMTX republishes the same path over RTSP, which is what
OpenCV reads here (OpenCV cannot read SRT directly).

**Two decoder facts this module is built around** (measured 2026-10-19 through
SRT -> MediaMTX -> RTSP on this machine, `docs/streaming.md`):

* a *failed* ``read()`` is normal and transient - it happens on a decode failure and on
  momentary starvation, and the next frames are fine. Ending the capture on the first
  one reported "Stream ended or disconnected" over a live stream, so a failure is now
  retried and counted, and the stream is called lost only after
  :data:`MAX_FAILED_READS` in a row;
* an **H.265** publisher costs the first ~120 frames about 21 of them ("Could not find
  ref with POC ..."): the reader joins mid-GOP. H.264 costs none. That is a setting on
  the phone, so the codec is asked from MediaMTX (:mod:`daphnia.streaming.mediamtx`) and
  reported once as a fact - ``codec_known`` - for the status bar to word it.
"""

import os

from PySide6.QtCore import QThread, Signal

from daphnia import settings
from daphnia.streaming import mediamtx

# Latency-tuned FFmpeg options: TCP transport avoids UDP packet loss over Wi-Fi.
FFMPEG_CAPTURE_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"

#: Failed reads **in a row** before the stream is called lost. One failed read is not a
#: disconnection: measured 2026-10-19 on an H.265 stream, a decode failure and a
#: momentary starvation both return ``False`` while the frames after them decode fine -
#: and ending the capture there reported "Stream ended or disconnected" over a live
#: stream, which is exactly the false alarm this constant removes.
MAX_FAILED_READS = 5

#: Readers that are currently running. Qt aborts the process when a QThread is
#: destroyed while still running, so a reader whose caller has released it (for
#: example because a ``stop()`` timed out) must stay referenced until it finishes.
_ACTIVE = set()


class StreamReader(QThread):
    """Grab frames from a stream URL and emit them as BGR ``numpy.ndarray``.

    Signals:
        connected: emitted once the stream is open.
        codec_known(str): the publisher's video codec, when MediaMTX can be asked.
        frame_ready(object): the next frame (``numpy.ndarray``, BGR).
        degraded(int): frames could not be decoded (the running count), once per run
            of failures - the stream is still up unless ``connection_lost`` follows.
        connection_failed(str): the stream could not be opened.
        connection_lost(str): the stream ended or dropped mid-capture.
    """

    connected = Signal()
    codec_known = Signal(str)
    frame_ready = Signal(object)
    degraded = Signal(int)
    connection_failed = Signal(str)
    connection_lost = Signal(str)

    def __init__(self, url: str, buffer_size: int = 1, parent=None):
        super().__init__(parent)
        self.url = url
        self.buffer_size = buffer_size
        self._running = False
        self.frames_read = 0
        self.frames_failed = 0
        self.codec = ""              # what the publisher sends, when known

    def start(self, *args, **kwargs):
        """Start the thread, keeping it referenced while it runs (see ``_ACTIVE``)."""
        _ACTIVE.add(self)
        super().start(*args, **kwargs)

    def run(self):
        try:
            self._capture()
        finally:
            _ACTIVE.discard(self)

    def _capture(self):
        try:
            import cv2
        except ImportError:
            self.connection_failed.emit(
                "OpenCV is not available — install it (conda env install / opencv-python)."
            )
            return

        # FFmpeg's own decoder log is **not** where this app tells the operator what is
        # wrong: an H.265 publisher makes libavcodec print two lines per undecodable
        # frame (measured: 42 lines for one 120-frame read), and the operator cannot act
        # on that. `settings.FFMPEG_LOGLEVEL` quietens it; the same fact is stated once
        # in the status bar, with the setting to change (`service.stream_note`).
        os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", settings.FFMPEG_LOGLEVEL)
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", FFMPEG_CAPTURE_OPTIONS)

        # What is being published, asked before the first frame: it is what turns "no
        # frames arriving" into a cause the operator can fix in one phone setting.
        self.codec = mediamtx.stream_codec()
        if self.codec:
            self.codec_known.emit(self.codec)

        capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        if not capture.isOpened():
            self.connection_failed.emit(f"Could not open stream: {self.url}")
            return

        self._running = True
        self.connected.emit()
        failures = 0
        try:
            while self._running:
                ok, frame = capture.read()
                if ok:
                    failures = 0               # a run of failures ended: still alive
                    self.frames_read += 1
                    self.frame_ready.emit(frame)
                    continue
                failures += 1
                self.frames_failed += 1
                if failures == 1:
                    # Said at once, so a slow failure is visible while it is retried;
                    # the count is a fact, and the wording is the service's job.
                    self.degraded.emit(self.frames_failed)
                if failures >= MAX_FAILED_READS:
                    if self._running:
                        self.connection_lost.emit(
                            f"Stream ended or dropped after {failures} failed reads "
                            f"({self.frames_read} frame(s) received).")
                    break
        finally:
            capture.release()

    def stop(self, timeout_ms: int = 3000) -> None:
        """Ask the thread to stop and wait for it to finish.

        A stream wedged inside OpenCV never returns on its own; terminating it is
        the lesser evil, because letting the QThread be destroyed while running
        aborts the whole process.
        """
        self._running = False
        if not self.isRunning():
            return
        if self.wait(timeout_ms):
            return
        self.terminate()
        self.wait(timeout_ms)
