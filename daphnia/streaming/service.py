"""The one connection to the camera stream, shared by every screen.

The phone is connected **once** - normally on the Setup step - and each screen
that shows the live feed binds a preview to this service. Keeping the connection
state here instead of inside a widget is what makes "connect once, preview
everywhere" possible, and it lets the workflow gate on the connection.
"""

from collections import deque

from PySide6.QtCore import QObject, Signal

from daphnia import settings
from daphnia.streaming import mediamtx
from daphnia.streaming.reader import StreamReader
from daphnia.streaming.sources import read_rtsp_url
from daphnia.vision.frames import to_landscape


class StreamService(QObject):
    """Owns the reader thread, the frame buffer and the stream's aspect ratio.

    Signals:
        status(str): human-readable connection state, for any screen to display.
        opened/closed: the connection became available / went away.
        frames_changed: a new frame is buffered (pull it with ``current_frame()``).
        ratio_changed(float): the stream's width/height ratio became known.
    """

    status = Signal(str)
    opened = Signal()
    closed = Signal()
    frames_changed = Signal()
    ratio_changed = Signal(float)

    #: The states the connection passes through, as the app reports them. They are
    #: deliberately more than "connected / not": a socket can be open while nothing
    #: is being delivered, and that is the case an operator most needs to see.
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    WAITING = "no signal"
    LIVE = "live"

    def __init__(self, url=None, buffer=None, orientation=None, parent=None):
        super().__init__(parent)
        self.url = url or read_rtsp_url()
        self.orientation = orientation or settings.STREAM_ORIENTATION
        self._recent = deque(maxlen=buffer or settings.BURST_SIZE)
        self._reader = None
        self._aspect = None          # width / height of the incoming stream
        self._state = self.DISCONNECTED
        self._frames_seen = 0        # frames received since this connection opened
        self._frames_failed = 0      # frames the decoder could not produce
        self._codec = ""             # what the publisher sends, when MediaMTX says
        self._last_error = ""        # why the last attempt failed, if it did

    # -- state -------------------------------------------------------------
    @property
    def connected(self) -> bool:
        """True while a reader thread is attached."""
        return self._reader is not None

    @property
    def state(self) -> str:
        """One of :attr:`DISCONNECTED`, :attr:`CONNECTING`, :attr:`WAITING`,
        :attr:`LIVE`."""
        return self._state

    @property
    def frames_seen(self) -> int:
        """How many frames have arrived since the connection opened."""
        return self._frames_seen

    @property
    def frames_failed(self) -> int:
        """How many frames the decoder could not produce since this connection opened."""
        return self._frames_failed

    @property
    def codec(self) -> str:
        """The publisher's video codec, as MediaMTX reports it (``""`` while unknown)."""
        return self._codec

    @property
    def has_signal(self) -> bool:
        """True once at least one frame has actually arrived.

        This is what an operator means by "are we getting a picture" - and it is not
        the same question as :attr:`connected`, which only says the stream opened.
        """
        return self._frames_seen > 0

    def state_label(self) -> str:
        """Operator-facing wording for :attr:`state`, with any caveat that matters."""
        note = self.stream_note()
        text = self._state_text()
        return f"{text}  {note}" if note else text

    def _state_text(self) -> str:
        """The connection state alone, in words."""
        if self._state == self.LIVE:
            return "Camera live"
        if self._state == self.WAITING:
            return "Connected, but no frames arriving"
        if self._state == self.CONNECTING:
            return f"Connecting to {self.url} ..."
        if self._last_error:
            # "Not connected" alone hides the reason, which is the useful part.
            return f"Camera not connected. {self._last_error}"
        return "Camera not connected"

    def stream_note(self) -> str:
        """A caveat that changes what the operator should do, or ``""``.

        This is where a *decoder* fact becomes actionable. Measured 2026-10-19 through
        SRT -> MediaMTX -> RTSP: an H.265 publisher costs the first ~120 frames about 21
        of them ("Could not find ref with POC ...", the reader joining mid-GOP), at
        24.1 fps against 27.5 fps for H.264, which costs none - and the setting that
        fixes it is on the **phone**. Stating it here is what lets
        ``settings.FFMPEG_LOGLEVEL`` keep FFmpeg's own two-lines-per-frame log quiet
        without anything going unsaid.
        """
        notes = []
        if self._frames_failed:
            notes.append(f"{self._frames_failed} frame(s) could not be decoded")
        if mediamtx.is_hevc(self._codec):
            notes.append(f"the publisher sends {self._codec} - set the phone's video "
                         "codec to H.264")
        return f"({'; '.join(notes)})" if notes else ""

    @property
    def aspect(self):
        """width / height of the stream, or ``None`` before the first frame."""
        return self._aspect

    @property
    def frames_buffered(self) -> int:
        """How many frames the burst buffer currently holds."""
        return len(self._recent)

    def set_url(self, url):
        """Change the stream URL (takes effect on the next ``start()``)."""
        self.url = url

    def set_buffer(self, size):
        """Resize the burst buffer, keeping the frames already buffered."""
        self._recent = deque(self._recent, maxlen=max(1, int(size)))

    # -- lifecycle ---------------------------------------------------------
    def start(self, url=None):
        """Connect to the stream (idempotent)."""
        if self.connected:
            return
        if url:
            self.url = url
        self._state = self.CONNECTING
        self._frames_seen = 0
        self._frames_failed = 0
        self._codec = ""
        self._last_error = ""
        self.status.emit(self.state_label())
        self._reader = StreamReader(self.url)
        self._reader.connected.connect(self._on_connected)
        self._reader.codec_known.connect(self._on_codec)
        self._reader.frame_ready.connect(self._on_frame)
        self._reader.degraded.connect(self._on_degraded)
        self._reader.connection_failed.connect(self._on_failure)
        self._reader.connection_lost.connect(self._on_failure)
        self._reader.start()

    def stop(self):
        """Disconnect and drop the buffered frames."""
        reader, self._reader = self._reader, None
        if reader is not None:
            reader.stop()
        self._state = self.DISCONNECTED
        self._frames_seen = 0
        self._recent.clear()
        self.closed.emit()

    # -- frames ------------------------------------------------------------
    def current_frame(self):
        """Newest frame (BGR ``numpy.ndarray``), or ``None``."""
        return self._recent[-1] if self._recent else None

    def recent_frames(self) -> list:
        """Buffered frames, oldest first -- the candidates for a burst."""
        return list(self._recent)

    # -- slots -------------------------------------------------------------
    def _on_connected(self):
        """The stream opened - though nothing has been delivered yet."""
        self._last_error = ""
        self._state = self.WAITING
        self.status.emit(self.state_label())
        self.opened.emit()

    def _on_codec(self, codec):
        """The publisher's codec is known: name it, so a decoder cost has a cause."""
        self._codec = str(codec or "")
        self.status.emit(self.state_label())

    def _on_degraded(self, failed):
        """Frames could not be decoded - a count, not a disconnection (yet)."""
        self._frames_failed = int(failed)
        self.status.emit(self.state_label())

    def _on_frame(self, frame):
        if self.orientation == "landscape":
            frame = to_landscape(frame)
        self._recent.append(frame)

        self._frames_seen += 1
        if self._state != self.LIVE:
            # The first frame is the moment there is something to see, so the state
            # changes here rather than when the socket merely opened.
            self._state = self.LIVE
            self.status.emit(self.state_label())

        height, width = frame.shape[:2]
        aspect = width / height
        if self._aspect is None or abs(aspect - self._aspect) > 0.01:
            self._aspect = aspect
            self.ratio_changed.emit(aspect)
        self.frames_changed.emit()

    def _on_failure(self, message):
        self._last_error = str(message)
        self.stop()
        self.status.emit(self.state_label())


_SHARED = None


def shared_service() -> StreamService:
    """The app's single stream connection, created on first use.

    One camera, one connection: every screen observes the same service.
    """
    global _SHARED
    if _SHARED is None:
        _SHARED = StreamService()
    return _SHARED
