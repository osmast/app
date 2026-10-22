"""Streaming helpers: URL builders, MediaMTX status and the reader surface."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import numpy as np

from daphnia import settings
from daphnia.streaming import (
    StreamReader,
    StreamService,
    is_running,
    mediamtx,
    publish_srt_url,
    read_rtsp_url,
    status,
)
from daphnia.streaming.reader import MAX_FAILED_READS


class UrlBuilderTests(unittest.TestCase):
    def test_publish_srt_url(self):
        self.assertEqual(
            publish_srt_url("10.0.0.5", "live/cam", 8890),
            "srt://10.0.0.5:8890?streamid=publish:live/cam",
        )

    def test_read_rtsp_url(self):
        self.assertEqual(
            read_rtsp_url("127.0.0.1", "live/cam", 8554), "rtsp://127.0.0.1:8554/live/cam"
        )

    def test_defaults_follow_settings(self):
        expected = f"rtsp://{settings.STREAM_HOST}:{settings.RTSP_PORT}/{settings.STREAM_PATH}"
        self.assertEqual(read_rtsp_url(), expected)


class MediamtxTests(unittest.TestCase):
    def test_status_reports_expected_keys(self):
        info = status()
        for key in ("installed", "binary", "config", "config_exists", "running"):
            self.assertIn(key, info)

    def test_repository_config_exists(self):
        self.assertTrue(status()["config_exists"])

    def test_is_running_is_false_for_a_closed_port(self):
        self.assertFalse(is_running(port=1, timeout=0.5))


class StreamCodecTests(unittest.TestCase):
    """What is being published, asked from MediaMTX - and never invented.

    The answer is what turns "no frames arriving" into a cause with a fix, so the
    parsing has to keep three different things apart: a path publishing H.265, a path
    that exists and is idle, and an API that cannot be reached at all.
    """

    H265 = {"itemCount": 1, "items": [
        {"name": "live/cam", "online": True, "tracks": ["H265"]}]}
    IDLE = {"itemCount": 1, "items": [
        {"name": "live/cam", "online": False, "tracks": []}]}
    ELSEWHERE = {"itemCount": 1, "items": [
        {"name": "someone/else", "online": True, "tracks": ["H264"]}]}

    def _codec(self, payload, name="live/cam"):
        return mediamtx.codec_of(mediamtx.path_status(payload, name))

    def test_a_publisher_is_read_from_the_paths_answer(self):
        self.assertEqual(self._codec(self.H265), "H265")

    def test_an_idle_path_has_no_codec(self):
        self.assertIsNotNone(mediamtx.path_status(self.IDLE, "live/cam"))
        self.assertEqual(self._codec(self.IDLE), "")

    def test_a_path_that_is_not_listed_is_not_an_idle_path(self):
        self.assertIsNone(mediamtx.path_status(self.ELSEWHERE, "live/cam"))
        self.assertEqual(self._codec(self.ELSEWHERE), "")

    def test_an_unusable_answer_is_empty_rather_than_a_guess(self):
        for payload in (None, {}, [], {"items": "nope"}, {"items": [None, 5]}):
            with self.subTest(payload=payload):
                self.assertEqual(self._codec(payload), "")

    def test_an_unreachable_api_is_not_a_verdict(self):
        """Nothing listening is "I could not ask", never "the codec is fine"."""
        self.assertEqual(mediamtx.stream_codec(port=1, timeout=0.2), "")

    def test_the_api_url_follows_the_settings(self):
        self.assertEqual(mediamtx.api_url("/v3/paths/list", "10.0.0.9", 9998),
                         "http://10.0.0.9:9998/v3/paths/list")
        self.assertIn(str(settings.MEDIAMTX_API_PORT), mediamtx.api_url())

    def test_hevc_is_the_one_codec_this_pipeline_pays_for(self):
        for codec in ("H265", "h265", "HEVC", "hevc"):
            self.assertTrue(mediamtx.is_hevc(codec), codec)
        for codec in ("H264", "AV1", "", None):
            self.assertFalse(mediamtx.is_hevc(codec), codec)


class ReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_reader_keeps_url_and_exposes_signals(self):
        reader = StreamReader("rtsp://127.0.0.1:8554/live/cam")
        self.assertEqual(reader.url, "rtsp://127.0.0.1:8554/live/cam")
        for signal in ("connected", "codec_known", "frame_ready", "degraded",
                       "connection_failed", "connection_lost"):
            self.assertTrue(hasattr(reader, signal), signal)
        self.assertFalse(reader.isRunning())
        self.assertEqual(reader.frames_failed, 0)
        self.assertEqual(reader.codec, "")


class _FakeCapture:
    """A scripted ``cv2.VideoCapture``: yields the given reads, then fails forever."""

    def __init__(self, results):
        self.results = list(results)
        self.released = False

    def set(self, *_args):
        return True

    def isOpened(self):
        return True

    def read(self):
        return self.results.pop(0) if self.results else (False, None)

    def release(self):
        self.released = True


class ReaderRecoveryTests(unittest.TestCase):
    """A failed read is *not* a disconnection - measured on an H.265 live stream.

    The old reader ended the capture on the first failed ``read()`` and announced
    "Stream ended or disconnected" over a stream that was up and whose next frames
    decoded fine. These drive the real loop with a scripted capture.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _frame():
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def _run_reader(self, results, codec=""):
        """Drive ``StreamReader._capture`` over a scripted sequence of reads."""
        import cv2

        fake = _FakeCapture(results)
        original_capture = cv2.VideoCapture
        original_codec = mediamtx.stream_codec
        cv2.VideoCapture = lambda *_args, **_kwargs: fake
        mediamtx.stream_codec = lambda *_args, **_kwargs: codec
        self.addCleanup(lambda: setattr(cv2, "VideoCapture", original_capture))
        self.addCleanup(lambda: setattr(mediamtx, "stream_codec", original_codec))

        reader = StreamReader("rtsp://test/cam")
        received, lost, degraded, codes = [], [], [], []
        reader.frame_ready.connect(received.append)
        reader.connection_lost.connect(lost.append)
        reader.degraded.connect(degraded.append)
        reader.codec_known.connect(codes.append)
        reader._capture()
        return reader, received, lost, degraded, codes

    def test_one_failed_read_is_not_a_disconnection(self):
        reader, received, lost, _degraded, _codes = self._run_reader(
            [(True, self._frame())] * 3 + [(False, None)]
            + [(True, self._frame())] * 2)

        self.assertEqual(len(received), 5)          # the stream kept delivering
        self.assertEqual(len(lost), 1)              # ... and was called lost once
        self.assertIn(f"after {MAX_FAILED_READS} failed reads", lost[0])
        self.assertIn("5 frame(s) received", lost[0])
        self.assertEqual(reader.frames_failed, 1 + MAX_FAILED_READS)

    def test_each_run_of_failures_is_reported_once_with_its_count(self):
        reader, _received, _lost, degraded, _codes = self._run_reader(
            [(False, None), (True, self._frame()), (False, None), (False, None)])

        self.assertEqual(degraded, [1, 2])          # once per run, with the count
        self.assertEqual(reader.frames_failed, 1 + MAX_FAILED_READS)

    def test_the_quiet_decoder_log_is_set_unless_the_operator_set_it(self):
        """FFmpeg's two lines per undecodable frame are replaced by the status bar."""
        previous = os.environ.pop("OPENCV_FFMPEG_LOGLEVEL", None)
        try:
            self._run_reader([(True, self._frame())])
            self.assertEqual(os.environ.get("OPENCV_FFMPEG_LOGLEVEL"),
                             settings.FFMPEG_LOGLEVEL)

            os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "32"
            self._run_reader([(True, self._frame())])
            self.assertEqual(os.environ["OPENCV_FFMPEG_LOGLEVEL"], "32")
        finally:
            if previous is None:
                os.environ.pop("OPENCV_FFMPEG_LOGLEVEL", None)
            else:
                os.environ["OPENCV_FFMPEG_LOGLEVEL"] = previous

    def test_the_publisher_codec_is_asked_and_reported(self):
        reader, _received, _lost, _degraded, codes = self._run_reader(
            [(True, self._frame())], codec="H265")

        self.assertEqual(reader.codec, "H265")
        self.assertEqual(codes, ["H265"])

    def test_an_unknown_codec_is_not_invented(self):
        reader, _received, _lost, _degraded, codes = self._run_reader(
            [(True, self._frame())], codec="")

        self.assertEqual(reader.codec, "")
        self.assertEqual(codes, [])


class ServiceStateTests(unittest.TestCase):
    """Connection state: an attached socket is not the same as a picture.

    These drive the service through the slots its reader emits, so the state machine
    is covered without opening a socket or starting a thread.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _service(self):
        # orientation="native" keeps the frame path free of the rotation helper.
        return StreamService(url="rtsp://test/cam", orientation="native")

    def test_a_new_service_is_disconnected_and_has_no_signal(self):
        service = self._service()
        self.assertEqual(service.state, StreamService.DISCONNECTED)
        self.assertFalse(service.connected)
        self.assertFalse(service.has_signal)
        self.assertEqual(service.frames_seen, 0)
        self.assertIn("not connected", service.state_label().lower())

    def test_an_open_stream_is_not_yet_a_picture(self):
        """The distinction the operator actually cares about."""
        service = self._service()
        service._on_connected()
        self.assertEqual(service.state, StreamService.WAITING)
        self.assertFalse(service.has_signal)
        self.assertIn("no frames", service.state_label().lower())

    def test_the_first_frame_is_what_makes_it_live(self):
        service = self._service()
        service._on_connected()
        service._on_frame(np.zeros((20, 30, 3), dtype=np.uint8))
        self.assertEqual(service.state, StreamService.LIVE)
        self.assertTrue(service.has_signal)
        self.assertEqual(service.frames_seen, 1)
        self.assertEqual(service.state_label(), "Camera live")

    def test_the_state_change_is_announced_once(self):
        """The bar updates on the transition, not on every single frame."""
        service = self._service()
        announcements = []
        service.status.connect(announcements.append)
        service._on_connected()
        service._on_frame(np.zeros((20, 30, 3), dtype=np.uint8))
        service._on_frame(np.zeros((20, 30, 3), dtype=np.uint8))
        self.assertEqual(announcements.count("Camera live"), 1)
        self.assertEqual(service.frames_seen, 2)

    def test_a_failure_says_why_and_disconnects(self):
        service = self._service()
        service._on_connected()
        service._on_failure("Could not open stream: rtsp://test/cam")
        self.assertEqual(service.state, StreamService.DISCONNECTED)
        self.assertFalse(service.has_signal)
        self.assertIn("Could not open stream", service.state_label())

    def test_disconnecting_drops_the_buffered_frames(self):
        """A stale picture after a drop would be worse than none."""
        service = self._service()
        service._on_connected()
        service._on_frame(np.zeros((20, 30, 3), dtype=np.uint8))
        service.stop()
        self.assertIsNone(service.current_frame())
        self.assertFalse(service.has_signal)
        self.assertEqual(service.frames_seen, 0)
        self.assertEqual(service.state, StreamService.DISCONNECTED)

    # -- what the publisher is sending, and what it costs ------------------
    def test_an_h265_publisher_is_named_with_the_setting_that_fixes_it(self):
        """Measured 2026-10-19: H.265 costs the first ~120 frames 21 of them."""
        service = self._service()
        service._on_codec("H265")
        service._on_frame(np.zeros((20, 30, 3), dtype=np.uint8))

        label = service.state_label()
        self.assertEqual(service.codec, "H265")
        self.assertIn("Camera live", label)
        self.assertIn("H265", label)
        self.assertIn("H.264", label)

    def test_an_h264_publisher_adds_no_caveat(self):
        service = self._service()
        service._on_codec("H264")
        service._on_frame(np.zeros((20, 30, 3), dtype=np.uint8))

        self.assertEqual(service.state_label(), "Camera live")
        self.assertEqual(service.stream_note(), "")

    def test_a_frame_that_could_not_be_decoded_is_counted_and_said(self):
        """The stream is still up, so this is a note - not a disconnection."""
        service = self._service()
        service._on_degraded(7)

        self.assertEqual(service.frames_failed, 7)
        self.assertIn("7 frame(s) could not be decoded", service.state_label())


if __name__ == "__main__":
    unittest.main()
