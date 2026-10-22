"""URL builders for the MediaMTX endpoints used by the pipeline.

    phone  --SRT publish-->  MediaMTX  --RTSP read-->  OpenCV (pipeline)

The phone publishes SRT; MediaMTX republishes the same path over RTSP, which is what
OpenCV reads. Ports and the ingest path come from :mod:`daphnia.settings` and match
``config/mediamtx.yml``.
"""

import socket

from daphnia import settings


def publish_srt_url(host=None, path=None, port=None) -> str:
    """SRT URL the phone application publishes to.

    SRT retransmits lost packets (so a lossy link cannot deliver a truncated H.264
    frame) and needs no ICE negotiation, so a VPN interface cannot hijack the media
    path. The stream id uses MediaMTX's ``publish:<path>`` syntax.
    """
    host = host or settings.STREAM_HOST
    port = port or settings.SRT_PORT
    path = path or settings.STREAM_PATH
    return f"srt://{host}:{port}?streamid=publish:{path}"


def read_rtsp_url(host=None, path=None, port=None) -> str:
    """RTSP URL OpenCV reads frames from (TCP transport)."""
    host = host or settings.STREAM_HOST
    port = port or settings.RTSP_PORT
    path = path or settings.STREAM_PATH
    return f"rtsp://{host}:{port}/{path}"


def lan_ip() -> str:
    """Best-effort LAN address of this machine, for the phone to publish to.

    A UDP socket is connected (no packet is sent) and its local address read back -
    that is the interface the operating system would use for outbound traffic, which
    is the one the phone on the same Wi-Fi can reach.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()
