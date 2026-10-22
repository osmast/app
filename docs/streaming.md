# Live streaming (MediaMTX)

The phone streams to a MediaMTX server running on the Mac; the vision pipeline
reads the same stream over RTSP.

```
phone (SRT publish)  ->  MediaMTX  ->  RTSP read  ->  OpenCV pipeline
```

The phone publishes SRT; MediaMTX republishes the same path over RTSP, which is what
OpenCV reads. SRT retransmits lost packets, so a lossy link cannot deliver a
truncated H.264 frame.

## 1. Install and configure MediaMTX

```bash
brew install mediamtx                  # v1.21+ tested
./scripts/setup_mediamtx.sh            # installs config/mediamtx.yml system-wide
brew services start mediamtx           # or: make mediamtx-run
```

`setup_mediamtx.sh` backs up the existing
`/opt/homebrew/etc/mediamtx/mediamtx.yml` with a timestamp suffix before copying
the repository version. Override the target directory with
`MEDIAMTX_CONFIG_DIR=... ./scripts/setup_mediamtx.sh`.

The configuration in `config/mediamtx.yml` keeps the ports aligned with
`daphnia/settings.py`:

| Protocol | Port | Used for |
| -------- | ---- | -------- |
| SRT | 8890 (UDP) | **the phone publishes here** (`?streamid=publish:live/cam`) |
| RTSP | 8554 | OpenCV reads frames (TCP transport) |
| API | 9997 | control API, localhost only |

## 2. Publish from the phone (SRT)

Install an SRT broadcaster. **[Moblin](https://apps.apple.com/app/id6466745933)** is
free, MIT-licensed, on the App Store, and publishes SRT. Set it to SRT so lost
packets are retransmitted instead of corrupting the frame:

1. *Streams* -> add a stream -> *Custom* -> **SRT(LA)**.
2. Paste the publish URL the Setup step shows, for example:

   ```
   srt://192.168.0.24:8890?streamid=publish:live/cam
   ```

   The `streamid` encodes the action (`publish`) and the path (`live/cam`); Moblin
   reads it from the URL automatically.
3. Video codec **H.264** (best compatibility with MediaMTX and OpenCV).
4. Lock the optics before every session - see *Advanced camera settings* below.

Find the Mac's LAN IP with `ipconfig getifaddr en0`.

No phone at hand? Publish a test pattern from the Mac over the same SRT path:

```bash
make stream-publish          # ffmpeg test pattern -> srt://127.0.0.1:8890?streamid=publish:live/cam
```

### Advanced camera settings (Moblin)

Anything that changes the optics invalidates the calibration preset, so re-verify
these at the start of every session:

* **Fixed zoom - 1.0x** (matching `calibration_profiles.camera_zoom_ratio`):
  *Settings -> Camera -> Zoom*, add a back-camera preset with an exact **X = 1.0**,
  then tap that preset on the live screen.
* **Autofocus off + manual focus lock** (`focus_mode = 'infinity_locked'`): enable
  *Settings -> Camera -> Camera controls*, then on the live screen tap **FOC**,
  drag the slider to the far end (infinity) and tap the **lock** icon so autofocus
  is disabled. Turn *Settings -> Camera -> Tap screen to focus* **off** so a stray
  tap cannot refocus.
* Optional but recommended: lock **EXP/ISO** and **WB** the same way for a
  repeatable exposure.

## 3. Read it back

```bash
make stream                  # probe: resolution, fps, read rate
make stream                  # or: python scripts/check_stream.py --save frame.png
```

From Python:

```python
from daphnia.streaming import StreamReader, read_rtsp_url

reader = StreamReader(read_rtsp_url())     # rtsp://127.0.0.1:8554/live/cam
reader.frame_ready.connect(handle_frame)   # numpy.ndarray, BGR
reader.start()
...
reader.stop()
```

## Configuration

`daphnia/settings.py` reads environment variables so nothing is hard-coded:

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `DAPHNIA_STREAM_HOST` | `127.0.0.1` | host used to build URLs |
| `DAPHNIA_STREAM_PATH` | `live/cam` | ingest path |
| `DAPHNIA_SRT_PORT` | `8890` | SRT port (the phone publishes here) |
| `DAPHNIA_RTSP_PORT` | `8554` | RTSP port (OpenCV reads here) |
| `DAPHNIA_BURST_SIZE` | `5` | frames per well in a burst |
| `DAPHNIA_RECORD_MAX_FRAMES` | `300` | longest a **Study 2 recording** may run (frames) before it stops itself and stores what arrived - ~10 s at 30 fps, ~360 MB |
| `DAPHNIA_MEDIAMTX_API_HOST` | `127.0.0.1` | control API, asked what codec is published |
| `DAPHNIA_MEDIAMTX_API_PORT` | `9997` | control API port |
| `DAPHNIA_FFMPEG_LOGLEVEL` | `-8` | FFmpeg's own decoder log (`-8` = quiet; e.g. `32` for `AV_LOG_WARNING`) |

## Troubleshooting

* **Phone cannot connect** — phone and Mac must be on the same network, and the
  macOS firewall must allow incoming connections for `mediamtx` (**UDP 8890** for
  SRT).
* **SRT publishes but nothing shows up** — the stream id must be `publish:<path>`
  and the path must match `DAPHNIA_STREAM_PATH` (default `live/cam`). Check with
  `curl -s http://127.0.0.1:9997/v3/paths/list`.
* **`error while decoding MB ... bytestream -N`** — frames are arriving corrupted.
  Check the SRT stats (`curl -s http://127.0.0.1:9997/v3/srtconns/list`): rising
  `packetsReceivedRetrans` / `packetsReceivedDrop` and a climbing RTT mean the link
  is out of headroom — lower the phone's bitrate or raise its SRT latency.
* **Stream opens but no frames** — check `mediamtx` is running
  (`make stream` prints the status) and that the publisher path matches
  `DAPHNIA_STREAM_PATH`.
* **`[hevc @ ...] Could not find ref with POC ...` / `Error constructing the frame RPS`**
  — the phone is publishing **H.265/HEVC**. Measured 2026-10-19 through
  SRT → MediaMTX → RTSP at 1280x720@30, joining a running stream: H.265 loses **7 to 21
  of the first 120 frames** (the reader joins mid-GOP, so the decoder is handed frames
  whose reference pictures never arrived) and decodes at **24.1 fps against 27.5 fps for
  H.264, which loses none**. OpenCV still answers `True` for every read - it repeats the
  last good frame - so this is a picture problem, not a crash; but a burst captured in
  that first second can hold duplicates.
  **Set the phone's video codec to H.264** (*Streams → your stream → Video codec* in
  Moblin). The app now says it for you: the status bar reads
  `Camera live  (the publisher sends H265 - set the phone's video codec to H.264)`, and
  `make stream` prints the codec and the same advice. FFmpeg's own log is quiet by
  default (`DAPHNIA_FFMPEG_LOGLEVEL`) so this flood no longer reaches the terminal.
* **A read fails and the preview says "no frames arriving" for a moment** — that is
  normal, not a disconnection: a decode failure and a moment with nothing buffered both
  return `False`, and the frames after them are fine. `daphnia.streaming.reader` retries
  and counts them (the bar shows `N frame(s) could not be decoded`) and only declares the
  stream lost after `MAX_FAILED_READS = 5` failed reads in a row (`SR-CORE-020`).
* **`QThread: Destroyed while thread is still running` / the app aborts on exit** —
  the reader thread was still running when Qt destroyed it. The app now keeps the
  reader referenced and stops it on quit; make sure you are on the current code.
* **`OpenCV is not available`** — the reader imports `cv2` lazily; install the
  dependencies with `make env-update`.
