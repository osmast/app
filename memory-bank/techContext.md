# Tech Context — stack, setup, constraints

## Runtime

| Item | Value |
| --- | --- |
| Language | **Python 3.13** (`requires-python >= 3.13`) |
| GUI | **PySide6 6.11** (Qt 6) |
| CV | **OpenCV (headless)**, NumPy |
| ML | **PyTorch 2.14.0** (conda-forge/defaults; **MPS available** on this M4) — `pytorch` is in `environment.yml`, imported **lazily** by `daphnia/vision/rtmdet/` |
| Store | **SQLite** via stdlib `sqlite3` (`zoo.db`) |
| Env manager | **conda**, environment name **`z-app`** |
| Tests | **stdlib `unittest`** (`make test`) — 1220 tests (1 skipped); measured 2026-10-22 |
| Package | `daphniascope` v0.1.0; `pyproject.toml` `dependencies = []` on purpose — conda owns deps |

## Commands

```bash
conda env create -f environment.yml     # once
conda activate z-app
python run.py

make run              # launch the desktop app
make test             # unittest discover -s tests -t . -v
make ui-check         # paint every surface offscreen, full screen and back
make db               # (re)initialise the SQLite database
make db-reset         # delete zoo.db (recreated on next launch)
make stream           # probe the configured MediaMTX stream
make stream-publish   # publish an ffmpeg test pattern to MediaMTX over SRT
make mediamtx-setup   # install config/mediamtx.yml system-wide
make mediamtx-run     # run MediaMTX with the repository config
make train-check      # prove PyTorch + MPS + OpenCV + a fwd/bwd pass work
make train-inspect    # describe the corpus datasets/bb, and its warnings
make train            # train RTMDet on datasets/bb -> models/rtmdet_daphnia.pt
make train-pose       # train RTMPose on datasets/kp -> models/rtmpose_daphnia.pt
make repair-boxes     # report boxes that leave their image (dry run; --apply writes)
make clean            # remove __pycache__ and zoo.db
```

## Launch path (implemented)

```
run.py | python -m daphnia
  -> daphnia.ui.app.main(argv)
       parse_launch(args)           surface key; 'hub' when none is given
       usage()                      --help lists the launch points
       create_app(argv[:1])         QApplication + name/org metadata
       paths.ensure_runtime_dirs()  datasets/, models/, recordings/
       init_db()                    schema.sql + migrate()  (idempotent)
       apply_theme(app)             daphnia/ui/theme.py + assets/theme/daphniascope.qss
                                    (light/dark tokens -> QPalette + QSS; the scheme
                                    follows the desktop, DAPHNIA_THEME pins it)
       app.aboutToQuit -> shared_service().stop    (QThread safety)
       hub.window_for(key)          HubWindow | MainWindow | SurfacePlanWindow
       app.exec()
```

CLI: `python run.py [hub|study|bench|dataset|optics]`, the hub by default. An
unknown key exits `2` with the usage on stderr. A surface whose window is not
registered in `hub.WINDOW_FACTORIES` opens its **specification** instead - never
another surface's window.

## Streaming (Ingest)

Phone publishes **SRT** -> MediaMTX -> republishes **RTSP** -> OpenCV reads.

| Setting | Default | Env var |
| --- | --- | --- |
| Stream host | `127.0.0.1` | `DAPHNIA_STREAM_HOST` |
| Stream path | `live/cam` | `DAPHNIA_STREAM_PATH` |
| SRT port | `8890` | `DAPHNIA_SRT_PORT` |
| RTSP port | `8554` | `DAPHNIA_RTSP_PORT` |
| Orientation | `landscape` | `DAPHNIA_STREAM_ORIENTATION` |
| Burst size | `5` | `DAPHNIA_BURST_SIZE` |
| Longest recording, frames | `300` | `DAPHNIA_RECORD_MAX_FRAMES` (Study 2's Start/Stop take: ~10 s, ~360 MB) |
| Calibration trust limit, hours | `12` | `DAPHNIA_CALIBRATION_MAX_AGE_HOURS` |
| Weak-calibration threshold, % | `2.0` | `DAPHNIA_CALIBRATION_WEAK_UNCERTAINTY_PCT` |
| Operator | OS user | `DAPHNIA_OPERATOR` |
| DB path | `app/zoo.db` | `DAPHNIA_DB` |
| Data root (what the app **writes**) | `app/` (the checkout) | `DAPHNIA_DATA_DIR` |
| MediaMTX binary | `mediamtx` | `DAPHNIA_MEDIAMTX_BINARY` |
| MediaMTX API host | `127.0.0.1` | `DAPHNIA_MEDIAMTX_API_HOST` |
| MediaMTX API port | `9997` | `DAPHNIA_MEDIAMTX_API_PORT` |
| FFmpeg decoder log | `-8` (quiet) | `DAPHNIA_FFMPEG_LOGLEVEL` |

Publish URL example: `srt://192.168.0.24:8890?streamid=publish:live/cam`
(Moblin iOS app, SRT mode, **H.264** — measured 2026-10-19: H.265/HEVC loses 7-21 of the
first 120 frames at 24.1 fps against 27.5 fps, and the viewer starts mid-GOP; H.264 loses
none).
Read URL: `rtsp://127.0.0.1:8554/live/cam`.
One shared `StreamService` (`shared_service()` singleton) is established **once**
on Setup; every other screen binds a preview to it — nothing reconnects per
screen. Full detail: `docs/streaming.md`.

## Portability (another machine, and packaging)

MediaMTX is a **single dependency-free executable on Windows as well as macOS**
(its README: *"Compatible with Linux, Windows and macOS"*; release assets
`mediamtx_v1.21.2_windows_amd64.zip` / `…_darwin_arm64.tar.gz`). `binary_path()` is
`shutil.which(settings.MEDIAMTX_BINARY)`, so on Windows `mediamtx.exe` on `PATH`
(or `DAPHNIA_MEDIAMTX_BINARY`) plus `mediamtx.exe config\mediamtx.yml` is the whole
install; inbound **UDP 8890** and **TCP 8554** must be allowed.

What is macOS-shaped here, and only *informational*: `scripts/setup_mediamtx.sh`,
`mediamtx.SYSTEM_CONFIG` (`/opt/homebrew/etc/mediamtx/…`, reported by `status()`),
`scripts/dev_publish_stream.sh` (bash + ffmpeg + srt-live-transmit) and the `make`
targets. `vision/images.py`'s `sips` path is guarded by `sys.platform != "darwin"`.
**PyTorch's MPS backend is Apple-only** — on Windows the detector and pose model run
on CPU, and `train.pick_device` *raises* rather than falling back, so latency there
is unmeasured until someone measures it.

**`DAPHNIA_DATA_DIR`** (`paths.DATA_ROOT`) is where the app writes `zoo.db`,
`recordings/`, `datasets/` and `models/`; it defaults to the checkout. It exists for
packaged builds — before it, only the DB could be moved and a frozen app would have
written inside its own (read-only / signature-breaking) bundle.

**Packaging status: nothing is built.** No `.spec`, no `make package`, no CI job, no
measured bundle size. PyInstaller's own rule decides the shape of the work: *"The
output of PyInstaller is specific to the active operating system and the active
version of Python … you run PyInstaller on that OS, under that version of Python"* —
so `.exe` on Windows, `.app`/`.dmg` on macOS, one build per OS **and** architecture
(`conda-pack` is the same). A bundle must carry `assets/ui/*.ui`,
`assets/theme/daphniascope.qss` and `daphnia/db/schema.sql` (all read from disk at
run time), and `pytorch` is optional because it is imported lazily — a build without
it loses only Predict. The full route, costs and caveats are in the README's
*Packaging* section.

## Calibration / distortion stack

The scale is the project's metric (`px_per_um`), so the tools around it are listed with
the same care as the ML stack, and the one library that is *in* the environment carries
its licence and its install note:

| Tool / library | Type | Licence | In `z-app`? | Taken for | Not taken for |
| --- | --- | --- | --- | --- | --- |
| **OpenCV** | library (C++, Python) | Apache-2.0 | **yes** (conda-forge) | reading frames, the app's own CV | `calibrateCamera`/`findCirclesGrid`: it needs *several* views of a target it can identify as a whole (measured 2026-10-19: the whole question here is what *one* photo can say) |
| **OpenCV.js** | library (WebAssembly) | Apache-2.0 | no (not used) | the same maths in a browser, `undistort` in a web app | — nothing depends on it today |
| **Discorpy 1.7.0** | Python library | **Apache-2.0** | **yes, optional pip** (`environment.yml`; imported *lazily*, like PyTorch) | radial + perspective distortion from a **single** image, by direct computation; the centre of distortion; its own sub-pixel residual as the acceptance test | anything the app imports at start-up - no surface imports it (`scripts/calibration_probe.py` does) |
| **ImageJ / FIJI** | desktop app (Java) | public domain/BSD | no | the independent bio-metrology referee for pixel->micron readings | — |
| **Micro-Manager** | desktop microscopy UI | BSD/LGPL (check) | no | what a calibration *record* should contain; hardware automation reference | as a dependency - built for a desktop microscope, not a phone clamped to an eyepiece |

`discorpy` pulls `scipy`, `scikit-image`, `h5py`, `Pillow` and `matplotlib`, **all
already present** in `z-app` (check with `conda list`), so it adds no second CV stack -
which is why it is acceptable while `label-studio`'s `opencv-python-headless` pin
(constraint 7) is not. Detail, licence, the flow, and what was measured on this
project's photos: **`docs/calibration.md`**.

> **Caveat against constraint 7, measured rather than waved away.** Constraint 7 says
> nothing third-party is `pip install`ed into `z-app`. `discorpy` is pip-only (it is on
> PyPI, not in conda-forge), so the exception has to be justified: verified 2026-10-19
> with `conda run -n z-app python -m pip install --dry-run discorpy` -
> **`Requirement already satisfied` for every dependency**, i.e. the transaction would
> install nothing but discorpy itself and cannot touch `numpy`/`opencv`. Re-run that
> command before any `conda env update -f environment.yml`; if it ever wants to change
> `numpy`/`opencv`, discorpy leaves `z-app` and gets its own environment, like
> `label-studio`.

## Hard technical constraints

1. **QSS is a CSS2 subset.** No layout properties, no CSS variables, no classes →
   no Tailwind on Qt widgets.
2. **QThread safety.** A `QThread` destroyed while running aborts the process;
   readers stay referenced (`streaming.reader._ACTIVE`) and are stopped on quit.
3. **SQLite concurrency.** `connect()` sets `PRAGMA foreign_keys = ON` only —
   **no WAL, no `busy_timeout`.** Separate processes for separate surfaces would
   hit "database is locked"; add `journal_mode=WAL` + `busy_timeout` first.
4. **`CREATE TABLE IF NOT EXISTS` never alters** an existing table → every new
   column needs a `COLUMN_MIGRATIONS` entry; `make db-reset` only for a pristine
   schema (e.g. to add the FK that `ALTER TABLE` cannot). A **constraint** change
   cannot be an `ALTER` at all (the implicit UNIQUE index cannot be dropped): rebuild
   the table by copying it, as `connection.rescope_corpus_labels()` does.
5. **OpenCV is imported lazily** so the Qt app still starts when CV deps are
   absent.
6. **calibration_profiles has one `is_active` row** — the global scale. A
   prerequisite to fix for multi-objective / small-organism work.
7. **Never `pip install` third-party tools into `z-app`.** conda-forge's
   `py-opencv` ships `opencv_python-*.dist-info` **without a `RECORD`**, so pip can
   never uninstall/replace OpenCV (`error: uninstall-no-record-file`) — and a pip
   transaction that dies mid-way leaves a half-migrated env. `numpy` is owned by
   conda-forge too: pip happily replaces it and conda stops tracking it
   (`conda list` shows `numpy … pypi_0`). Repaired on 2026-10-16 with
   `conda install -n z-app -c conda-forge numpy` (2.5.4, single-package plan).
8. **`z-app` has user site-packages enabled, and they come *first*.** `sys.path`
   is `… /z-app/…/site-packages` **after**
   `/Users/os/.local/lib/python3.13/site-packages`, so anything installed there is
   visible to the app and `pip check` reports it (e.g. `rich-click 1.9.8 requires
   rich` — that is esptool's copy, **not** a project defect). `pip install` in the
   env still installs into the env; `pip uninstall` resolves by path order, which
   makes the two easy to confuse.
9. **A conda environment can end up *mixed-channel*, and the ABI failures are
   confusing.** `z-app` drifted into a state where `opencv`/`libopencv`/`py-opencv`
   and `pytorch`/`libtorch` (`gpu_mps`) came from **pkgs/main** while
   `libprotobuf`, `libsqlite`, `freetype`, `harfbuzz` came from **conda-forge** —
   plus a pip-installed `protobuf`. The symptoms looked unrelated: `cv2`,
   `_sqlite3` and `import torch` each failed with a *missing symbol* naming a
   different library, and two conda packages claimed files (`libfreetype.6.dylib`,
   `libsqlite3.0.dylib`) that were not on disk. What to do:
   * read the failing symbol's **`Referenced from` / `Expected in`** pair — that
     names the two packages that disagree, and the fix is to make them come from
     one channel;
   * ``conda list -n z-app --show-channel-urls`` **is the first thing to run**;
   * if `conda install … --override-channels -c <channel>` says "all requested
     packages already installed", the version is already satisfied and conda will
     not switch the build: `conda remove --force -y <pkg>` then install it;
   * `--force` skips the dependent check, so **no dependency cascade** is triggered
     — which is exactly why it is the right tool here (and why it can leave a
     half-consistent env if it fails part-way).
   Repaired 2026-10-16 to `cv2 5.0.0`, `sqlite3 3.54.0`, `torch 2.14.0` (MPS).
10. **`environment.yml` is the source of truth for channels.** It declares
   `- conda-forge` only. Anything installed from `defaults` into `z-app` is drift
   waiting to become a bug — verify with `--show-channel-urls` after any install.

## External tools (never app dependencies)

Tools used *alongside* the app stay out of `z-app`, so the CV stack cannot be
downgraded by their pins:

- **Label Studio 1.23.2 (Community)** — installed as an isolated `uv` tool
  (`uv tool install label-studio --python 3.13` → binary
  `~/.local/bin/label-studio`, already on `PATH`). Start with
  `label-studio start --port 8080 --no-browser`; its data dir is
  `~/Library/Application Support/label-studio`. Verified 2026-10-16: `/health` →
  `200`, `/` → `302`.
  Added **only for use beside the app**, not wired into any surface.
  - It requires `opencv-python-headless>=4.12,<4.13` + `pyarrow`, which collide
    with `z-app`'s conda `py-opencv 5.0.0` — that conflict is *why* it is isolated
    (constraint 7).
  - Do **not** use `brew install humansignal/tap/label-studio`: it leans on the
    deprecated `python@3.10` and would upgrade `postgresql@14`, `sqlite`, `xz`,
    `readline`, `krb5` and `openssl@3` system-wide for a 240 MB bottle.

## Git / local state

`.gitignore` excludes `*.db`, `auto.crt`, `auto.key`, `*.log`, and the contents
of `datasets/{raw,processed,annotations}`, `recordings/*`, `models/*`
(folders kept via `.gitkeep`). `.clineignore` keeps agents off `data/`,
`outputs/`, `.git/`, `node_modules/`, caches.

## Test suite (`tests/`)

`test_workflow`, `test_study`, `test_ui_smoke`, `test_repository`, `test_db`,
`test_calibration`, `test_calibrations`, `test_grid`, `test_capture`,
`test_streaming`, `test_quality`, `test_protocols`,
`test_templates`, `test_maintenance`, `test_plates`, `test_units`,
`test_image_canvas`, `test_images`, `test_bench`, `test_surfaces`,
`test_annotation`, `test_dataset`, **`test_export`**, **`test_rtmdet`**,
**`test_predict`**, **`test_canvas_workspace`**, **`test_crop`**,
**`test_pose_corpus`**, **`test_rtmpose`**, **`test_individuals`**, **`test_lattice`**,
**`test_wells`**, **`test_study2`**, **`test_paths`** —
**1220 tests** (1 skipped on this machine; measured 2026-10-22), stdlib `unittest`, `make test`.

* `test_export` needs **no** third-party module: the export reader is pure stdlib,
  which is why the corpus contract is testable on any machine. **`test_pose_corpus`**
  is its pose twin for the same reason (`daphnia/vision/pose_corpus.py` is Qt- and
  torch-free). **`test_individuals`** is the same kind of test for the Bench's grouping
  rule (`daphnia/vision/individuals.py`): hand-computed containment, nesting and the
  midpoint, no Qt and no torch.
* **A gesture test must drive the gesture.** The crop loop's clicks were covered by a
  helper that called `canvas.set_points(...)` directly, so nothing noticed that a real
  press inside the crop selected the box instead of placing the keypoint
  (`SR-DATASET-028`). `DatasetBoxQueueTests._mouse`/`_click_view` and
  `EditAllTests._gesture` send real `QMouseEvent`s for exactly this reason; a paint rule
  ("a ghost", "two boxes") is checked by counting pixels (`canvas.grab()`).
* `test_rtmdet` and the model half of `test_predict` are `skipUnless(torch)` — a
  machine without PyTorch **skips** them instead of failing the suite
  (`SR-PREDICT-N02`). **`test_rtmpose`** is likewise `skipUnless(torch)`; its
  round-trip and oracle cases (the two bugs this project already paid for) are in it.
* **`test_predict`** also covers the **pose overlay and the scale** on the Predict
  surface: the Keypoints tool hiding without a model, the head→tail line and its length
  label reaching the canvas, both models loading through the one button, and the unit
  story (µm from the active calibration, px labelled *unscaled* without one, WEAK/STALE
  flags, and a scale switch that re-labels without re-running the models). It runs on a
  **private database** (`DAPHNIA_DB` in a temp dir) so the test's scale is the test's.
* `test_ui_smoke` asserts page/attribute inventory and zoom controls — **new
  surfaces must extend it** (the Predict tile and window tests were added there).
* `test_surfaces` guards the registry, the launch dispatch and the Qt-free domain
  (`PurityTests`), including the assertion that only the *five* built surfaces are
  `available`.

> **`make test` is the floor, not the whole story for ML code.** The ML discipline —
> which tests an ML change needs (oracle, round-trip, determinism, shape-contract), the
> empirical checks no unit test replaces, and the reporting format — is in
> **`memory-bank/mlContext.md`** §3-4 and §8.
