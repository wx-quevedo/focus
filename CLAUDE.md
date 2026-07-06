# gazetrack — Phase 1: webcam gaze tracking MVP

Real-time on-screen dot that follows the user's gaze. Spec: `PHASE-01-gaze-mvp.md`
(read it before touching architecture). Hardware is fixed: Logitech C920 at
1920x1080@30 (MJPG, autofocus OFF, auto-exposure OFF), laptop RTX 4090, Windows,
CUDA inference only.

## Architecture rule (strict, never mix layers)

- **Layer 1 — gaze direction** (`gaze/`): ETH-XGaze ResNet-18 on CUDA. Input:
  normalized face crop (XGaze normalization, 600 mm / 960 px focal / 224 px).
  Output: yaw/pitch in degrees, camera space. Owns MediaPipe face mesh, head
  pose (PnP), and all model preprocessing. Knows nothing about screens.
- **Layer 2 — screen mapping** (`mapping/`): scikit-learn
  `PolynomialFeatures(degree=2)` + Ridge with `sample_weight` edge weighting.
  Learns `(yaw, pitch, head_pose) -> (screen_x, screen_y)` per user/session.
  Persisted to YAML in `data/`. Never calls the CNN or touches frames.

Do not put screen logic in layer 1 or model logic in layer 2. WebGazer-style
regression as layer 1 is explicitly forbidden (caps at 5-6 deg).

## Environment

- Venv: `.venv` (Python 3.10.6). Run everything with `.venv\Scripts\python.exe`
  from the project root (imports are package-relative to the root).
- Pinned deps in `requirements.txt` (torch 2.5.1+cu121). Keep versions pinned;
  new deps go in pinned.
- Model weights: `models/eth-xgaze_resnet18.pth` (auto-downloaded, SHA-256
  verified in `gaze/eth_xgaze.py`).
- Camera hardware (measured, not assumed): the actual device is a
  **Logitech c922 Pro Stream Webcam at index 1** (index 0 is the laptop's
  integrated webcam; Quest/OBS virtual cameras also enumerate and can
  shuffle indices). `data/camera.yaml` persists index/focus/exposure,
  written by `python -m capture.camera --focus-sweep`. Precedence: CLI
  flag > config > default.
- Capture strategy (empirical, do not "simplify" back): **MSMF with
  width/height/fps params and NO FOURCC** is the only OpenCV path that
  delivers 1080p30 on this camera - it negotiates MJPEG internally. DSHOW
  always lands on YUY2\@1080p which the UVC descriptor caps at 5 fps, and
  MSMF fails to open if CAP_PROP_FOURCC is passed. MSMF readbacks for
  FOURCC/focus/exposure are garbage while its writes DO work; measured fps
  is the ground truth. Autofocus is disabled, so manual focus MUST be
  tuned once with the sweep (0-255 step 5, 0=infinity, ~60 cm lands in
  25-50) or the image stays blurred. A long exposure (>= -4) also produces
  the ~5 fps signature; `verify()` prints the diagnosis.
- `python check_env.py` must pass before debugging anything else; a CPU-only
  torch build is a hard failure.

## Screen geometry (physical, required for cm/deg)

- `data/screen.yaml` holds `width_cm` / `height_cm` / `distance_cm`, set via
  `python main.py screen --screen-width-cm <W> --screen-height-cm <H>
  --distance-cm <D>`. calibrate/demo/evaluate read it automatically;
  precedence flag > screen.yaml > unset. There is NO silent cm default — an
  unconfigured run is reported as such and cm/deg are omitted (px and
  %-of-width stay valid). The real screen resolution is auto-detected
  (DPI-aware) and printed at startup.
- `evaluate` reports error in px, %-of-screen-width (the cross-monitor
  metric), and — only when geometry is set — cm and deg. Test rig here is a
  49" ultrawide (~120 cm wide), detected at 3840x1080; the old 34.4 cm laptop
  default made every cm/deg figure wrong, hence the persistent config.

## Commands

- `python check_env.py [--camera]` — environment + CUDA + camera verification.
- `python -m capture.camera --preview` — verify C920 mode (it silently falls
  back to 720p; MJPG must be set before resolution to get 1080p@30).
- `python -m gaze.mediapipe_head` — landmarks/head-pose visual check.
- `python -m gaze.eth_xgaze [--show-patch]` — live gaze inference check.
- `python main.py screen|calibrate|demo|record|evaluate` — main CLI.
  Calibration protocol: look at each point NATURALLY (move head and eyes as
  usual), hold steady while the ring contracts. On a wide monitor head
  motion is unavoidable and is a mapping feature; consistency between
  calibration and use matters more than rigidity. `calibrate` prints
  per-target residuals, head yaw/pitch coverage, and flags targets with high
  inter-frame variance (user moved) or high residual (garbage points).

## Conventions (permanent, apply to all future phases)

- Code and comments in English. CLI/console text in English.
- NO emojis anywhere: not in UI, CLI output, logs, README, commit messages or
  any future executable/web interface. All visual interfaces must look
  professional and sober: clean typography, neutral palette (dark gray panels,
  off-white text), no playful decorations or childish tones.
- Angle conventions are documented in `gaze/mediapipe_head.py` and
  `gaze/eth_xgaze.py` docstrings; keep new code consistent with them
  (yaw positive = subject's right, pitch positive = up, "left/right" always
  means the subject's left/right).
- `_PNP_MODEL_POINTS` MUST stay in real millimetres (adult anthropometric
  means). ETH-XGaze normalization re-projects the face to a fixed 600 mm /
  960 px camera; an oversized model makes solvePnP report a huge distance
  and the crop magnifies until only nose+mouth fit, cutting the eyes out and
  killing the gaze signal (head pose still works, gaze looks "stuck"). PnP
  uses `SOLVEPNP_SQPNP`, not ITERATIVE: a near-frontal face is near-planar
  and ITERATIVE flips to the mirror branch (tvec_z < 0, roll ~180). Do not
  revert either. The live check prints a PnP distance sanity line
  (plausible desktop band 400-1000 mm) as the canary for a bad model scale.
- Out of scope for phase 1 (do NOT implement): concentration/productivity
  score, active-window tracking, heatmaps, dashboards.

## PHASE-02 productivity (2A/2B done; 2C gated)

Spec: `PHASE-02-productivity.md`. Delivery is gated: 2A -> user works over real
apps -> 2B -> review activity.db over >=1h -> 2C. Permanent rules: 100% local;
keyboard/mouse activity is COUNTS per interval only, never key content
(keylogging forbidden - documented in `activity_logger` on_press); `--no-titles`
flag; no emojis anywhere.

- **2B done** - `python main.py record --background`. Overlay (visible or
  `--hide-overlay`) + behavior logging to `data/activity.db`, in the tray with
  a Pausar/Reanudar registro toggle. Consumer of the SAME GazeWorker (one
  camera): the worker now emits `sampled(GazeSample)` per frame in addition to
  the cursor signal.
  - `overlay/activity_db.py`: SQLite **WAL from first write**, synchronous
    NORMAL; each 5 s interval is its own committed transaction, so the app's
    hard `os._exit` loses at most the in-progress interval. On open, a session
    left without an end (previous kill) is closed to its last sample. Schema:
    `sessions(id,start,end,notes)`, `samples(ts, session_id, gaze_zone,
    on_screen, tracking_state, face_present, process, window_title, kb_events,
    mouse_events)`, index on ts. Measured ~1.8-2.3 MB/day (5 s intervals).
    `SAMPLE_QUERIES` holds the review queries.
  - `overlay/activity_logger.py`: aggregates the worker's per-frame samples +
    a 1 Hz foreground-window sample (win32+psutil) + pynput keyboard/mouse
    COUNTERS into one 5 s row. gaze_zone from a `grid_cols x grid_rows` grid
    (config, default 6x3, editable in the panel's Registro section).
    `--no-titles` stores process only. on_screen/face_present are interval
    FRACTIONS; gaze_zone/tracking_state/process are the interval MODE.
  - Robustness: the worker reconnects the camera on grab failure with backoff
    and records the gap as `tracking_state=camera_lost` (no crash, no fake
    data). Note (hardware): rapid open/close cycling can wedge the C920/c922 so
    `cv2.VideoCapture` blocks for tens of seconds on the next open - the logger
    correctly records camera_lost meanwhile.
  - Deps added: psutil 6.0.0.

- **2A done** - `python main.py overlay`. Transparent, click-through,
  always-on-top gaze cursor over real apps (`overlay/gaze_overlay.py`). It is
  a CONSUMER of the pipeline: reuses `GazePipeline.sample()`, `GazeMapper`,
  and `PointSmoother` (One-Euro). Layers 1/2 untouched except one metadata
  field (`head_yaw/pitch_range_deg`) added to `fit_combined` so the overlay
  can fade outside the calibrated head-pose range.
  - Window: PyQt6 (`WindowTransparentForInput` + `WA_TranslucentBackground` +
    Tool/topmost) with a win32 touch-up adding WS_EX_LAYERED | TRANSPARENT |
    TOPMOST | NOACTIVATE (NOACTIVATE = never steal focus). Render is
    dirty-rect only (`update(QRect)` around the cursor) - never a fullscreen
    30 fps repaint.
  - Gaze cursor: soft ring + halo + fill, neutral off-white, parametrizable
    (`--dot-size/-opacity/-color`). One-Euro + `--deadzone` (12 px default)
    for stillness in fixations. Fade (250 ms) on face loss, on head pose
    beyond the calibrated range +`--pose-margin` (fixed +-45/30 fallback if
    the calibration lacks the range), and when the prediction lands
    >`--offscreen-margin` (10% width) off-screen.
  - Worker on a `QThread` (camera+CNN off the UI thread); global hotkeys via
    `pynput.GlobalHotKeys` (Ctrl+Alt+G show/hide, Ctrl+Alt+S settings,
    Ctrl+Alt+Q quit) marshalled to the UI thread by a Qt signal;
    `QSystemTrayIcon` (code-drawn ring, no asset) with Mostrar/Ocultar,
    Configuracion, Salir.
  - Shutdown: quit() must NOT call pynput's hotkey stop() (hangs from another
    thread on Windows) nor QThread.wait(msec) (PyQt6 int overload ignored the
    timeout and blocked). It calls app.quit() to break exec(), then run()
    does `os._exit()` - native camera/CUDA/hook threads do not unwind cleanly
    and would hang the process; the OS reclaims the camera on exit.
  - **2A.1 settings panel** (`overlay/settings_dialog.py`, `overlay/config.py`):
    a normal (focusable) QDialog opened from the tray or Ctrl+Alt+S. Live edit
    of cursor / smoothing / fade / (read-only) hotkeys; every change applies
    immediately via `OverlayApp.apply_config` (window/worker/smoother setters,
    pipeline untouched) and persists debounced to `data/overlay.yaml`.
    Precedence flag > overlay.yaml > default (`overlay.config.resolve`). Spanish
    UI, "Restaurar valores por defecto". The config dataclass is meant to grow
    (2B/2C add logging/alarm sections).
  - Measured: gaze pipeline ~1.7 cores (~5% of the 32-core machine) at
    ~28 fps; GPU ~20% util / ~3.3 GB (model + torch). The overlay's OWN cost
    (dirty-rect repaint) is marginal. Exceeds the spec's aspirational
    <15%/core because real-time FaceMesh+CNN is inherently that costly - it is
    layer-1/2 cost, not the UI. Click-through limitation: elevated (admin)
    windows are higher integrity and draw OVER the overlay (documented, not a
    bug).
  - Deps added: PyQt6 6.7.1, pywin32 306, pynput 1.7.7.

## PHASE-01B precision iteration (in progress)

Spec: `PHASE-01B-pursuit.md`. Baseline 2026-07-05: 246 px / 6.41% / ~5.85 deg,
evaluate ~= LOOCV (healthy). Two fronts: smooth-pursuit calibration data and
killing the right-side regional degradation.

- **(a) done** - evaluate prints a 3x3 spatial region grid (`eval/grid.py`,
  `_region_stats`) and flags any region above `REGION_FLAG_RATIO` (1.5x) the
  median region error; stored under `regions` in the report.
- **(b)+(c)+(d) done** - `python main.py calibrate --pursuit` now runs a
  COMBINED session: discrete grid first (fresh anchors), then smooth pursuit,
  then a degree/alpha model chosen by block-CV. `--pursuit-csv <file>` reuses
  a saved pursuit stream instead of recollecting (needs the `.meta.yaml`
  sidecar, else assumes defaults + current geometry).
- Pursuit ball (`mapping/pursuit.py`): fixed polyline (serpentine, perimeter
  and 2 diagonals, both directions per axis) at a DELIBERATELY slow 350 px/s
  (~8-9 deg/s on this rig; do NOT raise it thinking the target is 30 deg/s).
  Quality filter drops blinks, saccades (gaze speed > max(3x target,
  80 deg/s)) and 300 ms after each direction change (~8% discard observed).
  The latency lag (grid 0-450 ms) is chosen by block-CV with the target
  evaluated ANALYTICALLY at `position_at(t - lag)` (not nearest-frame).
- **Latency**: the measured optimum lag was 250 ms. It is NOT frame buffering
  (MSMF ignores CAP_PROP_BUFFERSIZE, verified, and has no backlog) and NOT
  our processing (grab->features is ~6 ms + ~4 ms CNN). It is inherent
  webcam capture + display latency, which the lag compensation absorbs (so it
  does not hurt calibration accuracy). Pursuit sample times are stamped at
  the camera GRAB (`Camera.last_grab_time`, `GazePipeline.features_ts`), not
  after processing.
- **Block-CV is mandatory** (`calibrator.block_kfold_indices` /
  `block_cv_error_px` / `_combined_block_cv`): contiguous temporal folds,
  never random - consecutive pursuit samples are autocorrelated and random
  k-fold would report an optimistic CV error. The combined CV holds out
  pursuit BLOCKS while keeping the discrete anchors in every training fold.
- Model selection (`select_model_block_cv`): degrees 2 and 3 x ALPHA_GRID,
  pick the min block-CV; both degrees' CV are reported. `FEATURE_VERSION=3`
  (grid+pursuit format) - old calibrations rejected on load. Guardrails
  (feature/sample ratio, raw-unclamped evaluate, calibration traceability,
  3x3 region residual in the calibration report) all intact.

## Mapping model (layer 2) - overfit fix applied 2026-07-05

Feature vector is `[gaze_yaw, gaze_pitch, head_yaw, head_pitch]` (v2,
`FEATURE_VERSION=2`). Absolute face position (face_x/y/z) was REMOVED: it
carried large, unstable coefficients (face_x coef ~-218 with 6.9 mm std) that
made predictions run away with normal between-session head drift. The old
7-feature vector gave 35 polynomial terms for 33 samples and overfit
(training residual ~60 px, evaluate ~1350 px, ~16% of predictions off-screen,
inverted zone pattern from the screen clamp folding runaway predictions onto
the edges). Now 4 inputs -> 14 degree-2 terms / 33 samples (ratio 0.42).

- Ridge alpha is chosen by leave-one-out CV over `ALPHA_GRID` (0.1..1000).
  Calibrate reports the chosen alpha and the LOOCV error - trust that, not the
  in-sample training residual, as the real-accuracy predictor.
- Guardrail: calibrate warns if polynomial-terms / samples exceeds
  `MAX_FEATURE_SAMPLE_RATIO` (0.5) - the canary if features or degree grow.
- Old calibrations are rejected on load by `feature_version`/`feature_names`
  with a clear recalibrate message (the feature set changed).
- `evaluate` measures error on RAW (unclamped) predictions and prints a
  per-target target-vs-raw-prediction table plus the off-screen fraction, so
  a runaway model cannot hide behind the clamp.
- `demo`/`record`/`evaluate` print which calibration they loaded (path +
  created timestamp) and warn if camera.yaml/screen.yaml changed afterwards.
- Serialization and feature construction were ruled out as causes
  (live-vs-reloaded roundtrip is bit-identical; calibrate/evaluate share
  `GazePipeline.features()`).

## Phase 1 status

- Complete: steps (a)-(i). Layer 1 verified live by the user; layer 2
  (calibration, One-Euro smoothing, dot overlay, evaluation, CLI) implemented
  and smoke-tested headless (mapper fit + YAML round trip, outlier rejection,
  filter behaviour, CLI wiring).
- Calibration: 33 targets (5x5 grid at 8% margin, weight 1.0 center / 1.5
  border, plus 8 edge/corner extras at 2.5% margin, weight 2.0), per target
  0.6 s settle then 0.4 s collection, MAD outlier rejection, averaged sample
  per target. Persisted to `data/calibration.yaml` (plain lists, no pickle).
- Feature vector (order is a contract, see `FEATURE_NAMES` in
  `mapping/calibrator.py`): gaze yaw/pitch, head yaw/pitch, face center
  x/y/z mm. Changing it invalidates saved calibrations (load() checks).
- Not yet done (future phases): smooth-pursuit calibration samples,
  transparent click-through overlay (current dot draws on an opaque
  fullscreen OpenCV window), SQLite recording (CSV only).
