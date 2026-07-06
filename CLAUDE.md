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

## Known issue: mapping overfits (fix pending)

The degree-2 polynomial map has 35 features but only 33 calibration samples,
with Ridge alpha=1.0 - it nearly interpolates training (residual ~60 px) yet
extrapolates wildly (evaluate ~1350 px, ~16% of in-distribution predictions
land off-screen). The screen clamp then folds runaway predictions onto the
edges, which makes corner targets look accurate and INVERTS the zone pattern
(center worse than corners). Serialization and feature construction were
ruled out (live-vs-reloaded roundtrip is bit-identical; calibrate/evaluate
share `GazePipeline.features()`). The fix (regularization / fewer features /
drop unstable head-position terms) is a LAYER-2 model change, pending
explicit go-ahead. Until then:

- `evaluate` measures error on RAW (unclamped) predictions and prints a
  per-target target-vs-raw-prediction table plus the off-screen fraction, so
  a runaway model cannot hide behind the clamp.
- `demo`/`record`/`evaluate` print which calibration they loaded (path +
  created timestamp) and warn if camera.yaml/screen.yaml changed afterwards.

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
