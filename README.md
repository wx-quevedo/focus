# gazetrack

Webcam gaze tracking: an on-screen dot that follows the user's gaze in real
time. Logitech C920 (1920x1080@30) + ETH-XGaze on CUDA (RTX 4090), with a
per-session polynomial screen mapping calibrated on a 33-target routine.

Full phase spec: `PHASE-01-gaze-mvp.md`.

## Setup

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe check_env.py --camera
.venv\Scripts\python.exe -m capture.camera --focus-sweep
```

Model weights (44.8 MB) download automatically on first use and are verified
by SHA-256. `check_env.py` must pass; a CPU-only PyTorch build is a hard
error and the script prints the exact reinstall command.

The focus sweep is a one-time step (repeat it if the camera or seating
distance changes): autofocus is disabled for stability, so the lens needs an
explicit manual focus value. Center your face, adjust with W/S (focus) and
E/D (exposure) until the sharpness bar peaks and the image is crisp, then
press ENTER. The chosen values persist to `data/camera.yaml` and every mode
reads them automatically (explicit CLI flags still win). Every camera mode
prints the real runtime properties at startup (decoded FOURCC, fps reported
and measured, exposure, focus); trust the measured fps over the FOURCC
readback, which is unreliable under DirectShow.

## Commands

All commands run from the project root with the venv interpreter.

```bat
.venv\Scripts\python.exe main.py screen --screen-width-cm 120 --screen-height-cm 34 --distance-cm 70
.venv\Scripts\python.exe main.py calibrate    # 33-target calibration (5x5 grid + edge/corner extras)
.venv\Scripts\python.exe main.py demo         # live gaze dot, One-Euro smoothed
.venv\Scripts\python.exe main.py record       # log a session to data/session_*.csv
.venv\Scripts\python.exe main.py evaluate     # 5x5 grid accuracy report by zone
```

Set your physical screen geometry once with `screen` (values persist to
`data/screen.yaml`; the real resolution is auto-detected). Then the flow is
**calibrate**, then **demo**, then **evaluate**. Recalibrate whenever lighting
or sitting position changes. `--index N` selects the camera if it is not
device 0.

During calibration, look at each point **naturally** — move your head and
eyes as you normally would — and hold steady while the ring contracts. On a
wide monitor, head movement is expected and is used as a mapping feature.

`evaluate` reports mean error in px, as a percentage of screen width (the
metric comparable across monitors), and — when geometry is configured — in
cm and degrees, split by zone (center / edges / corners). Without geometry
it still reports px and %, and says so. Peripheral zones are expected to be
worse than the center; the phase target is a mean error of 3.5 degrees or
better in the center zone.

## Component checks

```bat
.venv\Scripts\python.exe -m capture.camera --preview   # C920 at 1080p/MJPG/30
.venv\Scripts\python.exe -m gaze.mediapipe_head        # landmarks + head pose
.venv\Scripts\python.exe -m gaze.eth_xgaze             # live gaze inference
```

## Known limitations (inherent, documented in the spec)

- Head movement degrades accuracy even though ETH-XGaze compensates
  partially; keep a stable posture for best results.
- Glasses with reflections interfere with eye appearance.
- Lighting changes require recalibration.
- Edge and corner accuracy is worse than the center on any RGB-webcam
  system; the evaluation quantifies it rather than hiding it.
