"""MediaPipe Face Mesh wrapper: landmarks, iris centers and head pose.

This module is part of layer 1 (gaze direction). It produces the raw face
geometry that the gaze model needs: 478 pixel-space landmarks (468 mesh +
10 iris), iris centers, and the head pose (rotation + translation in camera
coordinates) solved via PnP against a generic 3D face model.

The generic face model is defined in millimetres but is not the user's
actual face, so absolute distances carry a scale error. That is acceptable:
layer 2 (screen mapping) is calibrated per user/session and absorbs the
constant bias; what matters here is frame-to-frame consistency.

Conventions:
- Camera frame: x right, y down, z forward (OpenCV).
- "left"/"right" in attribute names refer to the SUBJECT's left/right,
  not image left/right.
- Euler angles in degrees: pitch positive = head tilting up, yaw positive =
  head turning to the subject's left, roll positive = head tilting toward
  the subject's right shoulder (derived from the OpenCV camera frame).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh landmark indices (with refine_landmarks=True, 478
# points total; 473..477 = subject-left iris, 468..472 = subject-right iris).
NOSE_TIP = 1
CHIN = 152
RIGHT_EYE_OUTER = 33      # subject's right eye, outer corner (image left)
RIGHT_EYE_INNER = 133
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263      # subject's left eye, outer corner (image right)
MOUTH_RIGHT = 61          # subject's right mouth corner (image left)
MOUTH_LEFT = 291          # subject's left mouth corner (image right)
IRIS_RIGHT_CENTER = 468
IRIS_LEFT_CENTER = 473

# 3D face model in REAL millimetres (adult anthropometric means). The model
# frame is: x toward subject's left, y up, z toward camera; the nose tip is
# the origin and the recessed features have negative z.
#
# The scale MUST be real millimetres, not arbitrary units. The ETH-XGaze data
# normalization re-projects the face to a virtual camera at a fixed 600 mm
# distance with a 960 px focal length, then crops 224 px. That geometry only
# frames the whole face (eyes included) if solvePnP reports a physically
# correct distance, which in turn requires a real-mm model: the pose scale is
# fixed by the model, so an oversized model makes solvePnP place the face far
# away and the normalization magnifies the crop until only the nose and mouth
# fit, cutting the eyes out and destroying the gaze signal. The startup
# distance sanity check in the live gaze estimator guards against this.
# Values from standard facial anthropometry: outer canthal (ex-ex) width
# ~90 mm, mouth (ch-ch) width ~50 mm, nose tip to gnathion ~65 mm.
_PNP_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # nose tip (pronasale)
    (0.0, -65.0, -13.0),    # chin (gnathion)
    (-45.0, 33.0, -28.0),   # subject-right eye outer corner (exocanthion)
    (45.0, 33.0, -28.0),    # subject-left eye outer corner
    (-25.0, -28.0, -23.0),  # subject-right mouth corner (cheilion)
    (25.0, -28.0, -23.0),   # subject-left mouth corner
], dtype=np.float64)

_PNP_LANDMARK_IDS = [NOSE_TIP, CHIN, RIGHT_EYE_OUTER, LEFT_EYE_OUTER,
                     MOUTH_RIGHT, MOUTH_LEFT]

# Plausible eye-to-camera distance for desktop use, in millimetres. Outside
# this band the PnP scale is almost certainly wrong (see the model comment
# above); the live estimator warns so the regression is not silently trained.
PLAUSIBLE_DISTANCE_MM = (400.0, 1000.0)

# Points averaged (after posing the model) to define the face center used by
# the gaze normalization: eye corners + mouth corners, per ETH-XGaze practice.
_FACE_CENTER_MODEL_POINTS = _PNP_MODEL_POINTS[2:]


def make_camera_matrix(frame_width: int, frame_height: int) -> np.ndarray:
    """Approximate intrinsics: fx = fy = width, principal point at center.

    Good enough for an uncalibrated webcam; per-session screen calibration
    absorbs the residual error.
    """
    return np.array([
        [frame_width, 0.0, frame_width / 2.0],
        [0.0, frame_width, frame_height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


@dataclass
class FaceObservation:
    """Single-frame face geometry in pixel and camera coordinates."""

    landmarks: np.ndarray          # (478, 2) float32, pixels
    iris_left: np.ndarray          # (2,) pixels, subject's left eye
    iris_right: np.ndarray         # (2,) pixels, subject's right eye
    rvec: np.ndarray               # (3, 1) Rodrigues rotation, model->camera
    tvec: np.ndarray               # (3, 1) translation, millimetres
    rotation: np.ndarray           # (3, 3) rotation matrix, model->camera
    euler_deg: np.ndarray          # (3,) pitch, yaw, roll in degrees
    face_center: np.ndarray        # (3,) camera coords, millimetres

    @property
    def head_pitch(self) -> float:
        return float(self.euler_deg[0])

    @property
    def head_yaw(self) -> float:
        return float(self.euler_deg[1])

    @property
    def head_roll(self) -> float:
        return float(self.euler_deg[2])


def _euler_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Head pitch/yaw/roll in degrees from the model->camera rotation."""
    # The model frame is y-up/z-toward-camera while the camera frame is
    # y-down/z-forward; the fixed flip diag(1,-1,-1) aligns them so that an
    # identity result means a frontal head.
    aligned = rotation @ np.diag([1.0, -1.0, -1.0])
    sy = np.hypot(aligned[0, 0], aligned[1, 0])
    if sy > 1e-6:
        pitch = np.arctan2(-aligned[2, 1], aligned[2, 2])
        yaw = np.arctan2(aligned[2, 0], sy)
        roll = np.arctan2(-aligned[1, 0], aligned[0, 0])
    else:
        pitch = np.arctan2(-aligned[1, 2], aligned[1, 1])
        yaw = np.arctan2(aligned[2, 0], sy)
        roll = 0.0
    return np.degrees([pitch, yaw, roll])


class HeadTracker:
    """Face Mesh + iris + head pose for a single face."""

    def __init__(self,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.6):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # adds the 10 iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._camera_matrix: np.ndarray | None = None
        self._frame_size: tuple[int, int] | None = None

    def camera_matrix(self, frame_width: int,
                      frame_height: int) -> np.ndarray:
        if self._frame_size != (frame_width, frame_height):
            self._camera_matrix = make_camera_matrix(frame_width,
                                                     frame_height)
            self._frame_size = (frame_width, frame_height)
        return self._camera_matrix

    def process(self, frame_bgr: np.ndarray) -> FaceObservation | None:
        """Detect the face in a BGR frame. Returns None if no face found."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        raw = result.multi_face_landmarks[0].landmark
        landmarks = np.array([(lm.x * w, lm.y * h) for lm in raw],
                             dtype=np.float32)

        image_points = landmarks[_PNP_LANDMARK_IDS].astype(np.float64)
        camera_matrix = self.camera_matrix(w, h)
        # SQPNP, not ITERATIVE: a near-frontal face is nearly planar, which
        # gives solvePnP a two-fold mirror ambiguity. ITERATIVE has no global
        # view and readily converges to the flipped branch (face behind the
        # camera, tvec_z < 0, roll ~180 deg); SQPNP is a global solver that
        # picks the correct branch. Verified against the same faces the
        # ITERATIVE path flipped.
        ok, rvec, tvec = cv2.solvePnP(
            _PNP_MODEL_POINTS, image_points, camera_matrix,
            distCoeffs=None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok or tvec[2][0] <= 0:
            return None

        rotation, _ = cv2.Rodrigues(rvec)
        posed = (rotation @ _FACE_CENTER_MODEL_POINTS.T + tvec).T
        face_center = posed.mean(axis=0)

        return FaceObservation(
            landmarks=landmarks,
            iris_left=landmarks[IRIS_LEFT_CENTER].copy(),
            iris_right=landmarks[IRIS_RIGHT_CENTER].copy(),
            rvec=rvec,
            tvec=tvec,
            rotation=rotation,
            euler_deg=_euler_from_rotation(rotation),
            face_center=face_center,
        )

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self) -> "HeadTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _draw_debug(frame: np.ndarray, obs: FaceObservation,
                camera_matrix: np.ndarray) -> None:
    """Draw landmarks, irises and head-pose axes onto the frame."""
    for x, y in obs.landmarks[::8]:
        cv2.circle(frame, (int(x), int(y)), 1, (90, 90, 90), -1, cv2.LINE_AA)
    for iris in (obs.iris_left, obs.iris_right):
        cv2.circle(frame, (int(iris[0]), int(iris[1])), 3, (235, 235, 235),
                   1, cv2.LINE_AA)

    axes = np.float64([[80, 0, 0], [0, 80, 0], [0, 0, 80], [0, 0, 0]])
    projected, _ = cv2.projectPoints(axes, obs.rvec, obs.tvec, camera_matrix,
                                     None)
    projected = projected.reshape(-1, 2).astype(int)
    origin = tuple(projected[3])
    for end, color in zip(projected[:3],
                          [(80, 80, 220), (80, 200, 80), (220, 160, 60)]):
        cv2.line(frame, origin, tuple(end), color, 2, cv2.LINE_AA)

    text = (f"head pitch {obs.head_pitch:+6.1f}  "
            f"yaw {obs.head_yaw:+6.1f}  roll {obs.head_roll:+6.1f}")
    cv2.rectangle(frame, (0, 0), (520, 34), (32, 32, 32), -1)
    cv2.putText(frame, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (235, 235, 235), 1, cv2.LINE_AA)


def main() -> int:
    """Visual smoke test: landmarks + head pose over the live camera."""
    from capture.camera import Camera

    window = "gazetrack - head tracking check"
    with Camera() as camera, HeadTracker() as tracker:
        camera.verify()
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            frame = camera.read()
            if frame is None:
                print("ERROR: frame grab failed.")
                return 1
            obs = tracker.process(frame)
            if obs is not None:
                h, w = frame.shape[:2]
                _draw_debug(frame, obs, tracker.camera_matrix(w, h))
            else:
                cv2.putText(frame, "no face detected", (10, 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 220),
                            1, cv2.LINE_AA)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
