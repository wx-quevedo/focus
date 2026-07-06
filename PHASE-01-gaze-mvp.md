# PHASE 01 — Gaze Tracking MVP (el puntito que sigue la mirada)

Proyecto: `gazetrack`
Objetivo de la fase: puntito en pantalla que sigue la mirada del usuario en tiempo real,
calibrado, empujando la precisión hasta el techo práctico de una webcam RGB.

## Contexto de hardware (fijo para esta fase)
- Webcam: Logitech C920 (1080p). Forzar 1080p@30, autofocus OFF, auto-exposure OFF.
- Máquina: laptop RTX 4090 16GB, Windows. Inferencia por CUDA (no MPS).
- Distancia a pantalla objetivo: ~60cm, luz frontal difusa constante.
- Techo de precisión realista: 2.5°–3.5° (~2.5–4cm en pantalla). El centro será más
  preciso que las esquinas — limitación inherente, no bug.

## Decisión de arquitectura: DOS CAPAS (no confundir)
Capa 1 — Gaze direction (modelo pesado, GPU):
  ETH-XGaze (ResNet backbone) → vector 3D de mirada (yaw, pitch en grados).
  Input: crop de cara normalizado. Robusto a head pose (mejor que L2CS para esto).
  L2CS-Net como fallback seleccionable por flag. MediaPipe iris como fallback final.
Capa 2 — Mapeo a pantalla (calibración por usuario/sesión):
  Regresión polinómica de 2do grado con edge weighting.
  Aprende: (yaw, pitch, head_pose) → (screen_x, screen_y).
  Específica a la geometría cámara-pantalla-cara del usuario.

NO usar regresión clásica tipo WebGazer como capa 1 (se queda en 5-6°).

## Stack técnico
- Python 3.10, PyTorch + CUDA (verificar torch.cuda.is_available()).
- Captura: OpenCV. Forzar resolución y desactivar autofocus/auto-exposure vía
  cv2.CAP_PROP_AUTOFOCUS=0, CAP_PROP_AUTO_EXPOSURE, CAP_PROP_FRAME_WIDTH/HEIGHT.
- Detección/crop de cara + head pose: MediaPipe Face Mesh (468 landmarks + iris).
- Modelo gaze: ETH-XGaze pesos preentrenados (descargar en setup).
- Mapeo: scikit-learn PolynomialFeatures(degree=2) + Ridge, con sample_weight
  para edge weighting.
- Suavizado del puntito: filtro de Kalman (o One-Euro filter) para que no tiemble.
  ESTE PASO es lo que hace que "se sienta" como Tobii aunque el error crudo sea mayor.
- Overlay del puntito: capa transparente fullscreen. Empezar con ventana OpenCV
  fullscreen; migrar a overlay real (PyQt6 o win32 layered window) después.

## Comandos (CLI, patrón calibrate/demo/record/evaluate)
- `calibrate` : rutina de calibración densa. Ver sección abajo.
- `demo`      : puntito en vivo siguiendo la mirada, sobre pantalla.
- `record`    : graba sesión de gaze a CSV/SQLite con timestamps.
- `evaluate`  : grid de test (5×5 mínimo) → reporta error angular y error en px/cm.

## Calibración (clave para máxima precisión — prioridad del usuario)
Modo básico: 9 puntos.
Modo máxima precisión (default de esta fase):
  - 25 puntos en grid 5×5 + puntos EXTRA en los 4 bordes y 4 esquinas (mitiga la
    pérdida de precisión periférica conocida de webcam gaze).
  - Multi-frame averaging: recolectar ~0.4s de muestras por punto, promediar,
    descartar outliers.
  - Edge weighting: mayor peso a puntos de borde/esquina en la regresión.
  - Smooth pursuit adicional (opcional pero recomendado): una bola que se mueve
    lento por trayectorias en la pantalla mientras el usuario la sigue → cientos de
    muestras (yaw,pitch)→(x,y). Mucho más denso que puntos discretos.
  - Guardar calibración a YAML por usuario (recalibrar si cambia luz/posición).

## Métrica de éxito de la fase
- `evaluate` reporta error angular medio y error en cm.
- Meta MVP: error medio ≤ 3.5° en centro, puntito visualmente "sigue" la mirada
  por cuadrantes finos sin saltar.
- Documentar error por zona (centro vs bordes vs esquinas) — se espera degradación
  periférica; el objetivo es cuantificarla, no eliminarla.

## Estructura de proyecto sugerida
```
gazetrack/
  main.py                 # CLI: calibrate/demo/record/evaluate
  capture/camera.py       # OpenCV + config C920 (res, autofocus off)
  gaze/eth_xgaze.py       # wrapper modelo + carga de pesos CUDA
  gaze/mediapipe_head.py  # face mesh, iris, head pose
  mapping/calibrator.py   # regresión polinómica + edge weighting + persistencia YAML
  mapping/smoother.py     # Kalman / One-Euro
  overlay/dot.py          # render del puntito fullscreen
  eval/grid.py            # grid 5×5 y métricas angulares
  data/                   # calibraciones YAML, sesiones CSV/SQLite
  requirements.txt
```

## Fuera de alcance de esta fase (fases futuras)
- Score de concentración / productividad.
- Ventana activa del SO + actividad teclado/mouse.
- Heatmaps de horas productivas.
- Dashboard Laravel/React.
Esta fase es SOLO el puntito preciso. Todo lo demás se construye sobre esta base.

## Riesgos conocidos (avisar, no ocultar)
- Movimiento de cabeza degrada precisión aunque ETH-XGaze lo compense parcialmente.
- Lentes con reflejos interfieren.
- Cambios de luz obligan a recalibrar.
- Esquinas/bordes menos precisos que el centro (inherente a webcam RGB).
- C920 baja a 720p por defecto a veces → verificar que quede en 1080p.
