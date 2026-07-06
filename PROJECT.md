# PROJECT.md — gazetrack

Documento maestro del proyecto. Fuente de verdad sobre estado, decisiones y avances.
No confundir con los docs de fase (PHASE-XX-*.md), que son specs de trabajo para el
agente de Claude Code. Este documento lo mantiene Claude.ai y se actualiza con cada
avance relevante.

Última actualización: 2026-07-05

---

## 1. Visión del proyecto

Hipótesis original: usar la webcam para analizar comportamiento frente al computador
y determinar concentración, registrar actividad, detectar distracciones y encontrar
las horas del día de mayor productividad, con alarmas ante distracción.

Análisis inicial concluyó:
- Lo confiable con webcam: presencia, dirección de mirada (gaze), head pose, blink
  rate/fatiga, expresión base.
- Hipótesis débil: un "score de concentración" solo con visión — la cara no distingue
  trabajo profundo de scroll. La solución futura es combinar visión con actividad del
  SO (ventana activa, actividad de teclado/mouse en conteo, nunca keylogging).
- La parte más valiosa y sólida (horas productivas) es estadística agregada sobre
  semanas de datos, no visión avanzada.

Redefinición del MVP (decisión del usuario): antes que productividad, construir
gaze estimation estilo Tobii — un puntito en pantalla que sigue la mirada, calibrado,
empujando la precisión hasta el techo práctico del hardware RGB disponible. La capa
de productividad se construye después sobre esta base.

## 2. Hardware y entorno (fijo)

- Webcam: Logitech c922 Pro Stream (1080p), en índice 1 (el índice 0 es la webcam
  integrada del laptop; hay cámaras virtuales Quest/OBS que pueden mover índices —
  el config fija el 1). Inicialmente se creía C920; mismo techo de precisión
  (~2.5°–3.5°, ~2.5–4 cm en pantalla a ~60 cm). Descriptor UVC: YUY2@1080p solo a
  5 fps; MJPEG@1080p30 y MJPEG 720p@60 disponibles. Backend obligatorio: MSMF
  (DSHOW elige YUY2@5fps ignorando el FOURCC pedido).
- Máquina de tracking: laptop Windows, RTX 4090 16GB, CUDA.
- Monitor de trabajo: 49" ultrawide (~120 cm de ancho físico; resolución exacta
  pendiente de confirmar). Es el escenario más exigente para webcam gaze:
  extremos a ±39° de ángulo visual desde ~75 cm, movimiento de cabeza
  inevitable. La geometría de pantalla debe estar SIEMPRE configurada
  explícitamente en evaluate (nunca defaults).
- Proyecto en: c:\xampp\htdocs\focus  (raíz del repo, nombre de proyecto: gazetrack)
- Python 3.10.6 en venv (.venv). Todo corre local; nada sale a la nube (privacidad:
  video de la cara del usuario).

Referencias de precisión para calibrar expectativas:
- Tobii 5 (hardware IR dedicado): ~0.5°–1°.
- Webcam + regresión clásica (WebGazer-style): 4°–6°. Descartado como capa principal.
- Webcam + CNN appearance-based (ETH-XGaze / L2CS): 2°–4° en buenas condiciones.
- Limitación inherente conocida: precisión se degrada en bordes y esquinas de la
  pantalla; el centro siempre será mejor. Objetivo: cuantificarla, no eliminarla.

## 3. Arquitectura (decisión central)

Dos capas estrictas, sin mezclar responsabilidades:

- Capa 1 — Gaze direction (GPU): ETH-XGaze (ResNet-18, PyTorch/CUDA) produce el
  vector de mirada (yaw, pitch) desde el crop de cara normalizado. MediaPipe Face
  Mesh (478 pts + iris) provee detección de cara y head pose (PnP).
  Convenciones fijadas por el agente y consumidas por la capa 2: yaw positivo =
  mirada hacia la derecha del sujeto; pitch positivo = hacia arriba; espacio de cámara.
- Capa 2 — Mapeo a pantalla (por sesión/usuario): regresión polinómica de 2do grado
  (scikit-learn PolynomialFeatures + Ridge) con edge weighting, que aprende
  (yaw, pitch, head_pose) → (screen_x, screen_y). Calibración densa: grid 5×5 +
  puntos extra en bordes/esquinas + multi-frame averaging (~0.4 s/punto con descarte
  de outliers) + opcional smooth pursuit (bola en movimiento). Persistencia en YAML.
- Suavizado del puntito: One-Euro filter (elegido sobre Kalman: más simple y mejor
  respuesta en gaze). No mejora el error crudo; hace que el puntito "se sienta"
  estable.

Decisiones descartadas y por qué:
- WebGazer / regresión clásica como capa 1: techo de 5°–6°, insuficiente.
- L2CS-Net: previsto como fallback si los pesos de ETH-XGaze no eran descargables;
  sí lo fueron (release de GitHub, SHA-256 verificado), así que no se implementó.
  La interfaz del estimador quedó aislada para enchufar otro backend por flag.
- Repo gaze_track_webcam (ChiShengChen): evaluado como base, descartado (macOS/MPS
  only, dependencias innecesarias, proyecto frágil). Se tomaron de él técnicas:
  multi-frame averaging, edge weighting, patrón CLI calibrate/demo/record/evaluate,
  evaluación en grid 5×5.

## 4. Estructura del proyecto

```
focus/
  requirements.txt          torch 2.5.1+cu121, torchvision 0.20.1+cu121,
                            mediapipe 0.10.14, opencv-contrib 4.10.0.84,
                            numpy 1.26.4, scikit-learn 1.5.2, PyYAML 6.0.2
  check_env.py              verificación de entorno y CUDA
  capture/camera.py         C920: MJPG antes de resolución, 1080p@30, AF/AE off
  gaze/mediapipe_head.py    Face Mesh 478 pts + iris + head pose (PnP)
  gaze/eth_xgaze.py         normalización XGaze + ResNet-18 CUDA + live check
  mapping/calibrator.py     calibración 33 targets + poly2/Ridge + YAML sin pickle
  mapping/smoother.py       One-Euro escalar + wrapper 2D con timestamps
  overlay/dot.py            fullscreen OpenCV, DPI-aware, interfaz aislada
  eval/grid.py              grid 5x5, error por zona, px/cm/grados, p95, YAML
  main.py                   CLI: calibrate/demo/record/evaluate + flags geometría
  models/eth-xgaze_resnet18.pth   pesos verificados por SHA-256
  CLAUDE.md, .claude/skills/, .agents/skills/
  data/                     calibraciones YAML, sesiones
```

## 5. Cronología de avances

2026-07-05 — Definición y Fase 1 en marcha
- Análisis de viabilidad de la hipótesis de productividad; separación entre señales
  confiables e hipótesis débiles. Redefinición del MVP hacia gaze estilo Tobii.
- Investigación de modelos: L2CS-Net (3.92° MPIIGaze), ETH-XGaze (mejor robustez a
  head pose). Elegido ETH-XGaze como base.
- Escrito PHASE-01-gaze-mvp.md (spec para el agente) y prompt de arranque con orden
  de implementación a→i y puerta de confirmación tras (d).
- Agente completó etapas a→d: cámara con verificación de resolución real (1080p@30
  confirmados en smoke test), check de CUDA (RTX 4090 Laptop, torch cu121),
  MediaPipe head pose, wrapper ETH-XGaze con inferencia a 3.8 ms/frame.
- Autoskills: primera ejecución del agente fue incorrecta (creó CLAUDE.md + skill
  manual en lugar de ejecutar la herramienta). Corregido: npx autoskills detectó
  Python/NumPy/Scikit-Learn desde requirements.txt e instaló 6 skills
  (.agents/skills/ con symlinks en .claude/skills/).
- Seguridad: autoskills marcó python-executor con warning (ejecución remota en
  inference.sh, Bash amplio, URL raw sin auditar). Decisión: eliminarla — contradice
  el diseño 100% local y es redundante. Orden dada al agente.
- Prueba en vivo del usuario reveló tres problemas (ver sección 6): imagen
  desenfocada, 5.5 fps, rayo de gaze no visible. Prompt de corrección redactado.
- Agente construyendo etapas e→i en paralelo (no dependen de la cámara para
  escribirse).
- python-executor eliminado por completo, verificado con listado de ambos
  directorios de skills (6 en .claude/skills, 5 en .agents/skills, sin rastro).
- Agente completó etapas e→i con smoke tests headless: calibrador de 33 targets
  (grid 5x5 al 8% de margen + 8 puntos al 2.5% en bordes/esquinas, pesos
  1.0/1.5/2.0, rechazo de outliers por MAD, reintentos), StandardScaler →
  PolynomialFeatures(2) → Ridge con sample_weight, YAML sin pickle con validación
  al cargar. One-Euro verificado: jitter de fijación 13.5→6.3 px, sacada de
  1000 px asentada en <1 s. Eval con desglose centro(9)/bordes(12)/esquinas(4),
  media/p95/peor caso en px, cm y grados, comparación contra meta ≤3.5° centro,
  reporte a data/eval_*.yaml. CLI completo con flags de geometría y suavizado.
  Mapper validado sobre ground truth sintética (residual 5.4 px con ruido σ=4),
  roundtrip YAML idéntico.
- Diseño destacable del agente: la capa 2 nunca ve frames ni el modelo — recibe
  un sample_provider que main.py cablea a la capa 1. Separación de capas intacta.
- Fix de cámara entregado por el agente: (1) foco manual con --focus-sweep
  interactivo (W/S foco en pasos de 5, E/D exposición, métrica de nitidez por
  varianza del Laplaciano en tercio central, persistencia a data/camera.yaml,
  precedencia flag > config > default en los 4 modos); (2) impresión de
  propiedades reales en runtime en todo modo de cámara + verify() con diagnóstico
  de causa si fps <15; (3) rayo 2D con ganancia fija 9 px/grado + barras HUD
  yaw/pitch centradas en cero. Restricción respetada: inferencia intacta.
- Hallazgo técnico: el readback de FOURCC bajo DirectShow reporta YUY2 aunque el
  cable corre MJPG (quirk conocido; YUY2@1080p30 es imposible por USB 2.0). El
  fps medido es la verdad de terreno: 30.2 fps en captura pura.
- Causa del 5.5 fps del live check aún sin confirmar; el nuevo arranque
  desambigua: measured ~30 + HUD ~5 = cuello en procesamiento MediaPipe (smoke
  tests nunca tuvieron cara real delante); measured ~5 = exposición de cámara.
  Pendiente de la prueba del usuario. La explicación del rayo invisible fue
  geométrica: proyección 3D de un vector casi paralelo al eje de cámara colapsa
  a un punto.
- CAUSA RAÍZ del 5.5 fps resuelta: la cámara externa es una c922 Pro Stream (no
  C920) en índice 1; el índice 0 es la integrada del laptop, sobre la que se
  hicieron por error los smoke tests anteriores (por eso "pasaban"). El
  descriptor UVC de la c922 solo ofrece YUY2@1080p a 5 fps, y el backend DSHOW
  de OpenCV elige formato por resolución ignorando el FOURCC pedido → caía
  siempre en YUY2@5fps. Solución: MSMF como backend primario (negocia MJPEG solo
  si NO se le pasa FOURCC) con DSHOW como fallback. Verificado: 1080p a 29.8 fps
  medidos, escrituras de foco/exposición funcionando (probado empíricamente;
  los readbacks de MSMF son basura, las escrituras no). Falsa alarma de
  autofocus aclarada: flag 2 = manual en CameraControl, no booleano; el AF sí
  se apagaba. Advertencias persistidas en CLAUDE.md contra revertir a
  DSHOW-first o pasar FOURCC a MSMF. Índice 1 persistido en data/camera.yaml.
- Foco y fps verificados por el usuario (30.5 y 25.5 fps en HUD, imagen nítida),
  pero nuevo síntoma: el gaze responde a la cabeza y NO a los ojos (yaw clavado
  en ~+6° con ojos girados a extremos). Descartada la distancia como causa
  (probado a ~60 cm sin cambios notables; el usuario trabaja a 70-80 cm).
  Prompt de diagnóstico por niveles (patch → salida cruda → des-normalización)
  enviado al agente.
- CAUSA RAÍZ diagnosticada por el agente (Nivel 1 — normalización/warp): el
  modelo 3D del PnP (_PNP_MODEL_POINTS) era el genérico clásico de head-pose,
  ~3.3x sobredimensionado respecto a mm reales. La normalización de ETH-XGaze
  asume mm (cara a ~600 mm, focal 960 px); solvePnP ubicaba la cara a 1925 mm y
  la normalización magnificaba ~6x, recortando el patch a solo nariz/boca — los
  ojos ni aparecían en el input de la CNN. Evidencia: barrido de escala (a 0.30
  la distancia PnP da 577 mm ≈ los 600 mm asumidos; patch con ambos ojos
  visibles y centrados), head pose invariante a escala (+3.0° en todas las
  escalas), CNN sana (spread ~4.5° ante shear), det(rot_norm)=+1.000. Nota de
  proceso: la apuesta previa de Claude.ai era el Nivel 3 (des-normalización);
  la evidencia mandó.
- Fix confirmado al agente: modelo de 6 puntos en mm antropométricos reales
  (valores estándar, no el genérico x0.30), canario de sanidad de distancia PnP
  al arranque (advertir fuera de 400-1000 mm — habría detectado este bug el
  primer día), --show-patch disponible para verificación del usuario. No rompe
  calibraciones porque no existía ninguna aún. Pendiente de aplicar y verificar.
- Fix aplicado y validado headless por el agente: modelo antropométrico real en
  mm (canto externo ~90 mm, boca ~50 mm, nariz-mentón ~65 mm), con hallazgo
  colateral: el modelo real (casi plano) hacía que SOLVEPNP_ITERATIVE cayera en
  la rama espejo del PnP (tvec_z negativa, roll ~180°) → cambiado a
  SOLVEPNP_SQPNP (solver global, elige rama correcta; reproduce el head pose
  del modelo anterior exacto) + guard tvec_z<=0 descarta frame. Sanity check de
  distancia PnP al arranque (banda 400-1000 mm). Validación con foto real:
  patch pasó de "solo nariz/boca" a cara completa con ambos ojos centrados,
  head pose idéntico (+2.9 vs +3.0), CNN con cambio de signo limpio ante
  dirección de mirada. Descarte colateral: BGR/RGB nunca fue problema (0.39°).
  Advertencias en CLAUDE.md/skill contra revertir modelo o solver.
- FIX VERIFICADO EN VIVO por el usuario: gaze responde a ojos con rangos amplios
  (yaw +43.5° a -37.5°, pitch +22.6° a -37.9°), separación ojos/cabeza correcta
  (gaze +15° con head yaw -27° al mirar arriba), dirección consistente, 27-30
  fps. Capa 1 sana de punta a punta. Warnings de protobuf de MediaPipe:
  inofensivos.
- CONTEXTO NUEVO CRÍTICO: el monitor es un 49" ULTRAWIDE (~120 cm de ancho).
  Primera calibración + evaluate ejecutados, pero la línea base NO es válida:
  (1) el evaluate corrió con el default de 34.4 cm de ancho → cm/grados del
  reporte inválidos; los px son la verdad: ~900 px de error medio, jitter crudo
  150 px (~3.5° de ruido por frame, normal para la CNN); (2) el protocolo
  "mover solo los ojos" era inadecuado — en 120 cm a ~75 cm de distancia los
  extremos exigen ±39° de ángulo visual y el movimiento de cabeza es
  inevitable; la calibración debe capturar comportamiento natural cabeza+ojos
  consistente con el uso. Prompt enviado al agente: geometría de pantalla
  persistente (sin defaults silenciosos), diagnóstico por target de la
  calibración (residuales, rango de head pose, varianza), protocolo natural en
  pantalla, y error en % del ancho como métrica comparable.
- Expectativa ajustada: en un 49" ultrawide incluso el techo de ~3° son ~4 cm
  en pantalla — puntito regional, no quirúrgico (un Tobii real también sufre
  en pantallas de este tamaño; su rango recomendado es hasta 27-30").
  Suficiente de sobra para el objetivo de productividad. Pendiente del usuario:
  resolución del monitor y posición de montaje de la c922 (el head yaw de -22°
  en las capturas sugiere cámara descentrada).
- Agente entregó geometría persistente y diagnósticos: overlay/screen.py +
  data/screen.yaml (width/height/distance_cm, precedencia flag > config), sin
  default silencioso (sin config: "NOT CONFIGURED", se omiten cm/grados, px y %
  siguen válidos), autodetección DPI-aware de resolución, subcomando `screen`.
  Diagnósticos de calibración: residual por target ordenado descendente, rango
  de head yaw/pitch cubierto, flags MOVED (varianza >2.5° entre frames) y
  HIGH-RESIDUAL, persistidos en el YAML. Protocolo corregido en pantalla:
  mirada natural cabeza+ojos (la instrucción "solo ojos" contradecía el uso
  real en ultrawide). Evaluate con % del ancho por zona (métrica comparable
  entre monitores). Resolución detectada: 3840x1080 (DFHD 32:9). Línea base
  sucia registrada: ~900 px ≈ ~23% del ancho.
- Geometría física del 49" 32:9 por trigonometría: 119.8 x 33.7 cm. Comando de
  configuración entregado al usuario (distancia a ajustar a la real). Pendiente
  físico antes de recalibrar: centrar la c922 en el borde superior del monitor
  (head yaw -18/-22° constante en capturas sugiere cámara descentrada).
- Recalibración limpia del usuario (cámara centrada, geometría configurada
  119.8x33.7 cm @ 75 cm): residual de entrenamiento 60.6 px (~1.44°), 4/33
  targets HIGH-RESIDUAL, head yaw span 40.5° (protocolo natural funcionando),
  head pitch span 4.7° (esperado: pantalla baja). Pero evaluate: 1352.9 px
  medio (35% del ancho) — 22x el residual — con patrón por zonas INVERTIDO
  (centro 1659 px peor que esquinas 962 px) y jitter de solo 38.5 px. Primer
  evaluate desastroso previo explicado: corrió ANTES de recalibrar (timestamps),
  midió la calibración muerta. El segundo, con calibración fresca, confirmó
  bug estructural real.
- DIAGNÓSTICO DEL AGENTE (con evidencia dura): NO es serialización (roundtrip
  vivo-vs-recargado 0.0 px sobre los 33 targets reales; features consistentes
  entre calibrate/evaluate). ES SOBREAJUSTE de la capa 2: 35 features
  polinómicas (grado 2 x 7 inputs) contra 33 muestras con Ridge alpha=1.0 —
  casi interpola el entrenamiento y extrapola desbocado. Evidencia: 16% de
  predicciones fuera de pantalla con inputs de la propia distribución de
  calibración, rango predicho x=[-1768, 5841] vs pantalla 0-3840,
  ||coef_x||=1057 con face_x a coef -218 y sigma 6.9 mm entre sesiones. El
  clamp plegaba predicciones desbocadas al borde → esquinas "acertaban" por
  rebote → patrón invertido. Nota de proceso (2a vez): la hipótesis de
  Claude.ai (serialización) descartada por evidencia.
- Instrumentación permanente agregada: evaluate mide sobre predicción RAW (sin
  clamp — un modelo desbocado ya no puede esconderse), tabla por target
  target-vs-pred con marcas OFF, conteo de off-screen, y trazabilidad (imprime
  calibración cargada + timestamp, advierte si camera/screen.yaml cambiaron
  después).
- Fix confirmado al agente: quitar face_x/y/z del vector (4 inputs → 15
  features vs 33 muestras), alpha por LOOCV (RidgeCV, reportando alpha elegido
  y error LOOCV como predictor honesto), guardrail de ratio features/muestras
  >0.5, y versionado del vector de features (rechazar calibraciones viejas).
  Expectativa post-fix: 150-350 px (~3-7°); la hipótesis del usuario sobre el
  movimiento de cabeza en extremos se evalúa contra la tabla raw DESPUÉS del
  fix.
- LÍNEA BASE REAL ESTABLECIDA (hito): calibración con alpha=2.15 por LOOCV,
  LOOCV error 224.6 px (~5.34°) → evaluate 246.2 px medio (6.41% del ancho,
  ~5.85°), jitter crudo 76.4 px (2%). Los tres checks de sanidad pasaron:
  evaluate ≈ LOOCV (murió el 22x), off-screen 3/25 por márgenes mínimos,
  patrón sin la inversión patológica. Meta original de 3.5° reconocida como
  irrealista para este monitor: con head yaw de ±35-39° exigido por el
  ultrawide, ETH-XGaze opera cerca de su régimen difícil (~10° en Gaze360);
  5.9° aquí es un resultado sólido y para productividad ya sobra.
- Hallazgo en la línea base: error REGIONAL — columna x=2726 (centro-derecha)
  con 357-499 px (2x el resto), peor target de calibración en esquina inferior
  derecha, head yaw asimétrico [-39, +31.7] → causa física probable: cámara o
  asiento descentrados. Además esta sesión corrió a 25 fps con warning de modo
  no-MJPEG, y el foco del config cambió de nuevo (35 → 0 → 5).
- DECISIÓN DEL USUARIO: iterar precisión antes de Fase 2 → FASE 1B abierta:
  smooth pursuit (palanca principal de datos: serpentina + bordes + diagonales,
  compensación de latencia 0-250 ms elegida por CV, filtrado de parpadeos/
  sacadas/re-enganche, combinación ponderada con los 33 targets discretos) +
  selección de alpha y grado por k-fold POR BLOQUES TEMPORALES (innegociable:
  el pursuit está autocorrelacionado y el k-fold aleatorio mentiría) +
  diagnóstico regional 3x3 en evaluate. Meta orientativa: ≤4% del ancho
  (~150 px) y ninguna región >1.5x la mediana. Doc PHASE-01B-pursuit.md +
  prompt entregados. Prerrequisitos físicos del usuario antes de recalibrar:
  centrar c922 y asiento, puerto USB que sostenga 1080p30, focus-sweep final.
- Fase 1B (a)+(b) entregados por el agente con los tres ajustes de Claude.ai
  aplicados (velocidad 350 px/s documentada como ~8-9°/s reales — no 30 —,
  lag analítico con fórmula paramétrica exacta, sacada con umbral combinado
  max(3x objetivo, 80°/s)): grid 3x3 en evaluate con flag [!] a >1.5x mediana
  (verificado detectando la degradación derecha simulada), mapping/pursuit.py
  con colección separada de procesamiento, trayectoria serpentina + perímetro
  + diagonales bidireccional (~100 s), filtrado por causa, lag por CV por
  bloques, stream crudo a data/pursuit_*.csv. Smoke clave: lag de 100 ms
  inyectado recuperado exacto; filtrado 90% retenido. Detenido en la puerta:
  usuario debe cumplir prerrequisitos físicos y correr --pursuit, reportando
  % descarte (sano 10-25%), lag (sospechoso en extremos 0/250), block-CV y
  experiencia subjetiva.
- COLECCIÓN PURSUIT EJECUTADA (usuario, con físico corregido: cámara centrada,
  ~28 fps sin warning MJPEG, foco confirmado): 2790 muestras crudas, descarte
  8.4% (blink 0%, sacada 3.7%, re-enganche 4.7%), block-CV solo-pursuit
  126.8 px (3.3% del ancho, ~3.02°) — ya la MITAD de la línea base de 246 px
  antes de integrar. Bandera amarilla anticipada: lag clavado en el tope del
  grid (250 ms) con curva bajando monótona — el óptimo está más allá; 250+ ms
  excede la fisiología (~100-150 ms), el excedente es latencia del pipeline
  (buffering MSMF/OpenCV + timestamp post-procesamiento). Instrucción al
  agente: extender grid a 0-450 ms re-corriendo sobre el CSV guardado (sin
  recolectar), reducir latencia de captura (CAP_PROP_BUFFERSIZE=1, timestamp
  en el grab), y — hallazgo clave para (c) — el grid discreto de
  calibration.yaml está MUERTO (recolectado con la cámara en posición vieja):
  la sesión combinada debe recolectar grid fresco + pursuit en una sola
  corrida, con flag para reutilizar CSV previo.
- Agente resolvió punto 1 y entregó (c)/(d): el lag de 250 ms resultó MÍNIMO
  INTERIOR real (U limpia en grid 0-450; el scan original terminaba justo en
  el fondo del valle). Latencia diagnosticada con honestidad: procesamiento
  ~10 ms; el grueso es captura inherente (frames hasta 33 ms de antigüedad +
  pipeline USB c922 + display) — compensada por el lag para precisión; para
  el demo en vivo la enmascara el One-Euro (overlay de menor latencia queda
  como trabajo futuro). BUFFERSIZE=1 ignorado por MSMF y sin backlog real
  (probado por comportamiento). Sesión combinada: grid fresco + pursuit en
  una corrida, --pursuit-csv para reutilizar (sidecar .meta.yaml), selección
  de grado (2 vs 3) y alpha por block-CV combinado con anclajes discretos en
  cada train fold, pesos 20:1, FEATURE_VERSION=3, residual regional 3x3 en
  calibrate. Verificación: reutilización del CSV real reproduce lag/CV
  exactos. PUERTA FINAL: usuario corre sesión combinada + evaluate; criterios:
  ≤~150 px global, ninguna región >1.5x mediana, evaluate ≤1.3x block-CV.
- FASE 1B CERRADA (hito): sesión combinada (33 discretos x20 + 2760 pursuit =
  2793 muestras, lag 250 ms, grado 2 elegido por CV sobre grado 3 — 124.8 vs
  132.6 px —, alpha 0.215, ratio 0.005). Resultado final del evaluate:
  * Global: 246.9 → 166.9 px (6.41% → 4.35% del ancho) — mejora del 32%.
  * CENTRO: 105.3 px / 2.51° — CUMPLE la meta original de 3.5° que se había
    declarado irrealista para el 49" ultrawide.
  * Bordes: 107.6 px / 2.56° — parejo con el centro.
  * Degradación derecha: ELIMINADA (columna derecha 144/115/106 px, entre las
    mejores) — la causa era el descentre físico de la cámara, confirmado.
  * Jitter crudo: 49 px (1.28%), el más bajo del proyecto.
  * Asterisco: UN target patológico en la esquina superior izquierda (307,86)
    con 1482 px de error (siguiente peor: 234 px) infla esquinas a 483 px,
    la global sobre 150, y el ratio evaluate/CV a 1.34. Sin ese outlier la
    global ronda ~112 px y todos los criterios se cumplen con holgura.
  DECISIÓN DEL USUARIO: cerrar 1B como éxito y anotar la esquina para después.
- Fix de sobreajuste aplicado y verificado headless: vector 4D (gaze yaw/pitch,
  head yaw/pitch) → 14 términos polinómicos vs 33 muestras (ratio 0.42), alpha
  por LOOCV explícito (33 refits, grid 0.1-1000; elegido sobre el GCV de
  RidgeCV porque maneja target 2D + sample_weight), calibrate reporta LOOCV
  error como predictor honesto del error real, guardrail de ratio >0.5,
  FEATURE_VERSION=2 con rechazo de calibraciones v1 (probado contra el archivo
  real). Criterios de éxito de la línea base: evaluate ≈ LOOCV (no 22x), ~0%
  off-screen en tabla raw, patrón de zonas normal (centro mejor que extremos).

## 6. Problemas encontrados y estado

RESUELTOS
- Agente interpretó "autoskills" como crear skills a mano → corregido con
  instrucción explícita del comando npx; verificado con listado en disco.
- Skill python-executor con warning de seguridad → eliminada (symlink + origen),
  verificado con listado de ambos directorios.
- 5.5 fps en live check → causa raíz: c922 en índice 1 bajo DSHOW caía en
  YUY2@5fps (único modo YUY2@1080p de esa cámara). Resuelto con MSMF-first.
  Lección de proceso: verificar la IDENTIDAD del hardware antes de dar por
  buenas las verificaciones — los smoke tests corrían sobre la webcam integrada
  (índice 0) sin saberlo.
- Falsa alarma "autofocus ON" → error de interpretación de flags (2 = manual).

ABIERTOS
- (ninguno bloqueante; Fase 1 y 1B cerradas)

LIMITACIÓN CONOCIDA ANOTADA PARA DESPUÉS
- Esquina superior izquierda inestable: el target (307,86) evaluó a 1482 px
  (vs 172 px in-sample en calibración en esa región) — la pose más extrema de
  cabeza+ojos. Hipótesis a investigar cuando se retome: degeneración del
  modelo de gaze en ese ángulo extremo, o estrategia cabeza/ojos inconsistente
  del usuario entre calibración y uso en esa esquina. Afecta una región de
  ~40x36 cm; irrelevante para productividad, visible en la experiencia del
  puntito. Posibles palancas: más cobertura pursuit en esa esquina, detección
  de pose extrema con degradación elegante del cursor, o diagnóstico del gaze
  crudo en esa pose.

RIESGOS CONOCIDOS (aceptados, monitorear)
- Índice de cámara por número (1): cámaras virtuales de Quest/OBS pueden
  reordenar índices al conectarse. Síntoma: el focus-sweep muestra una imagen
  que no es la c922. Mejora futura si molesta: selección por nombre de
  dispositivo en vez de índice.
- Espejo en preview: si la imagen está espejada, el mapeo de la capa 2 puede
  invertir X (puntito va al lado contrario). Vigilar en la primera calibración.
- Head yaw ~+22° observado en el live check: cámara posiblemente descentrada
  respecto al usuario. Centrar la cámara sobre el monitor antes de calibrar.
- Cambios de luz obligan a recalibrar; lentes con reflejos interfieren.
- Driver DSHOW no confirma lectura de AF/AE tras el set (devuelve -1);
  verificación es visual.

## 7. Próximos pasos (en orden)

FASE 1 — CERRADA: sistema completo de gaze tracking funcionando de punta a
punta (cámara → CNN → calibración → puntito → evaluación honesta).

FASE 1B — CERRADA COMO ÉXITO: 166.9 px global (4.35% del ancho), centro
105 px / 2.51° (meta original cumplida), degradación derecha eliminada,
esquina superior izquierda anotada como limitación conocida.

SIGUIENTE: FASE 2 — ABIERTA (doc PHASE-02-productivity.md + prompt entregados,
2026-07-05). Estructura con puertas: 2A overlay transparente estilo Tobii
(PyQt6 layered/click-through, cursor final con presencia, fade en pose
extrema, hotkey, bandeja) → puerta: usuario trabaja con el overlay sobre apps
reales → 2B registro (gaze por zonas + presencia + ventana activa con
--no-titles + conteos de teclado/mouse, SQLite WAL) → puerta: revisión
conjunta de activity.db tras >= 1 h real → 2C dashboard Laravel 12 + Inertia
+ React (heatmap de horas productivas como vista central, línea de tiempo,
distribución de atención, clasificación por reglas editables, alarma toast
sobria). El score de concentración queda explícitamente fuera (después,
contra auto-reporte). Privacidad: 100% local, conteos nunca contenido.

TAMBIÉN EN COLA (fases futuras ya anotadas): compatibilidad universal de
pantalla, cursor de gaze de la interfaz final, overlay de menor latencia,
esquina superior izquierda.

AVANCE 2A: plan del agente aprobado con dos correcciones de Claude.ai:
(1) el umbral de pose extrema para el fade NO puede ser fijo — |yaw|>35°
haría fade durante uso válido, pues la calibración cubre head yaw hasta
-39/+31.7 en este monitor; debe leerse del rango cubierto en los diagnostics
del YAML de calibración + margen configurable (~5°), con fade adicional si
la predicción cae >10% del ancho fuera de pantalla; (2) zona muerta como
flag --deadzone default 12 px (6 px era fino para el jitter real de 49 px
crudo / ~10-20 tras One-Euro). Aprobado tal cual el resto: PyQt6 + retoque
win32 NOACTIVATE, repintado por región sucia, pynput GlobalHotKeys → señal
Qt, worker QThread para el pipeline, tray Qt con icono por código, hotkeys
Ctrl+Alt+G/Q, fade 250 ms, anillo ~40 px parametrizable.

2A ENTREGADO (en puerta de prueba del usuario): overlay PyQt6 + win32
NOACTIVATE confirmado en corrida real, repintado por región sucia, anillo
parametrizable (--dot-size/-opacity/-color), --deadzone 12, hotkeys y tray,
trazabilidad al iniciar. Fade adaptativo leyendo el rango de head pose de la
calibración (+metadato head_range en fit_combined; calibraciones previas usan
fallback generoso [-45,+45] hasta recalibrar). Consumo medido con honestidad:
GPU ~20-25% / 3.3 GB VRAM, pipeline ~1.7 cores a 28.5 fps (excede el objetivo
aspiracional de <15% de un core; es costo inherente de FaceMesh+CNN, no de la
UI; palancas anotadas: torch.set_num_threads, cap a 20 fps). Limitación de
Windows documentada: apps elevadas dibujan sobre el overlay. Pendiente:
prueba del usuario sobre apps reales (seguimiento, transparencia, fijaciones
con ajuste de deadzone, estética del anillo, fade) — recomendado recalibrar
una vez para el rango exacto. Su aprobación abre 2B.

2A APROBADO EN LA PUERTA (usuario): el overlay funciona sobre apps reales
("no es exacto pero se ve genial" — consistente con la precisión medida de
~3 cm centro; la exactitud vive en la capa de calibración, no en el overlay).

2A.1 ABIERTA (petición del usuario): panel de configuración para todo lo
configurable — diálogo Qt desde el tray (y Ctrl+Alt+S), secciones cursor/
suavizado/fade/hotkeys con descripciones en lenguaje humano, CAMBIOS EN VIVO
sin reiniciar, persistencia en data/overlay.yaml con precedencia flag >
config > default, botón restaurar defaults, estructura preparada para crecer
con secciones de 2B/2C. Prompt entregado. Tras su visto bueno → 2B.

2A.1 ENTREGADO Y APROBADO: panel Qt desde tray/Ctrl+Alt+S con cambios en
vivo (setters, sin tocar pipeline), persistencia data/overlay.yaml con
debounce 400 ms y precedencia flag > config > default, secciones cursor/
suavizado/fade/hotkeys (hotkeys solo-lectura por ahora), restaurar defaults.
Bonus del agente: bug de cierre cazado y arreglado — pynput.stop() se cuelga
desde otro hilo en Windows y QThread.wait ignoraba timeout en PyQt6; solución
pragmática documentada en CLAUDE.md: app.quit() + os._exit() (los hilos
nativos de cámara/CUDA no desenrollan limpio; el SO libera la cámara).

2B AUTORIZADA: registro en segundo plano según spec, con precisiones nuevas:
zona de gaze grid 6x3 configurable (sección en el panel), tray con estado
registrando/pausado, y — derivado del cierre duro de 2A.1 — commit/checkpoint
por intervalo en la DB WAL para que os._exit() no pierda más que el último
intervalo. Puerta: sesión real del usuario >= 1 h + revisión conjunta de
activity.db con 4-5 consultas SQL de muestra entregadas por el agente.

2B ENTREGADO (en puerta): record --background con tray (pausar/reanudar),
--no-titles verificado en vivo (window_title NULL), conteos jamás contenido
(prohibición documentada), grid 6x3 configurable desde el panel, SQLite WAL
con transacción por intervalo (os._exit pierde máx. 5 s), recovery de
sesiones huérfanas, cámara desconectada → tracking_state=camera_lost con
backoff (probado en vivo por accidente: la c922 del agente tardó 73 s y el
hueco quedó bien registrado), ~1.8 MB/día medido, 5 consultas SQL de muestra
en SAMPLE_QUERIES. Nota honesta del agente: sin cara frente a la cámara en
sus pruebas, las filas tracking_state=ok reales las produce la sesión del
usuario. PUERTA: sesión real >= 1 h del usuario (trabajando natural,
incluyendo distracciones reales, anotando 2-3 momentos de referencia) →
revisión conjunta de las 5 consultas contra su recuerdo → abre 2C.

AMPLIACIÓN DE 2C SOLICITADA (usuario, durante su sesión de puerta 2B):
dashboard web con métricas y gráficas (ya en spec) + NUEVO: capturas de los
momentos de concentración/distracción. Diseño acordado: snapshots POR EVENTO
(off-screen sostenido, cara ausente, retorno de distracción, inicio de foco
profundo — un frame o ráfaga corta etiquetada), no video continuo (~GB/día
sin valor). Decisiones de privacidad: opt-in explícito apagado por defecto
(las fotos cambian la sensibilidad de data/), retención automática
configurable (default 7-14 días; las métricas viven, las caras no), y el
dashboard SIEMPRE en localhost/LAN — nunca en el VPS (fotos de cara +
comportamiento no salen de la máquina, principio del día 1). Implementación:
módulo de snapshots como extensión 2B.1 del recorder; el dashboard 2C los
incrusta en la línea de tiempo. Spec de 2C se actualiza al pasar la puerta
de 2B.

TRABAJO PARALELO AUTORIZADO (mientras el usuario corre su sesión de puerta):
prompt entregado para (Bloque 1) 2B.1 snapshots por evento — detector sobre
el stream existente (OFF_SCREEN_SUSTAINED, FACE_ABSENT, DISTRACTION_RETURN,
FOCUS_BLOCK_START), frame JPEG etiquetado en data/snapshots/ + tabla events;
decisión fina: los EVENTOS se registran siempre (métrica), solo las IMÁGENES
son opt-in con retención default 14 días — y (Bloque 2) 2C-fundación:
esqueleto Laravel 12 + Inertia + React en dashboard/ (localhost only),
autoskills del proyecto nuevo con revisión de warnings, SQLite read-only
directo (WAL permite lectores concurrentes), endpoints espejando las 5
consultas, shell de navegación con placeholders honestos. Las vistas
analíticas ricas quedan explícitamente detrás de la puerta de validación
de datos.

FASES FUTURAS (fuera del alcance de Fase 1)
- Cursor de gaze de la interfaz final (decisión del usuario, 2026-07-05): la
  bola de calibración se mantiene pequeña y sobria (un objetivo grande mete
  ambigüedad sobre el punto exacto mirado y ensucia los datos); el cursor de
  gaze del demo/overlay final se diseñará aparte — más grande y con mejor
  presencia visual, dentro de la línea profesional del proyecto (sin emojis,
  nada infantil).
- Compatibilidad universal de pantalla (decisión del usuario, 2026-07-05): el
  sistema debe funcionar en cualquier tamaño, resolución y proporción de
  pantalla (laptop, monitor estándar, ultrawide 49", multi-monitor). La base ya
  apunta ahí (geometría persistente configurable en vez de hardcodeada); faltará
  un flujo de setup por pantalla (detección de resolución + captura de
  dimensiones físicas + perfil de calibración por pantalla/geometría, de modo
  que cambiar de monitor no obligue a reconfigurar a mano) y validar el mapeo
  en proporciones distintas. Se aborda después de tener línea base en el 49".
- Fase 2 — Productividad: ABIERTA (ver sección 7); spec en
  PHASE-02-productivity.md.
- Score de concentración: solo derivado después, calibrado contra auto-reporte
  del usuario; nunca inventado desde visión pura.

## 8. Dinámica de trabajo (estándar para todos los proyectos)

Roles:
- Claude.ai (este chat): diseña arquitectura y specs, investiga y valida opciones
  técnicas, redacta TODOS los prompts/consultas/confirmaciones para el agente,
  analiza las respuestas del agente y mantiene este documento maestro.
- Washington: ejecuta los prompts en el agente de Claude Code, prueba en el
  hardware real, pega las respuestas del agente de vuelta al chat, y toma las
  decisiones de producto.
- Agente de Claude Code: implementa según spec, reporta con evidencia.

Reglas de la dinámica:
- Todo avance del agente pasa por Claude.ai antes de la siguiente instrucción:
  Washington pega la respuesta del agente → Claude.ai la analiza → Claude.ai
  redacta la siguiente instrucción lista para copiar/pegar.
- Verificación con evidencia real, no reporte verbal: listados de archivos en
  disco, outputs de comandos, propiedades leídas en runtime. (Caso autoskills:
  el agente reportó hecho algo que no había ejecutado.)
- Avance por fases con puertas de confirmación explícitas; el agente se detiene
  en cada puerta y no avanza sin luz verde.
- Cada fase tiene su doc de spec (PHASE-XX-nombre.md) + prompt de arranque.
  Claude.ai entrega siempre el par completo.
- Reglas permanentes en todo prompt de agente: interfaces profesionales sin
  emojis (UI, CLI, logs, README); ejecutar npx autoskills tras el setup
  (--dry-run primero, instalar con -a claude-code, revisar warnings de seguridad
  y no instalar skills que ejecuten código en servicios externos sin aprobación
  explícita).
- No calibrar/medir sobre señal defectuosa: los problemas de captura se cierran
  antes de juzgar o ajustar las capas superiores.
