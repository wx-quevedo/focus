# PHASE 01B — Iteración de precisión: smooth pursuit + degradación lateral

Proyecto: `gazetrack`
Punto de partida (línea base 2026-07-05): error medio 246.2 px (6.41% del ancho,
~5.85°), LOOCV 224.6 px — modelo sano (evaluate ≈ LOOCV, sin sobreajuste).
Objetivo de la iteración: bajar el error medio global de forma medible (meta
orientativa: ≤ 4% del ancho, ~150 px) y eliminar la degradación regional del
lado derecho, manteniendo evaluate ≈ predictor de CV como criterio de sanidad.

## Contexto del problema

1. Error REGIONAL, no uniforme: la columna x=2726 (centro-derecha) concentra
   errores de 357-499 px (el doble del resto). El peor target de calibración
   (874 px) fue la esquina inferior derecha. Head yaw asimétrico [-39, +31.7]
   sugiere causa física (cámara/asiento descentrados) que el usuario corrige
   ANTES de esta fase. El software debe además diagnosticarlo mejor.
2. 33 muestras es poco para un mapeo sobre 70° de span de head yaw. El smooth
   pursuit multiplica los datos por 1-2 órdenes de magnitud y es la palanca
   principal de esta iteración.

## Prerrequisitos físicos (usuario, antes de recalibrar)
- c922 centrada en el centro geométrico del monitor; silla frente al centro.
- Puerto USB que sostenga MJPEG 1080p30 sin warning.
- Focus-sweep final con la cámara en posición definitiva; no volver a mover.

## Trabajo del agente

### 1. Recolección por smooth pursuit (calibrate --pursuit, aditivo al grid)
- Un objetivo circular se mueve por trayectorias que cubren la pantalla
  completa: serpentina horizontal (5-6 pasadas) + barrido en los 4 bordes +
  dos diagonales. Velocidad constante y lenta (~300-400 px/s en un 3840 de
  ancho; ajustable por flag) — el smooth pursuit humano sigue bien hasta
  ~30°/s; más rápido induce sacadas.
- Muestrear (features, target_x, target_y) continuamente durante el recorrido.
- COMPENSACIÓN DE LATENCIA: el ojo persigue con ~100-150 ms de retraso. Emparejar
  la muestra de gaze del frame t con la posición del objetivo en t-lag, con lag
  calibrado automáticamente: elegir el lag (grid 0-250 ms) que minimice el error
  de CV del ajuste. Reportar el lag elegido.
- FILTRADO DE CALIDAD: descartar muestras durante parpadeos (sin detección de
  cara/iris), durante sacadas (velocidad angular del gaze muy superior a la del
  objetivo), y los primeros ~300 ms tras cada cambio de dirección de la
  trayectoria (re-enganche). Reportar % de muestras descartadas y por qué causa.
- El resultado se COMBINA con los 33 targets discretos (que anclan con
  multi-frame averaging de alta calidad): pesos relativos configurables,
  default razonable (p. ej. cada target discreto pesa como ~20 muestras de
  pursuit).

### 2. Modelo: capacidad decidida por validación, no a mano
- Con cientos/miles de muestras, LOO es caro: pasar a k-fold (k=5) para la
  selección de alpha, PERO con folds POR BLOQUES TEMPORALES (bloques contiguos
  de la trayectoria), no muestras aleatorias — las muestras consecutivas de
  pursuit están autocorrelacionadas y el k-fold aleatorio mentiría (error CV
  optimista).
- Permitir grado 3 SOLO si el CV por bloques lo justifica: ajustar grado 2 y
  grado 3, reportar CV de ambos, elegir el mejor. El guardrail de ratio
  features/muestras se mantiene.
- El reporte de calibración incluye: n muestras (discretas + pursuit tras
  filtrado), lag elegido, alpha, grado elegido, error CV por bloques (el nuevo
  predictor honesto), y residual por REGIÓN (grid 3x3 de la pantalla) para
  vigilar la degradación lateral.

### 3. Diagnóstico regional en evaluate
- Además de las zonas actuales (centro/borde/esquina), reportar el error en un
  grid 3x3 de la pantalla (izquierda/centro/derecha x arriba/medio/abajo) para
  confirmar si la degradación derecha desapareció tras la corrección física.
- Mantener: tabla raw target-vs-pred, % off-screen, evaluate ≈ CV como check.

## Criterios de éxito de la iteración
- Error medio global ≤ 4% del ancho (~150 px) — orientativo, no dogma.
- Grid 3x3 sin región con error > 1.5x la mediana de las regiones.
- evaluate dentro de ~1.3x del error CV reportado por calibrate.
- El demo con One-Euro debe "sentirse" notablemente más pegado (subjetivo del
  usuario, pero es el criterio de producto).

## Fuera de alcance de esta iteración
- Cambiar de modelo de gaze (capa 1), reentrenar CNNs, hardware nuevo.
- Compatibilidad universal de pantalla (fase futura ya anotada).
- Capa de productividad (Fase 2).

## Riesgos conocidos
- Autocorrelación del pursuit inflando la confianza del CV → mitigada con
  folds por bloques; vigilar evaluate vs CV.
- Lag mal estimado mete sesgo espacial sistemático en la dirección del
  movimiento → el grid de lag por CV lo mitiga; las trayectorias en ambas
  direcciones lo promedian.
- Si la degradación derecha persiste tras centrar la cámara, la causa sería
  óptica/iluminación (ventana lateral) o del modelo de gaze en ángulos
  extremos — se diagnostica con el grid 3x3 antes de decidir más.
