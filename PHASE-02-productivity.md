# PHASE 02 — Overlay estilo Tobii + capa de productividad

Proyecto: `gazetrack`
Punto de partida: Fases 1/1B cerradas. Gaze tracking calibrado y honesto:
166.9 px global (4.35% del ancho), centro 105 px / 2.51°, jitter crudo 49 px,
lag de captura ~250 ms compensado en calibración. Limitación conocida: esquina
superior izquierda inestable (anotada, no bloquea).
Objetivo de la fase: (2A) el puntito flotando sobre las aplicaciones reales del
usuario, estilo Tobii; (2B) registro de comportamiento en segundo plano;
(2C) dashboard de productividad con heatmaps de horas productivas.

Regla de entrega: 2A → puerta de confirmación → 2B → puerta → 2C.

## Principios permanentes (aplican a toda la fase)
- 100% local. Nada sale de la máquina. Es video de la cara y comportamiento
  del usuario: privacidad no negociable.
- Actividad de teclado/mouse = CONTEO por intervalo (eventos por minuto),
  NUNCA contenido de teclas. Prohibido cualquier forma de keylogging.
- Interfaces profesionales, sin emojis, en ninguna superficie (UI, CLI, logs).
- Los guardrails de Fase 1 siguen vigentes: trazabilidad de calibración,
  evaluate raw, versionado de features.

## 2A — Overlay transparente (la experiencia Tobii)

### Ventana
- Windows layered window: transparente, click-through, always-on-top,
  cubriendo la pantalla completa (3840x1080). Opciones técnicas en orden de
  preferencia: PyQt6 con WA_TranslucentBackground + WindowTransparentForInput
  + WindowStaysOnTopHint; alternativa win32 pura (WS_EX_LAYERED |
  WS_EX_TRANSPARENT | WS_EX_TOPMOST). El usuario debe poder ver y clickear
  sus aplicaciones con total normalidad debajo del overlay.
- Render eficiente: solo se redibuja la región del cursor (no fullscreen
  repaint a 30 fps). El overlay no debe consumir GPU/CPU perceptible.

### Cursor de gaze (decisión de diseño ya tomada por el usuario)
- Más grande y con mejor presencia que el puntito del MVP, manteniendo la
  línea profesional: p. ej. anillo suave con relleno semitransparente y halo
  tenue, paleta neutra (blanco hueso/gris sobre cualquier fondo). Nada
  infantil, sin emojis. Parametrizable: tamaño, opacidad, color.
- Suavizado One-Euro ya existente aplicado; considerar además una zona muerta
  pequeña (el cursor no se mueve ante desplazamientos < umbral) para que en
  fijaciones se sienta quieto en lugar de vibrar.
- Comportamiento en pose extrema / pérdida de cara: el cursor se desvanece
  (fade) en lugar de saltar o congelarse. Degradación elegante — relevante
  por la limitación conocida de la esquina superior izquierda.

### Control
- Subcomando `overlay` en main.py. Hotkey global para mostrar/ocultar el
  cursor (p. ej. Ctrl+Alt+G) y para salir. Icono en la bandeja del sistema
  opcional pero deseable (pystray), con menú mínimo: mostrar/ocultar, salir.
- Al iniciar: imprime calibración cargada + timestamp (trazabilidad ya
  existente) y advierte si es vieja.

### Criterio de la puerta 2A
El usuario trabaja unos minutos con el overlay activo sobre sus aplicaciones
reales: el cursor sigue la mirada con la precisión conocida, no interfiere
con clicks ni foco de ventanas, no consume recursos perceptibles, y el fade
en poses extremas se siente natural.

## 2B — Registro de comportamiento (el sensor de productividad)

### Qué se registra (por segundo, agregado a intervalos de 5 s en disco)
- gaze: zona de pantalla mirada (grid configurable, default 6 columnas x 3
  filas del monitor), on-screen/off-screen, y estado (tracking ok / cara
  perdida / pose extrema).
- presencia: cara detectada sí/no.
- ventana activa: proceso (ej. Code.exe) y título de ventana (ej. archivo
  abierto), vía win32/psutil. El título puede contener información sensible:
  guardarlo es útil (qué proyecto/archivo) pero debe existir un flag de
  privacidad --no-titles que registre solo el proceso.
- actividad: conteo de eventos de teclado y de mouse por intervalo (pynput u
  hooks win32). SOLO conteos. Documentar en el código la prohibición de
  capturar contenido.

### Almacenamiento
- SQLite en data/activity.db, WAL mode (lección aprendida de mtrad). Esquema
  simple y consultable: tabla samples(ts, gaze_zone, on_screen, tracking_state,
  face_present, process, window_title, kb_events, mouse_events) + tabla
  sessions(start, end, notes). Índice por ts.
- Rotación/compactación no necesaria aún (estimar y reportar tamaño/día).

### Modo de ejecución
- `record --background`: corre overlay opcional + registro, minimizado a la
  bandeja. Robustez: si la cámara se desconecta, reintenta y registra el hueco
  como estado, no crashea. Auto-resume tras suspensión del SO.

### Criterio de la puerta 2B
Una sesión de trabajo real de >= 1 hora registrada: el usuario y Claude.ai
revisan el contenido de activity.db (consultas de muestra incluidas en el
reporte del agente) y validan que los datos son coherentes con lo que el
usuario realmente hizo.

## 2C — Dashboard y análisis (Laravel 12 + Inertia + React)

### Stack y convenciones (las del usuario en todos sus proyectos)
- Laravel 12 + Inertia.js + React/TypeScript/Tailwind. Código en inglés, UI
  en español latinoamericano. Sin emojis, diseño profesional y sobrio.
- El dashboard LEE data/activity.db (SQLite) — no se migra la captura a
  Laravel; Python sigue siendo el sensor, Laravel la vista. Conexión SQLite
  read-only o export periódico, a decidir por el agente con justificación.

### Vistas mínimas
1. Heatmap de horas productivas: matriz día-de-semana x hora-del-día,
   intensidad = tiempo de actividad real (presencia + actividad + app de
   trabajo). LA vista central del proyecto — responde la pregunta original
   del usuario: ¿cuándo soy productivo?
2. Línea de tiempo del día: bandas por app/proceso con actividad superpuesta,
   gaps de ausencia visibles.
3. Distribución de atención: tiempo por zona de pantalla y por aplicación,
   on-screen vs off-screen.
4. Sesiones: lista con duración, apps dominantes, ratio de actividad.

### Clasificación productivo/no-productivo
- Por reglas configurables (tabla de procesos → categoría trabajo/neutral/
  distracción, editable desde el dashboard). NADA de ML ni score de
  concentración en esta fase — eso queda para después, calibrado contra
  auto-reporte (decisión de proyecto desde el día 1).

### Alarmas (mínimo viable)
- Regla simple configurable: si off-screen o app de distracción > N minutos
  continuos dentro de una sesión de trabajo, notificación de Windows (toast)
  sobria. Sin sonidos estridentes, desactivable.

### Criterio de la puerta 2C
Con >= 3-5 días de datos: el heatmap y la línea de tiempo cuentan una historia
que el usuario reconoce como fiel a su semana real.

## Fuera de alcance de la fase
- Score de concentración (fase posterior, contra auto-reporte).
- Compatibilidad universal de pantalla (anotada como fase futura).
- Overlay de menor latencia (anotado; el One-Euro enmascara por ahora).
- Fix de la esquina superior izquierda (limitación conocida anotada).
- Multi-monitor.

## Riesgos conocidos
- Permisos/antivirus con hooks globales de teclado/mouse y hotkeys: usar APIs
  estándar y documentar si Windows Defender pide excepción.
- Click-through en apps elevadas (admin): el overlay no puede dibujar sobre
  ventanas con privilegios superiores; documentar la limitación.
- Rendimiento: cámara + CNN + overlay + registro simultáneos; presupuesto
  objetivo < 15% de un core + uso de GPU marginal. Medir y reportar.
- Títulos de ventana con datos sensibles: mitigado con --no-titles.
