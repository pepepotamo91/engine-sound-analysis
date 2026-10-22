# engine-sound-analysis

Análisis acústico de motores a partir de una grabación (móvil, WhatsApp, etc.): espectrograma por FFT, **waterfall 3D**, **estimación de rpm en el tiempo**, espectro de órdenes e **informe HTML** interactivo.

Este README es a la vez documentación y **brief para Claude**: si le pides a Claude que analice un audio nuevo en este repo, debe seguir la sección [Procedimiento para Claude](#procedimiento-para-claude) paso a paso.

---

## Uso rápido

```bash
pip install -r requirements.txt          # numpy, scipy, matplotlib, pytest
sudo apt install ffmpeg                  # decodificación de audio
python scripts/get_plotly.py             # deja vendor/plotly.min.js (informe offline)

# 1) análisis
python -m engine_audio analyze audio/mi_motor.opus --name mi_motor \
    --cylinders 4 --strokes 4 --idle 700-900 --max-rpm 6500

# 2) escribir reports/mi_motor/meta.json (interpretación, ver abajo)

# 3) informe
python -m engine_audio report reports/mi_motor
```

Salida en `reports/<nombre>/`:

| Fichero | Qué es |
|---|---|
| `analysis.json` | Resumen numérico: mesetas, hipótesis de conversión a rpm, evidencias |
| `data.json` | Matrices para los gráficos (waterfall, espectrograma, órdenes, curva) |
| `diag_spectrogram.png` | Espectrograma con k·f_comb superpuesto: **comprobación visual del seguimiento** |
| `diag_track.png` | f_comb(t), mesetas detectadas y confianza |
| `meta.json` | Lo escribe el analista: factor de conversión elegido y textos del informe |
| `report.html` | Informe para publicar como Artifact (carga `plotly.min.js` hermano) |
| `report_offline.html` | Informe autónomo (~6 MB) que funciona sin internet |

Ejemplo completo y validado: [`reports/predator200_ejemplo/`](reports/predator200_ejemplo/) (Predator 200 cc, monocilíndrico 4T).

---

## Fundamento

### 1. Lo que se mide: f_comb

El sonido de un motor a régimen estable es periódico, así que su espectro es un **peine de líneas equiespaciadas**. El pipeline sigue en el tiempo el espaciado más fino de ese peine, **f_comb**, sin suponer todavía qué significa físicamente.

Saliencia de cada candidato f0 (por cuadro de STFT, sobre el espectro "blanqueado", es decir, solo las líneas que sobresalen del ruido):

```
S(f0) = H(f0) · E(f0)
H = nivel medio en los armónicos k·f0            (¿están llenos?)
E = fracción de la energía de líneas que cae     (¿explica el peine todas las líneas?)
    cerca de algún armónico
```

- f0 correcto → H alto, E ≈ 1
- múltiplo n·f0 → deja líneas sin explicar → E ≈ 1/n
- divisor f0/n → la mayoría de sus armónicos vacíos → H bajo

Después, Viterbi (continuidad temporal) + suavizado Savitzky-Golay, y validación independiente con la autocorrelación del espectro.

### 2. De f_comb a rpm: la decisión clave

Órdenes = múltiplos de la frecuencia de giro del cigüeñal, f_rot = rpm/60.
Frecuencia de explosión de un motor de encendido regular con N cilindros:

| Motor | Orden de explosión | f_explosión |
|---|---|---|
| 4 tiempos | N/2 | N · rpm / 120 |
| 2 tiempos | N | N · rpm / 60 |

f_comb = q · f_rot, y **rpm = 60 · f_comb / q**. Hipótesis posibles para q:

| q | Qué es f_comb | Cuándo ocurre |
|---|---|---|
| 0,5 | ciclo completo 4T (2 vueltas) | siempre en monocilíndrico 4T; en multicilindro si los cilindros no suenan iguales (lo normal en escape real) |
| 1 | una vuelta | 2T; desequilibrios de giro; multicilindro con cilindros parejos |
| N/2 (4T) o N (2T) | la explosión | cilindros muy parejos y nada más periódico: solo se ve la familia de explosión |

`analysis.json → hypotheses` lista las tres con: rpm de cada meseta, rpm baja, compatibilidad con el motor, `firing_family_advantage_db` (cuánto destacan los armónicos de explosión sobre el resto de líneas del peine si la hipótesis es cierta) y avisos.

**Regla de decisión (en este orden):**

1. Descarta las hipótesis `compatible: false` y las que dan rpm físicamente imposibles (< 300, > máximo del motor).
2. Si el audio incluye ralentí, la rpm baja debe caer en el ralentí conocido del motor. Es la prueba más fuerte.
3. Estructura de órdenes: en la hipótesis correcta los múltiplos del orden de explosión deben ser los más fuertes (`firing_family_advantage_db` > ~3 dB). Si q = 0,5 en un 4 cilindros, la explosión cae cada 4 líneas del peine; si q = 1, cada 2.
4. Plausibilidad de régimen (un turismo no va a 15 000 rpm; una moto de 600 cc sí puede).
5. Si aún hay duda, **pregunta al usuario** (ralentí real, rpm aproximadas durante la grabación) antes de publicar. Nunca elijas en silencio: el informe debe decir qué hipótesis se eligió y por qué.

> Lección del ejemplo Predator: el primer informe usó rpm = 60·f (lectura de multicilindro) y dio un ralentí de 690 rpm. Con el dato "monocilíndrico 4T" la lectura correcta es rpm = 120·f (ralentí ~1400, coincide con el GX200). Pide siempre la configuración del motor antes de concluir.

### 3. Limitaciones típicas de la grabación

- Micrófono de móvil: casi nada por debajo de ~50 Hz → los órdenes bajos (incluida a veces la propia explosión) no se ven; las rpm se deducen del espaciado de los armónicos altos.
- Opus/AAC de mensajería (16–32 kb/s): poca información útil por encima de ~500–1000 Hz.
- Transiciones rápidas (acelerones): la ventana de ~1 s las emborrona; la confianza baja (curva marcada en naranja en el informe).

---

## Procedimiento para Claude

Sigue estos pasos al recibir un audio nuevo. Usa la lista de tareas para que el usuario vea el progreso.

### Paso 0 · Datos del motor (preguntar si faltan)

Necesitas, antes de concluir rpm:

- **Nº de cilindros** y **2T/4T** (imprescindible).
- Modelo / cilindrada (para buscar ralentí y régimen máximo de fábrica si el usuario no los sabe).
- Ralentí conocido y rpm máximas (opcional pero muy útil: `--idle`, `--max-rpm`).
- Qué se hizo durante la grabación (ralentí, acelerones, régimen fijo, en marcha…) y dónde estaba el móvil.

Si el usuario no está, ejecuta el análisis igualmente, deja la hipótesis más plausible señalada como provisional y pregunta.

### Paso 1 · Preparar

```bash
pip install -r requirements.txt && python scripts/get_plotly.py   # si vendor/ no existe
python -m pytest -q                                                 # debe pasar (incluye regresión Predator)
cp <audio subido> audio/<nombre>.<ext>
```

### Paso 2 · Analizar

```bash
python -m engine_audio analyze audio/<nombre>.<ext> --name <nombre> \
    --cylinders N --strokes 4 [--idle A-B] [--max-rpm M] [--rpm-range A-B]
```

El rango de búsqueda de f_comb y la banda de armónicos se calculan solos a partir de cilindros, tiempos y `--rpm-range` (por defecto 500–8000 multicilindro, 700–6000 monocilindro). Ajusta a mano con `--fmin/--fmax/--band` solo si el diagnóstico lo pide.

### Paso 3 · Revisar el diagnóstico (obligatorio)

1. **Mira `diag_spectrogram.png`** (con la herramienta de lectura de imágenes). Las líneas rojas k·f_comb deben caer sobre las líneas reales del espectrograma, y las azules (k+½)·f_comb en huecos. Si las rojas caen entre líneas o solo coinciden una de cada 2–3, hay error de octava: ajusta `--fmin/--fmax` y repite.
2. **Mira `diag_track.png`**: curva continua, mesetas razonables, confianza ≥ 3 en las zonas estables.
3. En `analysis.json`:
   - `track_summary.hit_search_limit: true` → la búsqueda tocó el borde del rango: amplíalo.
   - `frac_low_conf` > 0,4 → audio pobre o muy transitorio; dilo en el informe.
   - `autocorr_spacing_hz` de cada meseta debe coincidir con `f_mean` (±2 %) o con un múltiplo/divisor sencillo (explícalo si es así).
4. Aplica la **regla de decisión** de arriba y elige `q`.

### Paso 4 · Escribir `reports/<nombre>/meta.json`

Toma como plantilla [`reports/predator200_ejemplo/meta.json`](reports/predator200_ejemplo/meta.json). Campos:

| Campo | Contenido |
|---|---|
| `title`, `page_title` | Título visible y título corto de pestaña (2–4 palabras) |
| `eyebrow` | Línea de contexto: motor, configuración, fuente del audio, duración |
| `lede` | 1–3 frases con el resultado principal (rpm de cada fase) |
| `q` | **Hipótesis elegida** (0.5, 1, N/2…). El informe convierte todo con rpm = 60·f_comb/q |
| `plateau_labels` | Nombre corto por meseta (`{"A": "ralentí", …}`) |
| `overlay_orders` | Órdenes del cigüeñal a superponer en el espectrograma (por defecto múltiplos de la explosión) |
| `orders_caption`, `orders_xmax` | Opcionales: texto y rango del espectro de órdenes |
| `interpretation` | Lista HTML: qué se ve, por qué se eligió q, cómo encaja con el motor, observaciones (resonancias, irregularidades) |
| `precision` | Párrafos: dispersión en mesetas, error estimado |
| `limits` | Lista: limitaciones del audio y del método |
| `method` | Lista: pasos del método (copiar del ejemplo y ajustar parámetros) |
| `formula` | Línea con las fórmulas usadas para este motor |

Redacta en español, frases cortas, sin relleno. Cada cifra del texto debe salir de `analysis.json`. Separa lo medido de lo inferido (p. ej., "apunta a una resonancia del escape").

### Paso 5 · Generar y comprobar el informe

```bash
python -m engine_audio report reports/<nombre>
```

Comprueba una vez el renderizado (p. ej. Playwright con el Chromium del entorno sobre `report_offline.html`, ancho de escritorio y de móvil): que salgan los 5 gráficos y no haya scroll horizontal.

### Paso 6 · Entregar

- **Artifact**: publica `reports/<nombre>/report.html` con `files: {"plotly.min.js": "reports/<nombre>/plotly.min.js"}`. La librería debe ir como fichero hermano: desde el móvil el CDN externo puede no cargar.
- **Fichero**: envía `report_offline.html` (funciona sin internet).
- Resume al usuario en el chat: rpm por fase, hipótesis elegida y por qué, y dudas pendientes.
- Haz commit de `audio/<nombre>.*`, `reports/<nombre>/{analysis.json,meta.json,diag_*.png,report.html}` (los `data.json`, `plotly.min.js` y `report_offline.html` están en `.gitignore`: se regeneran).

### Errores frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| "No se pudo cargar la librería de gráficos" en el Artifact | Plotly no publicado junto a la página | Publicar con `files` (Paso 6) |
| La publicación rechaza `plotly.min.js` (ESC, U+FFFD) | Bundle sin sanear | Usar `scripts/get_plotly.py`, que lo sanea |
| rpm la mitad o el doble de lo esperado | Hipótesis q equivocada | Revisar Paso 3.4; preguntar ralentí al usuario |
| Curva con saltos ×2/×3 en un tramo | Error de octava local | Acotar `--rpm-range`; revisar `diag_spectrogram.png` |
| Todo con confianza baja | Audio muy comprimido, viento, voz | Pedir grabación más cercana al escape, sin hablar, idealmente WAV |

---

## Estructura

```
engine_audio/
  io.py          decodificación (ffmpeg) y remuestreo a 4 kHz
  spectral.py    STFT, blanqueado, picos de largo plazo, autocorrelación del espectro
  tracking.py    saliencia de peine, Viterbi, mesetas
  interpret.py   hipótesis f_comb → rpm, espectro de órdenes, evidencias
  analyze.py     paso 1 (analysis.json, data.json, figuras de diagnóstico)
  report.py      paso 2 (report.html, report_offline.html)
  templates/report.html
scripts/get_plotly.py   descarga y sanea plotly.js 2.35.2
tests/                  señales sintéticas 1 y 4 cilindros + regresión Predator
audio/  reports/
```

## Consejos para grabar

- Móvil a 0,5–1 m del escape o del motor, sin viento y sin hablar.
- Mejor nota de voz sin comprimir o grabadora en WAV/FLAC que audio de WhatsApp.
- Mantén cada régimen 3–5 s (ralentí, medio, alto) para tener mesetas limpias.
- Apunta qué hiciste en cada momento (o dilo en voz alta *antes* de empezar).
