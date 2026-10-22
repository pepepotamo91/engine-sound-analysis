"""Paso 2: informe HTML a partir de analysis.json + data.json + meta.json.

Genera dos variantes en el directorio del análisis:
  report.html          página para publicar como Artifact de Claude: carga
                       ``plotly.min.js`` como fichero hermano (hay que publicarlo
                       junto a la página con ``files``), con CDN de respaldo.
  report_offline.html  documento completo y autónomo, Plotly incrustado.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("templates") / "report.html"
PLOTLY = ROOT / "vendor" / "plotly.min.js"
PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"

REQUIRED_META = ["title", "eyebrow", "lede", "q", "interpretation", "precision", "limits", "method", "formula"]

MIME = {".opus": "ogg", ".ogg": "ogg", ".mp3": "mpeg", ".m4a": "mp4", ".aac": "aac", ".wav": "wav", ".flac": "flac"}


def _js(obj) -> str:
    # evita cerrar el <script> desde dentro de los datos
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def build(out_dir: str | Path, meta_path: str | Path | None = None, embed_audio_max_mb: float = 3.0) -> dict:
    out = Path(out_dir)
    analysis = json.loads((out / "analysis.json").read_text())
    data = json.loads((out / "data.json").read_text())
    meta = json.loads(Path(meta_path or out / "meta.json").read_text())
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"meta.json incompleto, faltan: {missing}")

    audio = Path(analysis["audio"]["file"])
    if not audio.is_absolute():
        audio = ROOT / audio
    if audio.exists() and audio.stat().st_size <= embed_audio_max_mb * 1e6:
        data["audio"] = base64.b64encode(audio.read_bytes()).decode()
        meta.setdefault("audio_mime", MIME.get(audio.suffix.lower(), "ogg"))

    tpl = TEMPLATE.read_text()
    page = (tpl.replace("__TITLE__", meta.get("page_title", meta["title"]))
               .replace("__DATA__", _js(data))
               .replace("__META__", _js(meta))
               .replace("__ANALYSIS__", _js(analysis)))

    art = page.replace("__PLOTLY__",
                       '<script src="plotly.min.js"></script>\n'
                       f"<script>if(!window.Plotly)document.write('<script src=\"{PLOTLY_CDN}\"><\\/script>')</script>")
    (out / "report.html").write_text(art)
    written = {"report.html": out / "report.html"}

    if PLOTLY.exists():
        lib = PLOTLY.read_text().replace("</script", "<\\/script")
        off = page.replace("__PLOTLY__", f"<script>{lib}</script>")
        doc = ('<!doctype html><html lang="es"><head><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
               f"</head><body>{off}</body></html>")
        (out / "report_offline.html").write_text(doc)
        written["report_offline.html"] = out / "report_offline.html"
        # copia de plotly junto al informe para publicar report.html
        (out / "plotly.min.js").write_text(PLOTLY.read_text())
    else:
        print(f"Aviso: falta {PLOTLY}; ejecuta `python scripts/get_plotly.py`. "
              "Solo se generó report.html (usará el CDN).")
    return written
