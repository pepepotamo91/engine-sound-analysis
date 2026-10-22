# Instrucciones para Claude

Este repo analiza grabaciones de motores (FFT, waterfall 3D, rpm, órdenes) y genera un informe HTML.

Cuando el usuario aporte un audio nuevo, sigue **al pie de la letra** la sección
"Procedimiento para Claude" de `README.md` (pasos 0 a 6). Puntos que no se pueden saltar:

- Pregunta nº de cilindros y 2T/4T antes de concluir rpm; con eso se elige la conversión f_comb → rpm.
- Mira `diag_spectrogram.png` y `diag_track.png` antes de escribir `meta.json`.
- El informe debe decir qué hipótesis (`q`) se eligió y por qué; nunca la elijas en silencio.
- Al publicar `report.html` como Artifact, publica `plotly.min.js` como fichero hermano (`files`).
- `python -m pytest -q` debe pasar antes de hacer commit si tocas `engine_audio/`.

Idioma de informes y mensajes: español. Código y comentarios: español.
