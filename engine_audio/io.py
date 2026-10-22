"""Carga de audio: cualquier formato que entienda ffmpeg -> mono float64."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sig


@dataclass
class Audio:
    x: np.ndarray          # señal mono, float64, ~[-1, 1]
    fs: float              # frecuencia de muestreo [Hz]
    path: Path
    original_fs: float
    codec: str
    bitrate_kbps: float | None

    @property
    def duration(self) -> float:
        return len(self.x) / self.fs


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def load(path: str | Path, target_fs: float = 4000.0) -> Audio:
    """Decodifica con ffmpeg a WAV mono y remuestrea a ``target_fs``.

    ``target_fs`` = 4 kHz deja banda útil hasta 2 kHz, suficiente para los
    armónicos de motor y mucho más rápido de procesar que 48 kHz.
    """
    path = Path(path)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg no está instalado (apt install ffmpeg)")
    info = probe(path)
    st = next(s for s in info["streams"] if s.get("codec_type") == "audio")
    ofs = float(st.get("sample_rate", 48000))
    br = info.get("format", {}).get("bit_rate")
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", str(int(ofs)), str(wav)],
            check=True,
        )
        fs, x = wavfile.read(wav)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float64)
    # remuestreo racional con filtro antialiasing
    from fractions import Fraction
    fr = Fraction(int(target_fs), int(fs)).limit_denominator(1000)
    y = sig.resample_poly(x, fr.numerator, fr.denominator)
    return Audio(
        x=y, fs=fs * fr.numerator / fr.denominator, path=path, original_fs=ofs,
        codec=st.get("codec_name", "?"), bitrate_kbps=(float(br) / 1000 if br else None),
    )
