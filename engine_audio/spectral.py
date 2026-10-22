"""Espectrogramas (STFT) y utilidades espectrales."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as sig


@dataclass
class Spec:
    f: np.ndarray      # [Hz]
    t: np.ndarray      # [s] centro de cada ventana
    P: np.ndarray      # potencia, forma (len(f), len(t))

    def db(self, ref: float | None = None, floor: float = -100.0) -> np.ndarray:
        L = 10 * np.log10(self.P + 1e-20)
        if ref is not None:
            L = L - ref
        return np.maximum(L, floor)


def stft(x: np.ndarray, fs: float, win_s: float, overlap: float, nfft: int | None = None,
         fmax: float | None = None) -> Spec:
    """STFT con ventana Hann.

    win_s   longitud de ventana en s (resolución ≈ 1/win_s Hz)
    overlap fracción de solape (0–0.95)
    nfft    zero-padding (por defecto: siguiente potencia de 2 ≥ 8·ventana)
    """
    n = int(round(win_s * fs))
    nov = int(round(n * overlap))
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(n * 8)))
    nfft = max(nfft, n)
    f, t, P = sig.spectrogram(x, fs, window="hann", nperseg=n, noverlap=nov, nfft=nfft,
                              scaling="spectrum", mode="psd")
    if fmax is not None:
        m = f <= fmax
        f, P = f[m], P[m]
    return Spec(f=f, t=t, P=P)


def whiten(L: np.ndarray, f: np.ndarray, smooth_hz: float = 30.0) -> np.ndarray:
    """Resta a cada cuadro (columna, en dB) su envolvente suavizada y recorta a ≥0.

    Deja solo las líneas espectrales que sobresalen del ruido de banda ancha.
    """
    df = f[1] - f[0]
    w = int(smooth_hz / df) | 1
    w = max(w, 5)
    E = sig.savgol_filter(L, w, 2, axis=0)
    return np.clip(L - E, 0, None)


def long_term_peaks(x: np.ndarray, fs: float, fmin=15.0, fmax=1000.0, res_hz=0.5,
                    prominence_db=6.0, top=30) -> list[dict]:
    """Picos del espectro medio (Welch) — inventario rápido de líneas."""
    n = int(fs / res_hz)
    n = min(n, len(x))
    f, P = sig.welch(x, fs, nperseg=n, noverlap=n // 2)
    L = 10 * np.log10(P + 1e-20)
    m = (f >= fmin) & (f <= fmax)
    pk, pr = sig.find_peaks(L[m], prominence=prominence_db)
    order = np.argsort(L[m][pk])[::-1][:top]
    out = [{"hz": float(f[m][pk][i]), "db": float(L[m][pk][i]),
            "prominence_db": float(pr["prominences"][i])} for i in order]
    return sorted(out, key=lambda d: d["hz"])


def comb_spacing_autocorr(spec: Spec, band=(40.0, 450.0), lag_range=(5.0, 60.0)) -> np.ndarray:
    """Espaciado dominante entre líneas por autocorrelación del espectro blanqueado.

    Método independiente del tracker, sirve para validar. Devuelve, por cuadro,
    el retardo (Hz) del mayor pico de autocorrelación en ``lag_range``.
    """
    L = 10 * np.log10(spec.P + 1e-20)
    m = (spec.f >= band[0]) & (spec.f <= band[1])
    C = whiten(L[m], spec.f[m])
    df = spec.f[1] - spec.f[0]
    out = np.full(len(spec.t), np.nan)
    for j in range(len(spec.t)):
        c = C[:, j]
        ac = np.correlate(c, c, "full")[len(c) - 1:]
        lags = np.arange(len(ac)) * df
        r = (lags >= lag_range[0]) & (lags <= lag_range[1])
        if ac[0] <= 0:
            continue
        pk, _ = sig.find_peaks(ac[r])
        if len(pk):
            out[j] = lags[r][pk[np.argmax(ac[r][pk])]]
    return out
