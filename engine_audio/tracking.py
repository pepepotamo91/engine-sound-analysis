"""Seguimiento del espaciado de la familia de armónicos (f_comb) a lo largo del tiempo.

f_comb es el espaciado más fino entre líneas espectrales del motor. Qué es
físicamente (giro del cigüeñal, ciclo de 4T o frecuencia de explosión) NO se
decide aquí: lo decide interpret.py con los datos del motor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as sig
from scipy.ndimage import maximum_filter1d

from .spectral import Spec, stft, whiten


@dataclass
class Track:
    t: np.ndarray        # [s]
    f: np.ndarray        # f_comb suavizada [Hz]
    f_raw: np.ndarray    # f_comb del Viterbi sin suavizar [Hz]
    conf: np.ndarray     # contraste (dB) armónicos vs puntos intermedios; <3 = poco fiable


def salience(C: np.ndarray, f: np.ndarray, f0s: np.ndarray, band=(45.0, 420.0),
             halfwidth_hz: float = 2.5, line_thr_db: float = 6.0) -> np.ndarray:
    """Saliencia de un peine k·f0 (C en dB blanqueados: ≥0, solo líneas).

        S(f0) = E(f0) · H(f0)
        H = nivel medio de C en los armónicos k·f0 (¿están llenos?)
        E = fracción de la energía de líneas de la banda que cae a ±halfwidth
            de algún armónico (¿explica el peine todas las líneas?)

    · f0 correcto: H alto y E ≈ 1.
    · múltiplo n·f0: deja líneas sin explicar -> E ≈ 1/n.
    · divisor f0/n: muchos armónicos vacíos -> H ≈ H/n.
    Si el motor tiene líneas de ciclo muy débiles, el peine elegido puede ser el de
    explosión en vez del de ciclo: interpret.py evalúa las dos lecturas.
    Solo cuentan como "líneas" los bins con C > line_thr_db (el ruido no puntúa).
    """
    J = C.shape[1]
    m = (f >= band[0]) & (f <= band[1])
    fb, Cb = f[m], C[m]
    df = fb[1] - fb[0]
    n = len(fb)
    w = max(1, int(round(halfwidth_hz / 2 / df)))
    Cmax = maximum_filter1d(Cb, size=2 * w + 1, axis=0)
    El = np.clip(Cb - line_thr_db, 0, None)
    cs = np.vstack([np.zeros((1, J)), np.cumsum(El, axis=0)])
    total = cs[-1] + 1e-9
    hw = max(1, int(round(halfwidth_hz / df)))
    S = np.full((J, len(f0s)), -np.inf)
    for i, f0 in enumerate(f0s):
        if f0 < 2.5 * halfwidth_hz:
            continue
        k = np.arange(np.ceil(band[0] / f0), np.floor(band[1] / f0) + 1)
        if len(k) < 3:
            continue
        idx = np.round((k * f0 - fb[0]) / df).astype(int)
        idx = idx[(idx >= 0) & (idx < n)]
        lo = np.clip(idx - hw, 0, n)
        hi = np.clip(idx + hw + 1, 0, n)
        E = np.minimum((cs[hi] - cs[lo]).sum(0) / total, 1.0)
        H = Cmax[idx].mean(0)
        S[:, i] = E * H
    return S


def viterbi(S: np.ndarray, max_jump_bins: int) -> np.ndarray:
    """Camino de máxima saliencia con salto acotado entre cuadros consecutivos."""
    J, K = S.shape
    D = S[0].copy()
    B = np.zeros((J, K), dtype=int)
    idx = np.arange(K)
    for j in range(1, J):
        best = np.full(K, -np.inf)
        arg = np.zeros(K, dtype=int)
        for d in range(-max_jump_bins, max_jump_bins + 1):
            src = idx + d
            ok = (src >= 0) & (src < K)
            cand = np.full(K, -np.inf)
            cand[ok] = D[src[ok]]
            better = cand > best
            best[better] = cand[better]
            arg[better] = src[better]
        D = best + S[j]
        B[j] = arg
    p = np.zeros(J, dtype=int)
    p[-1] = int(np.argmax(D))
    for j in range(J - 1, 0, -1):
        p[j - 1] = B[j, p[j]]
    return p


def track_comb(x: np.ndarray, fs: float, fmin: float = 8.0, fmax: float = 60.0,
               band=(45.0, 420.0), win_s: float = 0.8, hop_s: float = 0.1,
               step_rel: float = 0.002, max_jump_rel_per_s: float = 0.6,
               smooth_frames: int = 9) -> tuple[Track, Spec]:
    """Sigue f_comb(t).

    fmin/fmax   rango de búsqueda de f_comb [Hz] (rejilla logarítmica, paso step_rel)
    band        banda donde se buscan los armónicos [Hz]; para multicilindro a
                altas rpm hay que subirla (cli.py la calcula a partir del motor)
    max_jump_rel_per_s  máxima variación relativa de f_comb por segundo
                (0.6 = 60 %/s; súbela para acelerones en vacío muy bruscos)
    """
    win_s = max(win_s, 5.0 / fmin)  # resolución suficiente para separar líneas a fmin
    overlap = 1 - hop_s / win_s
    nfft = int(2 ** np.ceil(np.log2(max(2 ** 15, win_s * fs * 8))))
    spec = stft(x, fs, win_s, overlap, nfft=nfft, fmax=min(band[1] * 1.2, fs / 2))
    L = 10 * np.log10(spec.P + 1e-20)
    C = whiten(L, spec.f)
    f0s = np.exp(np.arange(np.log(fmin), np.log(fmax) + step_rel / 2, step_rel))
    S = salience(C, spec.f, f0s, band, halfwidth_hz=2.0 / win_s)
    mj = max(1, int(round(np.log1p(max_jump_rel_per_s * hop_s) / step_rel)))
    p = viterbi(S, mj)
    raw = f0s[p]
    conf = S[np.arange(len(p)), p]
    sm = sig.savgol_filter(raw, smooth_frames, 2) if len(raw) > smooth_frames else raw
    return Track(t=spec.t, f=sm, f_raw=raw, conf=conf), spec


def find_plateaus(tr: Track, tol_rel: float = 0.05, min_dur_s: float = 0.8,
                  min_conf: float = 3.0, trim_s: float = 0.2) -> list[dict]:
    """Tramos estables: f_comb dentro de ±tol_rel/2 de su media durante ≥ min_dur_s.

    Búsqueda voraz de izquierda a derecha: desde cada inicio con confianza
    aceptable se extiende el tramo mientras (max-min)/media ≤ tol_rel y la
    confianza mediana del tramo sea ≥ min_conf.
    """
    f, t, c = tr.f, tr.t, tr.conf
    out = []
    j = 0
    n = len(f)
    while j < n:
        if c[j] < min_conf:
            j += 1
            continue
        k = j + 1
        lo = hi = f[j]
        while k < n:
            lo2, hi2 = min(lo, f[k]), max(hi, f[k])
            if (hi2 - lo2) / ((hi2 + lo2) / 2) > tol_rel:
                break
            lo, hi = lo2, hi2
            k += 1
        # recorta cola de baja confianza
        while k - 1 > j and c[k - 1] < min_conf:
            k -= 1
        a, b = t[j] + trim_s, t[k - 1] - trim_s
        if b - a >= min_dur_s and np.median(c[j:k]) >= min_conf:
            m = (t >= a) & (t <= b)
            out.append({"a": round(float(a), 2), "b": round(float(b), 2),
                        "f_mean": float(f[m].mean()), "f_std": float(f[m].std()),
                        "f_min": float(f[m].min()), "f_max": float(f[m].max()),
                        "conf_mean": float(c[m].mean())})
            j = k
        else:
            j += 1
    return out


def low_end(tr: Track, min_conf: float = 2.0, pct: float = 5.0) -> float:
    """f_comb baja representativa (percentil ``pct`` de los cuadros fiables): suele ser el ralentí."""
    v = tr.f[tr.conf >= min_conf]
    return float(np.percentile(v if len(v) else tr.f, pct))
