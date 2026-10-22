"""Paso 1: análisis. Genera analysis.json, data.json y figuras de diagnóstico.

No decide qué significa f_comb: deja las hipótesis evaluadas en analysis.json
para que el analista elija ``q`` en meta.json (ver README).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import interpret
from .io import load
from .spectral import comb_spacing_autocorr, long_term_peaks, stft
from .tracking import find_plateaus, low_end, track_comb


def _r(a, n=2):
    return np.round(np.asarray(a, dtype=float), n).tolist()


def run(audio_path: str | Path, out_dir: str | Path, cylinders: int, strokes: int,
        idle_range: tuple[float, float] | None = None, max_rpm: float | None = None,
        rpm_range: tuple[float, float] | None = None,
        fmin: float | None = None, fmax: float | None = None, band: tuple[float, float] | None = None,
        wf_fmax: float | None = None, target_fs: float = 4000.0, figures: bool = True) -> dict:
    """Ejecuta el análisis completo y escribe los resultados en ``out_dir``.

    rpm_range  rango de rpm plausible del motor (por defecto 500–8000 multicilindro,
               700–6000 monocilindro). Con él se fijan el rango de búsqueda de
               f_comb y la banda de armónicos si no se dan fmin/fmax/band.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    au = load(audio_path, target_fs=target_fs)
    x, fs = au.x, au.fs
    sp = search_params(cylinders, strokes, rpm_range, fs)
    fmin = fmin or sp["fmin"]
    fmax = fmax or sp["fmax"]
    band = tuple(band or sp["band"])
    wf_fmax = wf_fmax or sp["wf_fmax"]
    tr, spec_tr = track_comb(x, fs, fmin=fmin, fmax=fmax, band=band)
    plateaus = find_plateaus(tr)

    # espectro de órdenes del peine en cada meseta
    spec_o = stft(x, fs, 0.8, 0.875, nfft=2 ** 15, fmax=max(wf_fmax, band[1]) * 1.5)
    ospec = []
    for i, p in enumerate(plateaus):
        kmax = float(np.clip(np.floor(0.9 * band[1] / p["f_mean"]), 6, 48))
        ks, O = interpret.comb_order_spectrum(spec_o, tr.t, tr.f, p["a"], p["b"], k_max=kmax)
        p["id"] = chr(ord("A") + i)
        p["subharmonic_contrast_db"] = round(interpret.subharmonic_contrast(ks, O), 1)
        ospec.append({"id": p["id"], "k": _r(ks, 3), "db": _r(O, 1)})

    # validación independiente: autocorrelación del espectro
    spec_ac = stft(x, fs, 1.0, 0.75, nfft=2 ** 14)
    ac = comb_spacing_autocorr(spec_ac, band=band, lag_range=(fmin, fmax))
    for p in plateaus:
        m = (spec_ac.t >= p["a"]) & (spec_ac.t <= p["b"])
        v = ac[m]
        v = v[np.isfinite(v)]
        p["autocorr_spacing_hz"] = round(float(np.median(v)), 2) if len(v) else None

    f_low = low_end(tr)
    hyps = interpret.evaluate(cylinders, strokes, plateaus, ospec, f_low, idle_range, max_rpm)

    analysis = {
        "audio": {"file": str(au.path), "codec": au.codec, "bitrate_kbps": au.bitrate_kbps,
                  "original_fs": au.original_fs, "analysis_fs": fs, "duration_s": round(au.duration, 2),
                  "rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)), 1)},
        "engine": {"cylinders": cylinders, "strokes": strokes,
                   "firing_order": interpret.firing_order(cylinders, strokes),
                   "idle_range": idle_range, "max_rpm": max_rpm},
        "params": {"fmin": fmin, "fmax": fmax, "band": list(band), "rpm_range": rpm_range},
        "long_term_peaks": long_term_peaks(x, fs, fmax=min(1000, fs / 2 - 50)),
        "track_summary": {"f_comb_min": round(float(tr.f.min()), 2), "f_comb_max": round(float(tr.f.max()), 2),
                          "f_comb_low_p5": round(f_low, 2),
                          "f_comb_first_1s": round(float(np.median(tr.f[tr.t <= tr.t[0] + 1.0])), 2),
                          "f_comb_last_1s": round(float(np.median(tr.f[tr.t >= tr.t[-1] - 1.0])), 2),
                          "conf_median": round(float(np.median(tr.conf)), 1),
                          "frac_low_conf": round(float(np.mean(tr.conf < 3)), 2),
                          "hit_search_limit": bool(tr.f_raw.min() <= fmin + 0.2 or tr.f_raw.max() >= fmax - 0.2)},
        "plateaus": plateaus,
        "hypotheses": hyps,
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False))

    # datos para el informe (en unidades de f_comb; el informe aplica el factor elegido)
    sw = stft(x, fs, 1.0, 0.75, nfft=int(fs), fmax=wf_fmax)
    ref = float(10 * np.log10(sw.P.max() + 1e-20))
    sg = stft(x, fs, 1.2, 11 / 12, nfft=2 ** 13, fmax=min(wf_fmax, max(300.0, 12 * fmax)))
    fc_wf = np.interp(sw.t, tr.t, tr.f)
    kk = np.arange(0.25, 16.0001, 0.025)
    Lw = sw.db(ref)
    OW = [np.interp(kk * fc_wf[j], sw.f, Lw[:, j]) for j in range(len(sw.t))]
    data = {
        "dur": round(au.duration, 2),
        "wf": {"f": _r(sw.f, 1), "t": _r(sw.t), "z": _r(Lw.T, 1), "fcomb": _r(fc_wf, 3)},
        "sg": {"f": _r(sg.f), "t": _r(sg.t), "z": _r(sg.db(ref), 1)},
        "ow": {"k": _r(kk, 3), "t": _r(sw.t), "z": _r(OW, 1)},
        "track": {"t": _r(tr.t), "fcomb": _r(tr.f, 3), "conf": _r(tr.conf, 1)},
        "orders": ospec,
    }
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")))

    if figures:
        _figures(out, sg, ref, tr, plateaus)
    return analysis


def search_params(cylinders: int, strokes: int, rpm_range, fs: float) -> dict:
    """Rango de búsqueda de f_comb y banda de armónicos a partir del motor.

    f_comb más baja posible: ciclo 4T a rpm mínima (rpm/120) o giro en 2T (rpm/60).
    f_comb más alta posible: frecuencia de explosión a rpm máxima.
    """
    if rpm_range is None:
        rpm_range = (700.0, 6000.0) if cylinders == 1 else (500.0, 8000.0)
    lo, hi = rpm_range
    fo = interpret.firing_order(cylinders, strokes)
    fmin = 0.85 * lo / (120 if strokes == 4 else 60)
    fmax = 1.15 * hi * fo / 60
    fmin = max(fmin, 3.0)
    top = 0.45 * fs
    band_hi = float(np.clip(10 * fmax, 420.0, top))
    return {"fmin": round(fmin, 2), "fmax": round(min(fmax, band_hi / 3), 2),
            "band": (45.0, round(band_hi, 1)), "wf_fmax": float(min(max(500.0, 6 * fmax), top))}


def _figures(out: Path, sg, ref, tr, plateaus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = sg.db(ref)
    fig, ax = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    for a in ax:
        a.pcolormesh(sg.t, sg.f, L, shading="auto", cmap="gray_r", vmin=np.percentile(L, 50))
        a.set_ylabel("Hz")
    for k in range(1, 30):
        ax[1].plot(tr.t, tr.f * k, "r", lw=0.6, alpha=0.7)
        ax[1].plot(tr.t, tr.f * (k + 0.5), "b:", lw=0.5, alpha=0.5)
    ax[0].set_title("Espectrograma (sin superponer)")
    ax[1].set_title("Rojo: k·f_comb   Azul punteado: (k+½)·f_comb  -> el rojo debe caer sobre las líneas")
    ax[1].set_xlabel("s")
    ax[1].set_ylim(0, sg.f.max())
    fig.tight_layout()
    fig.savefig(out / "diag_spectrogram.png", dpi=80)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax[0].plot(tr.t, tr.f_raw, ".", ms=2, color="0.6", label="Viterbi")
    ax[0].plot(tr.t, tr.f, "k", lw=1.5, label="suavizado")
    for p in plateaus:
        ax[0].axvspan(p["a"], p["b"], alpha=0.15)
        ax[0].text((p["a"] + p["b"]) / 2, p["f_mean"], f"{p['id']} {p['f_mean']:.2f} Hz", ha="center", va="bottom")
    ax[0].set_ylabel("f_comb [Hz]")
    ax[0].legend()
    ax[1].plot(tr.t, tr.conf, "k", lw=1)
    ax[1].axhline(3, color="r", lw=0.8, ls="--")
    ax[1].set_ylabel("confianza [dB]")
    ax[1].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(out / "diag_track.png", dpi=80)
    plt.close(fig)
