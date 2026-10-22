"""Hipótesis f_comb -> rpm según la configuración del motor, y espectro de órdenes.

Notación: un "orden" n es un múltiplo de la frecuencia de giro del cigüeñal
(f_rot = rpm/60). La frecuencia de explosión de un motor de encendido regular es
    4T: orden N/2      2T: orden N         (N = nº de cilindros)

f_comb = q · f_rot, con q = orden de la línea más fina de la familia. Hipótesis:
    q = 0.5  ciclo completo de 4T (2 vueltas): aparece si los cilindros no son
             idénticos (siempre en un monocilíndrico 4T)
    q = 1    una vuelta (desequilibrio, 2T, o multicilindro con cilindros parejos)
    q = Nexp frecuencia de explosión (cilindros idénticos y nada más periódico)
rpm = 60 · f_comb / q
"""
from __future__ import annotations

import numpy as np

from .spectral import Spec


def firing_order(cylinders: int, strokes: int) -> float:
    return cylinders / 2 if strokes == 4 else float(cylinders)


def hypotheses(cylinders: int, strokes: int) -> list[dict]:
    """Las tres lecturas posibles de f_comb, marcando las incompatibles con el motor."""
    fo = firing_order(cylinders, strokes)
    cands = {0.5: "ciclo 4T (2 vueltas)", 1.0: "giro del cigüeñal (1 vuelta)"}
    cands.setdefault(fo, f"explosión (orden {fo:g})")
    out = []
    for q in sorted(cands):
        names = [cands[q]]
        if q == fo and cands[q] != f"explosión (orden {fo:g})":
            names.append(f"explosión (orden {fo:g})")
        r = fo / q
        ok = r >= 1 - 1e-9 and abs(r - round(r)) < 1e-9 and not (strokes == 2 and q == 0.5)
        out.append({"q": q, "rpm_factor": 60.0 / q, "label": " = ".join(names),
                    "firing_every": int(round(r)) if ok else None, "compatible": ok})
    return out


def comb_order_spectrum(spec: Spec, t_tr: np.ndarray, f_tr: np.ndarray, a: float, b: float,
                        k_max: float = 20.0, step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Nivel medio (dB, normalizado al máximo) en múltiplos k·f_comb entre a y b [s].

    k va en unidades del peine (k=1 -> f_comb). Incluye fracciones (paso ``step``)
    para ver si hay sub-armónicos.
    """
    L = 10 * np.log10(spec.P + 1e-20)
    js = np.where((spec.t >= a) & (spec.t <= b))[0]
    ks = np.arange(step, k_max + 1e-9, step)
    rows = []
    for j in js:
        fc = np.interp(spec.t[j], t_tr, f_tr)
        rows.append(np.interp(ks * fc, spec.f, L[:, j]))
    O = np.mean(rows, axis=0)
    return ks, O - O.max()


def score_hypothesis(ks: np.ndarray, O: np.ndarray, firing_every: int, k_lo: float = 2.0) -> float | None:
    """dB de ventaja de los armónicos de explosión frente al resto de líneas del peine.

    Solo con k ≥ k_lo (el micro del móvil apenas capta < 50 Hz). None si la
    hipótesis no predice diferencia (firing_every == 1).
    """
    if firing_every <= 1:
        return None
    integer = np.isclose(ks, np.round(ks)) & (ks >= k_lo)
    kk = np.round(ks[integer]).astype(int)
    on = O[integer][kk % firing_every == 0]
    off = O[integer][kk % firing_every != 0]
    if len(on) == 0 or len(off) == 0:
        return None
    return float(on.mean() - off.mean())


def subharmonic_contrast(ks: np.ndarray, O: np.ndarray, k_lo: float = 2.0) -> float:
    """dB de las líneas enteras del peine frente a las de medio orden (k+0.5)."""
    integer = np.isclose(ks, np.round(ks)) & (ks >= k_lo)
    half = np.isclose(ks % 1, 0.5) & (ks >= k_lo)
    return float(O[integer].mean() - O[half].mean())


def evaluate(cylinders: int, strokes: int, plateaus: list[dict], order_spectra: list[dict],
             f_low: float, idle_range: tuple[float, float] | None, max_rpm: float | None) -> list[dict]:
    """Tabla de evidencias por hipótesis para que el analista (Claude) decida.

    f_low: f_comb baja representativa (≈ ralentí si la grabación lo incluye).
    """
    res = []
    for h in hypotheses(cylinders, strokes):
        fac = h["rpm_factor"]
        ev = {**h, "plateau_rpm": [round(p["f_mean"] * fac) for p in plateaus],
              "low_rpm": round(f_low * fac)}
        flags = []
        if not h["compatible"]:
            flags.append(f"incompatible con {cylinders} cil. {strokes}T: la explosión (orden "
                         f"{firing_order(cylinders, strokes):g}) no sería múltiplo del peine")
        if idle_range:
            lo, hi = idle_range
            ev["low_in_idle_range"] = bool(lo * 0.85 <= f_low * fac <= hi * 1.15)
            if not ev["low_in_idle_range"]:
                flags.append(f"rpm baja {f_low * fac:.0f} fuera del ralentí esperado {lo:.0f}–{hi:.0f} "
                             "(irrelevante si la grabación no incluye ralentí)")
        if max_rpm and plateaus and max(ev["plateau_rpm"]) > max_rpm * 1.05:
            flags.append(f"supera el máximo del motor ({max_rpm:.0f} rpm)")
        if f_low * fac < 300:
            flags.append("rpm < 300: físicamente implausible")
        if h["firing_every"]:
            sc = [score_hypothesis(np.array(o["k"]), np.array(o["db"]), h["firing_every"]) for o in order_spectra]
            sc = [v for v in sc if v is not None]
            ev["firing_family_advantage_db"] = round(float(np.mean(sc)), 1) if sc else None
        else:
            ev["firing_family_advantage_db"] = None
        ev["flags"] = flags
        res.append(ev)
    return res
