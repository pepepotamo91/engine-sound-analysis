import json
import subprocess
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

from engine_audio import analyze
from synth import engine_signal

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path, x, fs):
    p = tmp_path / "s.wav"
    wavfile.write(p, fs, (x * 0.5 * 32767).astype(np.int16))
    return p


def _rpm_profile(t):
    # ralentí 800 -> rampa -> meseta 2400 -> vuelta a 1500
    return np.interp(t, [0, 3, 5, 9, 11, 15], [800, 800, 2400, 2400, 1500, 1500])


def _matching(a, expected, tol=0.015):
    """Hipótesis compatibles, sin avisos, cuyas mesetas contienen todas las rpm esperadas."""
    ok = []
    for h in a["hypotheses"]:
        if not h["compatible"] or h["flags"]:
            continue
        if all(any(abs(r - e) <= tol * e for r in h["plateau_rpm"]) for e in expected):
            ok.append(h)
    return ok


def test_four_cylinder_4t_even(tmp_path):
    """Cilindros casi iguales: el peine es la explosión (orden 2) o el ciclo."""
    x, fs = engine_signal(_rpm_profile, 15, cylinders=4, strokes=4, cyl_spread=0.1)
    a = analyze.run(_write(tmp_path, x, fs), tmp_path / "out", 4, 4, idle_range=(650, 950),
                    max_rpm=7000, rpm_range=(600, 7000), figures=False)
    good = _matching(a, [800, 2400, 1500])
    assert len(good) == 1, json.dumps(a["hypotheses"], indent=1)


def test_four_cylinder_4t_uneven(tmp_path):
    """Cilindros desiguales: aparecen las líneas de ciclo (rpm/120) y la explosión cada 4."""
    x, fs = engine_signal(_rpm_profile, 15, cylinders=4, strokes=4, cyl_spread=0.5, seed=3)
    a = analyze.run(_write(tmp_path, x, fs), tmp_path / "out", 4, 4, idle_range=(650, 950),
                    max_rpm=7000, rpm_range=(600, 7000), figures=False)
    good = _matching(a, [800, 2400, 1500])
    assert len(good) == 1, json.dumps(a["hypotheses"], indent=1)


def test_single_cylinder_4t(tmp_path):
    x, fs = engine_signal(_rpm_profile, 15, cylinders=1, strokes=4)
    a = analyze.run(_write(tmp_path, x, fs), tmp_path / "out", 1, 4, figures=False)
    hyp = {h["q"]: h for h in a["hypotheses"]}
    assert hyp[1.0]["compatible"] is False
    assert any(abs(r - 2400) < 25 for r in hyp[0.5]["plateau_rpm"])


def test_predator_example(tmp_path):
    """Regresión: el audio real del Predator 200 debe dar ~2000 y ~3000 rpm."""
    a = analyze.run(ROOT / "audio" / "predator200_ejemplo.opus", tmp_path / "out", 1, 4,
                    idle_range=(1250, 1550), figures=False)
    r = [round(p["f_mean"] * 120) for p in a["plateaus"]]
    assert any(abs(v - 2000) < 60 for v in r), r
    assert any(abs(v - 2997) < 30 for v in r), r
