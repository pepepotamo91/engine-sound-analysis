"""Señales sintéticas de motor con rpm conocida, para validar el pipeline."""
import numpy as np


def engine_signal(rpm_fn, dur, fs=8000, cylinders=4, strokes=4, cyl_spread=0.15,
                  harmonics=40, noise=0.05, seed=0):
    """Tren de explosiones filtrado: cada cilindro con amplitud ligeramente distinta.

    rpm_fn(t) -> rpm. cyl_spread = dispersión relativa de amplitud entre cilindros
    (genera líneas a la frecuencia de ciclo rpm/120 además de la de explosión).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fs)) / fs
    rpm = rpm_fn(t)
    rev = np.cumsum(rpm / 60 / fs)                     # vueltas acumuladas
    cyc = rev / (2 if strokes == 4 else 1)             # ciclos acumulados
    amps = 1 + cyl_spread * rng.standard_normal(cylinders)
    x = np.zeros_like(t)
    # suma de armónicos del ciclo con ganancias por cilindro (serie de Fourier del tren)
    for k in range(1, harmonics + 1):
        c = np.sum(amps * np.exp(-2j * np.pi * k * np.arange(cylinders) / cylinders))
        x += np.abs(c) * np.cos(2 * np.pi * k * cyc + np.angle(c)) / k ** 0.5
    x /= np.abs(x).max()
    x += noise * rng.standard_normal(len(t))
    return x, fs
