"""CLI.

    python -m engine_audio analyze audio/mi_motor.opus --name mi_motor --cylinders 4 --strokes 4 \
        [--idle 650-900] [--max-rpm 6500] [--rpm-range 600-7000] [--fmin F --fmax F --band A-B]
    python -m engine_audio report reports/mi_motor
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _range(s: str | None):
    if not s:
        return None
    a, b = s.replace("–", "-").split("-")
    return float(a), float(b)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="engine_audio")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="paso 1: FFT, seguimiento, mesetas e hipótesis")
    a.add_argument("audio")
    a.add_argument("--name", help="carpeta de salida en reports/ (por defecto: nombre del audio)")
    a.add_argument("--cylinders", type=int, required=True)
    a.add_argument("--strokes", type=int, choices=(2, 4), required=True)
    a.add_argument("--idle", help="ralentí esperado, p. ej. 700-900")
    a.add_argument("--max-rpm", type=float)
    a.add_argument("--rpm-range", help="rpm mín-máx plausibles para acotar la búsqueda")
    a.add_argument("--fmin", type=float)
    a.add_argument("--fmax", type=float)
    a.add_argument("--band", help="banda de armónicos en Hz, p. ej. 45-900")
    a.add_argument("--no-figures", action="store_true")

    r = sub.add_parser("report", help="paso 2: genera report.html y report_offline.html")
    r.add_argument("dir")
    r.add_argument("--meta", help="ruta a meta.json (por defecto <dir>/meta.json)")

    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        from .analyze import run
        name = args.name or Path(args.audio).stem
        out = ROOT / "reports" / name
        res = run(args.audio, out, args.cylinders, args.strokes, idle_range=_range(args.idle),
                  max_rpm=args.max_rpm, rpm_range=_range(args.rpm_range), fmin=args.fmin,
                  fmax=args.fmax, band=_range(args.band), figures=not args.no_figures)
        print(f"Resultados en {out}")
        print(json.dumps({"params": res["params"], "track_summary": res["track_summary"]}, ensure_ascii=False))
        print("\nMesetas (f_comb):")
        for p in res["plateaus"]:
            print(f"  {p['id']}: {p['a']:.1f}–{p['b']:.1f} s  f_comb={p['f_mean']:.2f}±{p['f_std']:.2f} Hz  "
                  f"conf={p['conf_mean']:.1f}  autocorr={p['autocorr_spacing_hz']}  sub-armónicos={p['subharmonic_contrast_db']} dB")
        print("\nHipótesis:")
        for h in res["hypotheses"]:
            print(f"  q={h['q']:g} ({h['label']}): rpm = {h['rpm_factor']:g}·f_comb -> mesetas {h['plateau_rpm']}, "
                  f"baja {h['low_rpm']}, ventaja familia explosión {h['firing_family_advantage_db']} dB")
            for fl in h["flags"]:
                print(f"      ! {fl}")
    else:
        from .report import build
        w = build(args.dir, args.meta)
        for k, v in w.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
