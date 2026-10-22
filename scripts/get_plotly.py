"""Descarga plotly.js-dist-min 2.35.2 vía npm y lo deja en vendor/plotly.min.js.

Escapa los bytes de control y los U+FFFD literales que trae el bundle: dentro de
literales de cadena son equivalentes y así el fichero se puede publicar como
Artifact (la publicación rechaza esos caracteres).
"""
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

VERSION = "2.35.2"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "vendor" / "plotly.min.js"


def sanitize(src: str) -> str:
    for c in [chr(i) for i in range(32) if i not in (9, 10, 13)]:
        src = src.replace(c, "\\x%02x" % ord(c))
    return src.replace("�", "\\ufffd")


def main():
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["npm", "pack", f"plotly.js-dist-min@{VERSION}", "--silent"], cwd=td, check=True)
        tgz = next(Path(td).glob("plotly.js-dist-min-*.tgz"))
        with tarfile.open(tgz) as tf:
            src = tf.extractfile("package/plotly.min.js").read().decode("utf-8")
    DEST.parent.mkdir(exist_ok=True)
    DEST.write_text(sanitize(src), encoding="utf-8")
    print(f"OK {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
