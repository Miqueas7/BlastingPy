"""Extrae el texto de PDF de referencia a un .txt junto a cada PDF.

Uso: python tools/extraer_texto.py <pdf>... [--ocr]

Las páginas se separan con «=== p. N ===» (N = página física del PDF, desde 1).
Con --ocr, las páginas con menos de 200 caracteres de texto se renderizan a 300 ppp
y pasan por Tesseract (inglés).
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf

TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA = str(Path.home() / "tessdata")
MIN_CHARS = 200


def ocr_page(page: pymupdf.Page) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "p.png"
        page.get_pixmap(dpi=300).save(img)
        out = subprocess.run(
            [TESSERACT, str(img), "stdout", "--tessdata-dir", TESSDATA, "-l", "eng", "--psm", "6"],
            capture_output=True, text=True, encoding="utf-8", check=True,
        )
        return out.stdout


def extract(pdf: Path, ocr: bool) -> tuple[Path, int]:
    parts, n_ocr = [], 0
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text()
            if ocr and len(text.strip()) < MIN_CHARS:
                text, n_ocr = ocr_page(page), n_ocr + 1
            parts.append(f"=== p. {i} ===\n{text}")
    out = pdf.with_name(pdf.name + ".txt")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out, n_ocr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--ocr", action="store_true", help="OCR para páginas sin texto")
    args = ap.parse_args()
    for pdf in args.pdfs:
        out, n_ocr = extract(pdf, args.ocr)
        print(f"{pdf.name}: {out.name} ({n_ocr} páginas con OCR)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
