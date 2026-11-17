"""Extract the text of reference PDFs to a .txt file next to each PDF.

Usage: python tools/extraer_texto.py <pdf>... [--ocr]

Pages are separated by "=== p. N ===" (N = physical PDF page, from 1).
With --ocr, pages with fewer than 200 characters of text are rendered at 300 dpi and
run through Tesseract (English); the OCR text is appended after the native text.
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
                # se conserva el texto nativo (puede traer un número suelto) y el OCR va después
                text += f"--- OCR (Tesseract, eng) ---\n{ocr_page(page)}"
                n_ocr += 1
            parts.append(f"=== p. {i} ===\n{text}")
    out = pdf.with_name(pdf.name + ".txt")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out, n_ocr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdfs", nargs="+", type=Path, help="PDF files to extract")
    ap.add_argument("--ocr", action="store_true", help="also OCR pages with almost no text")
    args = ap.parse_args()
    for pdf in args.pdfs:
        out, n_ocr = extract(pdf, args.ocr)
        print(f"{pdf.name}: wrote {out.name} ({n_ocr} pages with OCR)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
