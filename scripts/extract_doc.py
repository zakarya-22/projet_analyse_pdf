"""
Extraction du document PDF en fichier Markdown structure.
Usage : uv run python scripts/extract_doc.py
"""
import time
from pathlib import Path
import pdfplumber

ROOT       = Path(__file__).parent.parent
PDF_PATH   = ROOT / "data" / "bnp-urd-2025-fr-mel3.pdf"
MD_PATH    = ROOT / "data" / "bnp_risque_credit.md"
PAGE_START = 413
PAGE_END   = 514


def extract(pdf_path: Path, md_path: Path, start: int, end: int) -> None:
    content = ["# Rapport de Risque de Credit - BNP Paribas\n"]

    with pdfplumber.open(pdf_path) as pdf:
        for idx in range(start - 1, min(end, len(pdf.pages))):
            page    = pdf.pages[idx]
            w, h    = page.width, page.height
            left    = page.within_bbox((0, 0, w / 2, h)).extract_text() or ""
            right   = page.within_bbox((w / 2, 0, w, h)).extract_text() or ""
            text    = (left + "\n\n" + right).strip()

            content.append(f"\n\n## Page {idx + 1}\n\n{text}")

            for table in page.extract_tables():
                if not table:
                    continue
                header = "| " + " | ".join(str(c or "").replace("\n", " ") for c in table[0]) + " |"
                sep    = "| " + " | ".join("---" for _ in table[0]) + " |"
                rows   = [
                    "| " + " | ".join(str(c or "").replace("\n", " ") for c in row) + " |"
                    for row in table[1:]
                ]
                block = f"\n\n### Tableau (Page {idx + 1})\n\n{header}\n{sep}\n" + "\n".join(rows)
                content.append(block)

    md_path.write_text("\n".join(content), encoding="utf-8")
    print(f"Extraction terminee : {md_path}")


if __name__ == "__main__":
    t0 = time.time()
    extract(PDF_PATH, MD_PATH, PAGE_START, PAGE_END)
    print(f"Duree : {time.time() - t0:.1f}s")
