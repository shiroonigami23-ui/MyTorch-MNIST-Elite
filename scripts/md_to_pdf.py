import argparse
from pathlib import Path

import markdown
from xhtml2pdf import pisa


def link_callback(uri: str, rel: str | None) -> str:
    if uri.startswith("http://") or uri.startswith("https://"):
        return uri
    base = Path(rel).parent if rel else Path.cwd()
    path = (base / uri).resolve()
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Markdown report to PDF")
    parser.add_argument("--input", default="docs/BENCHMARK_REPORT.md")
    parser.add_argument("--output", default="docs/BENCHMARK_REPORT.pdf")
    args = parser.parse_args()

    md_path = Path(args.input).resolve()
    pdf_path = Path(args.output).resolve()

    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    html = f"""
    <html>
    <head>
      <meta charset='utf-8'>
      <style>
        @page {{ size: A4; margin: 24mm; }}
        body {{ font-family: Helvetica, Arial, sans-serif; color: #1f2937; line-height: 1.4; }}
        h1, h2, h3 {{ color: #0f172a; }}
        h1 {{ border-bottom: 2px solid #d1d5db; padding-bottom: 6px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0 16px; }}
        th, td {{ border: 1px solid #d1d5db; padding: 7px; font-size: 11px; }}
        th {{ background: #f1f5f9; text-align: left; }}
        img {{ max-width: 100%; height: auto; margin: 12px 0; }}
        code {{ background: #f8fafc; padding: 2px 4px; border: 1px solid #e2e8f0; }}
      </style>
    </head>
    <body>{html_body}</body>
    </html>
    """

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with pdf_path.open("wb") as f:
        result = pisa.CreatePDF(src=html, dest=f, link_callback=lambda u, r: link_callback(u, str(md_path)))

    if result.err:
        raise SystemExit("PDF generation failed")

    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
