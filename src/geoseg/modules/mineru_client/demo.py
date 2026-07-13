"""MinerU demo: extract gxae PDF and inspect output.

Input: tests/fixtures/ph01/gxae11701.pdf
Output: runs/mineru/ extracted images + markdown
Verification: extraction completes successfully, images exist.
"""

import json
from pathlib import Path

from .client import upload_and_extract


def main():
    base = Path(__file__).resolve().parents[3]
    pdf_path = base / "tests" / "fixtures" / "ph01" / "gxae11701.pdf"
    out_dir = base / "runs" / "mineru"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting: {pdf_path}")
    result = upload_and_extract(pdf_path, is_ocr=False, enable_formula=True, enable_table=True, language="en")

    print(f"\nExtraction complete!")
    print(f"  state={result['state']}")
    print(f"  full_zip_url={result.get('full_zip_url', 'N/A')}")
    print(f"  err_msg={result.get('err_msg', '')}")

    # Save result metadata
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Download zip if available
    zip_url = result.get("full_zip_url")
    if zip_url:
        import requests
        print(f"\nDownloading zip ...")
        zip_resp = requests.get(zip_url, timeout=60)
        zip_resp.raise_for_status()
        zip_path = out_dir / "extracted.zip"
        zip_path.write_bytes(zip_resp.content)
        print(f"  Saved: {zip_path} ({len(zip_resp.content)} bytes)")

        # Unzip
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir / "extracted")
        print(f"  Extracted to: {out_dir / 'extracted'}")

    # List extracted files
    extracted_dir = out_dir / "extracted"
    if extracted_dir.exists():
        files = list(extracted_dir.rglob("*"))
        images = [f for f in files if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        md_files = [f for f in files if f.suffix.lower() == ".md"]
        print(f"\nExtracted: {len(images)} images, {len(md_files)} markdown files")
        for img in images[:10]:
            print(f"  {img.relative_to(extracted_dir)}")

    # Verification
    checks = []
    checks.append(("state_is_done", result.get("state") == "done"))
    checks.append(("zip_url_exists", bool(zip_url)))
    if extracted_dir.exists():
        checks.append(("has_images", len(images) > 0))
        checks.append(("has_markdown", len(md_files) > 0))

    print("\n=== Verification ===")
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nMinerU PASS: All checks passed")
    else:
        print("\nMinerU FAIL: Some checks failed")


if __name__ == "__main__":
    main()
