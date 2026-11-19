"""Assemble what is sent to JRC: the checked archive and the method summary as a PDF.

Run after `build_archive.py`. The script

  1. verifies the archive file itself, not the run folder it was packed from: checksum,
     one top-level folder, exactly one raster per PSU, every raster byte-identical to the
     run folder, and the JRC conformance checker plus the strict check on the unpacked
     files,
  2. renders docs/method_summary.md to a PDF: pandoc to HTML, then a Chromium-based
     browser (Edge or Chrome) in headless mode. If one of the two is not installed the
     Markdown file is copied instead,
  3. copies archive, checksum and summary into one folder.

Nothing is sent anywhere.

Usage:
    python scripts/build_submission.py --run-dir outputs/kalman_cusum --name mangidis_kalman_cusum
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from argparse import Namespace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_archive
import run_reference

SUMMARY = REPO / "docs" / "method_summary.md"
STYLE = REPO / "scripts" / "method_summary.css"
BROWSERS = (
    "msedge",
    "chrome",
    "chromium",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
)


def archive_problems(archive: Path, rasters: Path, name: str, expected: set[int]) -> list[str]:
    """What is wrong with the archive file: checksum, layout, completeness, content."""
    problems = []
    listed = archive.with_name(f"{name}.sha256")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if not listed.exists() or listed.read_text(encoding="utf-8").split()[0] != digest:
        problems.append(f"checksum in {listed.name} does not match the archive")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        found = set()
        for member in members:
            match = re.fullmatch(rf"{re.escape(name)}/disturbance_date_(\d{{7}})\.tif", member.name)
            if not member.isfile() or not match:
                problems.append(f"unexpected entry {member.name}")
                continue
            found.add(int(match[1]))
            source = rasters / Path(member.name).name
            if not source.exists() or tar.extractfile(member).read() != source.read_bytes():
                problems.append(f"{member.name} differs from the run folder")
    if len(members) != len(found):
        problems.append(f"{len(members)} entries for {len(found)} PSUs")
    if missing := sorted(expected - found):
        problems.append(f"{len(missing)} PSU(s) without a raster: {missing[:10]}")
    if unknown := sorted(found - expected):
        problems.append(f"raster(s) for unknown PSU(s): {unknown[:10]}")
    return problems


def check_unpacked(archive: Path, name: str, data_dir: Path) -> tuple[bool, str]:
    """Unpack the archive and run both conformance checks on what comes out."""
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        os.rename(Path(tmp) / name, Path(tmp) / "rasters")
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = run_reference.run_check(
                Namespace(run_dir=Path(tmp), data_dir=data_dir, remote_masks=False)
            )
    lines = [x for x in printed.getvalue().splitlines() if "Summary" in x or "check" in x.lower()]
    return code == 0, "\n".join(lines)


def find_program(candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if found := shutil.which(candidate):
            return found
        if Path(candidate).is_file():
            return candidate
    return None


def render_pdf(markdown: Path, style: Path, pdf: Path) -> str | None:
    """Write the PDF; return why not if it could not be written."""
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Pandoc" / "pandoc.exe"
    pandoc = find_program(("pandoc", str(local)))
    browser = find_program(BROWSERS)
    if not pandoc or not browser:
        return "pandoc or a Chromium-based browser is not installed"
    body = subprocess.run(
        [pandoc, "--from", "gfm", "--to", "html5", str(markdown)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>Method summary</title><style>{style.read_text(encoding='utf-8')}</style>"
        f"</head><body>{body}</body></html>"
    )
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "summary.html"
        html.write_text(page, encoding="utf-8")
        subprocess.run(
            [browser, "--headless", "--disable-gpu", "--no-pdf-header-footer"]
            + [f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={pdf.resolve()}"]
            + [html.as_uri()],
            check=True,
            capture_output=True,
            timeout=120,
        )
    return None if pdf.exists() else "the browser wrote no file"


def pdf_pages(pdf: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page(?![s\w])", pdf.read_bytes()))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--name", required=True, help="archive name given to build_archive.py")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "submission")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    args = ap.parse_args()

    archive = args.run_dir / f"{args.name}.tar.gz"
    if not archive.exists():
        print(f"{archive} does not exist; run build_archive.py first")
        return 1
    expected = build_archive.expected_psus(args.data_dir)
    problems = archive_problems(archive, args.run_dir / "rasters", args.name, expected)
    conforming, report = check_unpacked(archive, args.name, args.data_dir)
    print(report)
    for problem in problems:
        print(f"ARCHIVE: {problem}")
    if problems or not conforming:
        print("nothing assembled")
        return 1
    print(f"archive: {len(expected)} rasters, identical to {args.run_dir / 'rasters'}")

    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.out.iterdir():
        if path.is_file():
            path.unlink()  # one folder holds one submission
    shutil.copy2(archive, args.out / archive.name)
    shutil.copy2(archive.with_name(f"{args.name}.sha256"), args.out / f"{args.name}.sha256")
    pdf = args.out / "method_summary.pdf"
    reason = render_pdf(SUMMARY, STYLE, pdf)
    if reason:
        print(f"no PDF ({reason}); the Markdown file is copied instead")
        shutil.copy2(SUMMARY, args.out / SUMMARY.name)
    else:
        print(f"summary: {pdf_pages(pdf)} page(s)")
    for path in sorted(args.out.iterdir()):
        print(f"  {path}  {path.stat().st_size / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
