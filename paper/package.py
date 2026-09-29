"""Render the manuscript and tables through Pages, assemble one review PDF, copy figures, zip the supplement."""
import os, shutil, subprocess, zipfile, time
import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(HERE, "submission"); FIG = os.path.join(HERE, "figures"); ROOT = os.path.abspath(os.path.join(HERE, ".."))
RENDER = os.path.join(HERE, "render"); os.makedirs(RENDER, exist_ok=True)


def pages_pdf(docx, pdf):
    script = [
        'with timeout of 180 seconds',
        f'set inF to POSIX file "{docx}"', f'set outF to POSIX file "{pdf}"',
        'tell application "Pages"', 'set d to open inF', 'delay 3', 'export d to outF as PDF', 'close d saving no', 'end tell',
        'end timeout']
    args = ["osascript"]
    for s in script:
        args += ["-e", s]
    subprocess.run(args, check=True, capture_output=True, text=True)
    return pdf


def main():
    subprocess.run(["open", "-g", "-a", "Pages"]); time.sleep(4)
    reading = pages_pdf(os.path.join(SUB, "Manuscript with tables and figures.docx"), os.path.join(RENDER, "reading.pdf"))
    out = os.path.join(SUB, "Review copy (tables and figures in place).pdf")
    for old in ("Review copy (manuscript, figures and tables).pdf",):
        if os.path.exists(os.path.join(SUB, old)):
            os.remove(os.path.join(SUB, old))
    doc = fitz.open(reading); doc.save(out); print("review pdf pages:", doc.page_count)
    for i in range(1, 7):
        for ext in ("png", "pdf"):
            shutil.copy(os.path.join(FIG, f"Figure {i}.{ext}"), os.path.join(SUB, f"Figure {i}.{ext}"))
    keep = ["cmlm", "life", "tests", "sandbox", "data", "reproduce.py", "results.json", "LICENSE", "README.md"]
    with zipfile.ZipFile(os.path.join(SUB, "Supplemental Code S1.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for top in keep:
            p0 = os.path.join(ROOT, top)
            paths = [p0] if os.path.isfile(p0) else [os.path.join(r, f) for r, _, fs in os.walk(p0) for f in fs]
            for p in paths:
                if "__pycache__" in p or p.endswith(".pyc") or p.endswith(".DS_Store"):
                    continue
                z.write(p, os.path.join("Supplemental Code S1", os.path.relpath(p, ROOT)))
    with zipfile.ZipFile(os.path.join(SUB, "Supplemental Data S1.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(ROOT, "data", "real_session_sizes.csv"), "Supplemental Data S1/real_session_sizes.csv")
        z.write(os.path.join(ROOT, "results.json"), "Supplemental Data S1/results.json")
        z.write(os.path.join(ROOT, "sandbox", "live_by_hand_seed1.json"), "Supplemental Data S1/live_by_hand_seed1.json")
    print(sorted(os.listdir(SUB)))


if __name__ == "__main__":
    main()
