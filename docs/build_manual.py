"""Build the standalone PDF with pdfLaTeX and Biber; keep auxiliaries temporary."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile


DOC = Path(__file__).resolve().parent


def build():
    for name in ("pdflatex", "biber"):
        if shutil.which(name) is None:
            raise SystemExit(f"Required executable not found: {name}")
    with tempfile.TemporaryDirectory(prefix="starkzee-docs-") as temporary:
        directory = Path(temporary)
        # Keep bibliography discovery independent of the caller's directory.
        shutil.copy2(DOC / "references.bib", directory / "references.bib")
        # Search fresh auxiliaries before the source directory, which may
        # contain stale .aux or .bbl files from an earlier in-place build.
        environment = os.environ.copy()
        environment["TEXINPUTS"] = os.pathsep.join((
            str(directory), str(DOC), environment.get("TEXINPUTS", ""),
        ))
        latex = ["pdflatex", "--disable-installer", "-interaction=nonstopmode",
                 "-halt-on-error", "-file-line-error",
                 f"-output-directory={directory}", "manual.tex"]
        commands = [
            (latex, DOC),
            (["biber", "manual"], directory),
            (latex, DOC),
            (latex, DOC),
        ]
        for command, cwd in commands:
            result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    errors="replace", env=environment)
            if result.returncode:
                print(result.stdout)
                raise SystemExit(result.returncode)
        log = (directory / "manual.log").read_text(errors="replace")
        unresolved = ("There were undefined references", "There were undefined citations",
                      "Please (re)run Biber", "Rerun to get cross-references right")
        if any(message in log for message in unresolved):
            raise SystemExit("Unresolved references; destination PDF was not replaced.")
        for line in log.splitlines():
            if "Warning:" in line or "Overfull" in line:
                print(line)
        shutil.copyfile(directory / "manual.pdf", DOC / "manual.pdf")
        print(f"Built {DOC / 'manual.pdf'}")


if __name__ == "__main__":
    build()
