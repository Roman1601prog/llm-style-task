"""Archive a checkpoint or completed project without environments/full checkpoints."""
import argparse
import zipfile
from pathlib import Path
from common import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT.parent / "submission.zip"))
    args = parser.parse_args()
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or path.resolve() == destination:
                continue
            relative = path.relative_to(ROOT)
            if any(part in {"__pycache__", ".git", ".ipynb_checkpoints"} or part.startswith(".venv") or part.startswith("checkpoints_") for part in relative.parts):
                continue
            if path.suffix in {".zip", ".pyc", ".log"}:
                continue
            archive.write(path, Path(ROOT.name) / relative)
            count += 1
    print(f"{destination}: {count} файлов")


if __name__ == "__main__":
    main()

