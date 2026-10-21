"""Build the pinned Constriction integer interface (requires Cargo >= 1.75)."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    if sys.platform != "linux":
        raise SystemExit("This build entry currently supports Linux")
    crate = ROOT / "native/integer_range"
    target = ROOT / "build/integer_range"
    subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path", str(crate / "Cargo.toml"),
                    "--target-dir", str(target)], check=True)
    destination = ROOT / "neural_compression/_integer_range.so"
    shutil.copy2(target / "release/libmamba_integer_range.so", destination)
    print(destination)
