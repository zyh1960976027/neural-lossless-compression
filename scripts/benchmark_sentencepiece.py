"""Unified entry point for the byte/BPE/Unigram experiment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from mamba3_byte_compress.token_experiment import main
    main()
