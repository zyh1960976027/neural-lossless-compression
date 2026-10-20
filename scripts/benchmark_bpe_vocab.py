"""Run the BPE-only 8192/16384 vocabulary experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from mamba3_byte_compress.bpe_vocab_experiment import main
    main()
