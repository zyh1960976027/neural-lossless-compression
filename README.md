# Mamba-3 Multi-Token Prediction Experiments

`experiment/mtp` preserves the original MTP branch at 922f0cb and adds this
branch entry page. It contains the four-head byte prediction model, training
and evaluation tools, and v6 lossless coding with CUDA Graph acceleration.
The project mainline is `main`, which uses BPE 4096 and GPU integer quantization.

Use this branch's matching byte/MTP checkpoints and runtime. Datasets, weights,
and generated experiment files are managed outside Git. Install the dependencies
specified in this branch and use a CUDA GPU supporting BF16.

- [Environment setup](docs/environment.md)
- [Training and BPB evaluation](docs/evaluation.md)
- [MTP-4 experiment](docs/mtp4.md)
- [Lossless range codec](docs/codec.md)
- [Full enwik9 comparison](docs/enwik9-full-experiment.md)

Use `train_mtp.py --help`, `evaluate_mtp.py --help`, and `codec.py --help` for
the branch-specific commands. Run `python -m unittest discover -s tests -v`
in the configured environment; real GPU checks require their documented setup.
