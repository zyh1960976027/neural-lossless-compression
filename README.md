# Mamba-3 Model Scale Experiments

`experiment/model-scale` preserves the model depth and parameter-scale
experiment snapshot at 6ba330d and adds this branch entry page. The project
mainline is `main`, which uses BPE 4096 and GPU integer quantization.

This historical branch contains the byte-model training and probability
evaluation implementation for model-scale work. Use its own model configuration,
matching byte checkpoints, and command-line options. The later BPE token codec
and GPU integer range interface are available on `main`.

- [Environment setup](docs/environment.md)
- [Training and BPB evaluation](docs/evaluation.md)

Use `train.py --help` and `evaluate.py --help` from this branch. Datasets,
checkpoints, environments, and generated experiment files are excluded from Git.
Run `python -m unittest discover -s tests -v` in the configured environment.
