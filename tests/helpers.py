import numpy as np
import torch


class ControlledPredictor:
    probability_transfer = "gpu_int"

    def __init__(self, tokenizer, batch_size=2):
        self.model = torch.nn.Identity()
        self.device = torch.device("cpu")
        self.batch_size = batch_size
        self.vocab_size = tokenizer.vocab_size
        self.tokenizer_sha256 = tokenizer.sha256
        self.checkpoint_sha256 = "0" * 64
        self.runtime = {"test": "controlled"}
        self.reset(batch_size)

    def synchronize(self):
        pass

    def reset(self, batch_size):
        self.states = np.zeros(batch_size, dtype=np.int64)
        self.steps = 0

    def step(self, tokens, active=None, targets=None):
        self.states = (self.states + tokens) % self.vocab_size
        self.steps += 1
        if self.steps > 2047:
            raise AssertionError("Crossed a block reset")
        p = np.full(
            (len(tokens), self.vocab_size), 0.1 / self.vocab_size, dtype=np.float32
        )
        p[np.arange(len(tokens)), self.states] += 0.9
        from neural_compression.entropy import quantize_cdf, select_intervals

        cdf = quantize_cdf(torch.from_numpy(p)).numpy()
        return (cdf if targets is None else select_intervals(cdf, targets)), None
