"""Activation steering with the schema vectors of Zhang et al. (2026).

A vector file holds one direction per layer, shape (29, 3584): row 0 is the embedding output and row l is the output
of decoder block l-1 (the Hugging Face hidden_states convention). Steering at "layer l" therefore adds coef * row l to
the output of block l-1, exactly as the authors' code does (ActivationSteerer(layer_idx=layer-1)).

Like the authors' default (positions="response"), the vector is added only at the last position of each forward pass:
with a KV cache that is every newly generated token, plus the final prompt token. The prompt itself is not steered.
"""
from contextlib import contextmanager
from pathlib import Path

import torch

from diary_dialogues.paths import SCHEMAS, VECTOR_DIR


def load_vector(schema: str, layer: int, vector_dir: Path = VECTOR_DIR) -> torch.Tensor:
    """The schema's direction at one layer (a 1-D tensor of the model's hidden size)."""
    if schema not in SCHEMAS:
        raise ValueError(f"unknown schema {schema!r}; choose from {SCHEMAS}")
    vectors = torch.load(Path(vector_dir) / f"{schema}_response_avg_diff.pt", map_location="cpu")
    return vectors[layer]


@contextmanager
def steer(model, vector: torch.Tensor, layer: int, coef: float):
    """Inside this block, `model` generates with coef * vector added at `layer` (see the module docstring).

    vector: one direction (hidden,), added to every sequence in the batch, or one per sequence (batch, hidden), so
    sequences steered toward different schemas (or not at all: a zero row) can be generated in one batch.
    """
    blocks = model.model.layers
    if not 1 <= layer <= len(blocks):
        raise ValueError(f"layer must be 1..{len(blocks)}, got {layer}")
    p = next(model.parameters())
    delta = (coef * vector).to(device=p.device, dtype=p.dtype)

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, -1, :] += delta
        return (h, *output[1:]) if isinstance(output, tuple) else h

    handle = blocks[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
