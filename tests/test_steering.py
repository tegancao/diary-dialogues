"""The steering hook adds coef * vector to block (layer-1)'s output, at the last position only."""
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from diary_dialogues.simulator.steering import steer


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32)
    return Qwen2ForCausalLM(cfg).eval()


def block_outputs(model, ids):
    """Output of every decoder block, recorded after any earlier-registered hooks (e.g. steering) have run."""
    outs, handles = [], []
    for block in model.model.layers:
        handles.append(block.register_forward_hook(
            lambda m, i, o: outs.append((o[0] if isinstance(o, tuple) else o).detach().clone())))
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()
    return outs


def test_adds_vector_at_last_position_of_the_right_block(tiny):
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    vec, layer, coef = torch.randn(16), 2, 1.5            # layer 2 = output of block index 1
    plain = block_outputs(tiny, ids)
    with steer(tiny, vec, layer, coef):
        steered = block_outputs(tiny, ids)
    assert torch.equal(steered[0], plain[0])               # earlier block untouched
    diff = steered[layer - 1] - plain[layer - 1]
    assert torch.allclose(diff[0, -1], coef * vec, atol=1e-5)    # last position shifted by exactly coef * vec
    assert torch.allclose(diff[0, :-1], torch.zeros(4, 16))      # prompt positions untouched
    assert not torch.allclose(steered[-1], plain[-1])      # later blocks see the change


def test_hook_is_removed_after_the_block(tiny):
    ids = torch.tensor([[1, 2, 3]])
    before = block_outputs(tiny, ids)
    with steer(tiny, torch.randn(16), 3, 2.0):
        pass
    after = block_outputs(tiny, ids)
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_rejects_bad_layer(tiny):
    with pytest.raises(ValueError):
        with steer(tiny, torch.randn(16), 0, 1.0):
            pass


def test_one_vector_per_sequence_in_a_batch(tiny):
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    vecs = torch.stack([torch.randn(16), torch.zeros(16)])    # row 0 steered, row 1 unsteered
    plain = block_outputs(tiny, ids)
    with steer(tiny, vecs, 2, 2.0):
        steered = block_outputs(tiny, ids)
    diff = steered[1] - plain[1]
    assert torch.allclose(diff[0, -1], 2.0 * vecs[0], atol=1e-5)    # row 0: shifted at its last position
    assert torch.allclose(diff[1], torch.zeros(3, 16))               # row 1: untouched
