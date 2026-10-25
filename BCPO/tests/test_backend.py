import torch
from rynn_rl.hf_policy import Turn, token_statistics, append_tokens
from tests.helpers import TinyModel


def test_replay_alignment_excludes_prompt():
    torch.manual_seed(1)
    model = TinyModel()
    turn = Turn({"input_ids": torch.tensor([[1, 2, 3]])}, torch.tensor([4, 5]), "", True, torch.empty(0))
    logp, entropy = token_statistics(model, turn)
    expected = model(torch.tensor([[1, 2, 3, 4, 5]])).logits[0, 2:4].log_softmax(-1)
    assert torch.allclose(logp, expected[torch.arange(2), torch.tensor([4, 5])])
    assert logp.shape == entropy.shape == (2,)


def test_multimodal_token_types_extended_as_text():
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 1]]),
        "pixel_values": torch.randn(2, 3),
    }
    result = append_tokens(inputs, torch.tensor([[4, 5]]))
    assert result["mm_token_type_ids"].tolist() == [[0, 1, 1, 0, 0]]
    assert result["pixel_values"] is inputs["pixel_values"]
    assert inputs["input_ids"].shape == (1, 3)
