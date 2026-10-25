"""Build a tiny local Transformer checkpoint; no download or external service."""

import asyncio
import json
import torch
import pytest
from rynn_rl.train import parser, run
from rynn_rl.hf_policy import HFPolicy, token_statistics


def make_model(path):
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    vocab = {
        name: i
        for i, name in enumerate(
            [
                "[UNK]",
                "[PAD]",
                "[EOS]",
                "D",
                "G",
                "A",
                "B",
                "<route>",
                "</route>",
                "question",
                "system",
                "user",
                "assistant",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    torch.manual_seed(11)
    config = transformers.GPT2Config(
        vocab_size=len(vocab),
        n_positions=2048,
        n_embd=16,
        n_layer=1,
        n_head=2,
        eos_token_id=vocab["[EOS]"],
        pad_token_id=vocab["[PAD]"],
    )
    model = transformers.GPT2LMHeadModel(config)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def test_local_hf_training_entrypoint_and_checkpoint(tmp_path):
    checkpoint = tmp_path / "model"
    make_model(checkpoint)
    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps({"id": "test", "question": "question", "reference": "A", "task": "choice", "native": True}) + "\n"
    )
    out = tmp_path / "run"
    args = parser().parse_args(
        [
            "--model",
            str(checkpoint),
            "--data",
            str(data),
            "--output",
            str(out),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--text-only",
            "--batch-size",
            "1",
            "--max-new-tokens",
            "4",
            "--max-context",
            "2048",
            "--max-steps",
            "1",
        ]
    )
    asyncio.run(run(args))
    metrics = json.loads((out / "metrics.jsonl").read_text())
    assert metrics["step"] == 1 and (out / "final/config.json").is_file()
    saved = HFPolicy(
        str(out / "final"), device="cpu", dtype="float32", max_new_tokens=4, max_context=2048, multimodal=False
    )
    assert saved.reference is not None
    traces = json.loads((out / "rollout-000001.json").read_text())
    assert len(traces) == 8 and all(t["branch"] == 0 for t in traces)


def test_behavior_replay_and_frozen_route_prefix(tmp_path):
    make_model(tmp_path / "model")
    policy = HFPolicy(
        str(tmp_path / "model"), device="cpu", dtype="float32", max_new_tokens=4, max_context=2048, multimodal=False
    )
    messages = [{"role": "user", "content": "question"}]
    turn = asyncio.run(policy.sample(messages, route="G"))
    assert turn.inputs["input_ids"][0, -2].item() == policy.route_ids[1]
    assert turn.old_log_probs.shape == turn.tokens.shape
    current, _ = token_statistics(policy.model, turn)
    assert torch.allclose(current.detach(), turn.old_log_probs, atol=1e-6)
    logits = policy.route_logits(messages)
    assert logits.shape == (2,)
