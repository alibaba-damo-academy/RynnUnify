"""Single-device Hugging Face backend with exact generated-token replay.

Uses the checkpoint's native multimodal chat template and processor. No remote model code is enabled by default.
"""

from dataclasses import dataclass
import copy
import inspect
import torch


@dataclass
class Turn:
    inputs: dict
    tokens: torch.Tensor
    text: str
    complete: bool
    old_log_probs: torch.Tensor


def append_tokens(inputs, tokens):
    """Extend text-aligned fields; visual grids/pixels retain their native shape."""
    old_length = inputs["input_ids"].shape[1]
    inputs = dict(inputs)
    if "position_ids" in inputs:
        raise ValueError("Processor-supplied position_ids require a model-specific adapter")
    inputs["input_ids"] = torch.cat((inputs["input_ids"], tokens), dim=1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    for key in ("token_type_ids", "mm_token_type_ids"):
        if key in inputs:
            if inputs[key].shape != (1, old_length):
                raise ValueError("Unexpected token-type layout")
            inputs[key] = torch.cat((inputs[key], torch.zeros_like(tokens)), dim=1)
    return inputs


def token_statistics(model, turn):
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in turn.inputs.items()}
    tokens = turn.tokens.to(device)
    inputs = append_tokens(inputs, tokens[None])
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        inputs["logits_to_keep"] = tokens.numel() + 1
    outputs = model(**inputs, use_cache=False)
    # Each generated token is predicted by the preceding input position.
    logits = outputs.logits[:, -(tokens.numel() + 1) : -1, :].float()
    logdist = logits.log_softmax(-1)
    selected = logdist.gather(-1, tokens[None, :, None]).squeeze(0).squeeze(-1)
    entropy = -(logdist.exp() * logdist).sum(-1).squeeze(0)
    return selected, entropy


class HFPolicy:
    def __init__(
        self,
        model_path,
        device="cuda",
        dtype="bfloat16",
        max_new_tokens=2048,
        max_context=32768,
        multimodal=True,
        load_reference=True,
    ):
        from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM

        self.multimodal = multimodal
        self.processor = (AutoProcessor if multimodal else AutoTokenizer).from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer if multimodal else self.processor
        cls = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
        self.model = cls.from_pretrained(model_path, torch_dtype=getattr(torch, dtype)).to(device)
        self.model.eval()  # Disable dropout for reproducible behavior likelihoods; gradients still enabled.
        self.reference = copy.deepcopy(self.model).eval().requires_grad_(False) if load_reference else None
        self.device, self.max_new_tokens, self.max_context = device, max_new_tokens, max_context
        self.route_ids = [self.tokenizer.encode(c, add_special_tokens=False) for c in ("D", "G")]
        if any(len(x) != 1 for x in self.route_ids) or self.route_ids[0] == self.route_ids[1]:
            raise ValueError("D/G must be distinct single tokens for two-logit routing")
        self.route_ids = [x[0] for x in self.route_ids]
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.eos_ids.discard(None)
        if not self.eos_ids:
            raise ValueError("Checkpoint must define EOS token(s)")

    def encode(self, messages, route_prefix=False):
        messages = copy.deepcopy(messages)
        if not self.multimodal:
            for message in messages:
                if isinstance(message["content"], list):
                    if any(c["type"] != "text" for c in message["content"]):
                        raise ValueError("Text backend cannot consume images/videos")
                    message["content"] = "\n".join(c["text"] for c in message["content"])
        if route_prefix:
            messages.append({"role": "assistant", "content": "<route>"})
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=not route_prefix,
            continue_final_message=route_prefix,
        )
        if encoded["input_ids"].shape[1] + self.max_new_tokens > self.max_context:
            raise ValueError("Context budget exceeded; inputs are never silently truncated")
        return dict(encoded)

    def forced(self, messages, route):
        inputs = self.encode(messages, route_prefix=route is not None)
        if route is not None:
            index = 0 if route == "D" else 1
            suffix = [self.route_ids[index]] + self.tokenizer.encode("</route>\n", add_special_tokens=False)
            inputs = append_tokens(inputs, torch.tensor([suffix]))
            if inputs["input_ids"].shape[1] + self.max_new_tokens > self.max_context:
                raise ValueError("Forced route exceeds context budget")
        return inputs

    async def sample(self, messages, route=None, greedy=False):
        inputs = self.forced(messages, route)
        model_inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        # Temperature=1, top_p=1, top_k=0 keeps behavior probabilities identical to
        # the logits used by GSPO. Do not introduce filtering without correcting it.
        from transformers import GenerationConfig

        generation = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=not greedy,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            eos_token_id=sorted(self.eos_ids),
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else min(self.eos_ids),
        )
        output = self.model.generate(**model_inputs, generation_config=generation)
        tokens = output[0, model_inputs["input_ids"].shape[1] :].detach().cpu()
        if not tokens.numel():
            raise RuntimeError("Policy returned no action tokens")
        turn = Turn(
            inputs,
            tokens,
            self.tokenizer.decode(tokens, skip_special_tokens=True),
            int(tokens[-1]) in self.eos_ids,
            torch.empty(0),
        )
        with torch.no_grad():
            turn.old_log_probs = token_statistics(self.model, turn)[0].detach().cpu()
        return turn

    def route_logits(self, messages):
        inputs = self.encode(messages, route_prefix=True)
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        if "logits_to_keep" in inspect.signature(self.model.forward).parameters:
            inputs["logits_to_keep"] = 1
        return self.model(**inputs, use_cache=False).logits[0, -1, self.route_ids].float()

    def save(self, path):
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
