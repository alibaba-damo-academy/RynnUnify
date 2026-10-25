#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Plan Reasoner engine: the ``und`` branch of an OmniMoTModel.

Image preprocessing and prompt formatting follow the checkpoint training protocol.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from PIL import Image

from cosmos_framework.data.vfm.local_datasets.helper import get_aspect_ratio
from cosmos_framework.data.vfm.utils import VIDEO_RES_SIZE_INFO
from interleave_vp.utils import (
    TASK_PROMPT_TEMPLATE,
    TASK_PROMPT_TEMPLATE_DURATION,
    resize_and_center_crop,
)


PROCESSOR_PATH_ENV = "INTERLEAVE_VLM_PROCESSOR_PATH"


def _build_user_content(
    task: str,
    initial_state_img: Image.Image,
    history: list[tuple[str, Image.Image]],
    with_duration: bool = False,
) -> list[dict]:
    """Build the same action-result trajectory prompt used during training."""
    template = TASK_PROMPT_TEMPLATE_DURATION if with_duration else TASK_PROMPT_TEMPLATE
    history_content = [{"type": "text", "text": "None"}]
    if history:
        history_content = []
        for instruction, result_image in history:
            history_content.extend([
                {"type": "text", "text": f"{instruction}\n"},
                {"type": "image", "image": result_image},
            ])

    return [
        {
            "type": "text",
            "text": (
                f"{template}"
                f"\n\n[HIGH-LEVEL TASK]\n{task}"
            ),
        },
        {"type": "text", "text": "\n\n[INITIAL STATE]\n"},
        {"type": "image", "image": initial_state_img},
        {"type": "text", "text": "\n\n[EXECUTION TRAJECTORY]\n"},
        *history_content,
        {"type": "text", "text": "\n\n[NEXT STEP]\n"},
    ]


# Training-native conversation: system format instruction, then task, initial
# image, previous instructions (text only), current observation, and next step.
# The model answers <subplan>..</subplan><world_modeling>..</world_modeling><duration>..</duration>.
REASONER_TAGS_SYSTEM_PROMPT = (
    "You are an embodied visual planning and world-modeling expert. Given the high-level task, "
    "the provided visual observations, and the previous executed steps, predict exactly one "
    "executable next sub-plan. Keep every prediction grounded in the visible scene and do not "
    "infer actions unsupported by the visual evidence. "
    "Do not use large viewpoint shifts such as 'turn left' or 'turn right' unless strictly necessary to complete the task. "
    "Respond only with exactly one response in the following format:\n\n"
    "<subplan>one concise, atomic, executable action instruction</subplan>\n"
    "<world_modeling>integrated_multimodal_description: describe the expected visual evolution "
    "caused by that action.</world_modeling>\n"
    "<duration>the estimated time required to execute the action, in seconds (e.g., 4s)</duration>"
)


def _build_user_content_tags(
    task: str,
    initial_state_img: Image.Image,
    prev_instructions: list[str],
    current_obs_img: Image.Image,
) -> list[dict]:
    """User turn of the training-native conversation (see REASONER_TAGS_SYSTEM_PROMPT).

    History is TEXT-ONLY here (unlike the bare trajectory prompt): the SFT data
    lists previous instructions as plain lines and conditions the prediction on
    a single [CURRENT OBSERVATION] frame.
    """
    prev_block = "\n".join(prev_instructions) if prev_instructions else "None"
    return [
        {"type": "text", "text": f"[HIGH-LEVEL TASK]\n{task}\n\n[INITIAL STATE]"},
        {"type": "image", "image": initial_state_img},
        {"type": "text", "text": f"\n\n[PREVIOUS STEPS]\n{prev_block}\n\n[CURRENT OBSERVATION]"},
        {"type": "image", "image": current_obs_img},
        {"type": "text", "text": "\n\n[NEXT STEP]"},
    ]


CORE_GENERATION_ARGS = ("input_ids", "pixel_values", "image_grid_thw", "attention_mask")


def _supported_kwargs(fn, kwargs: dict) -> dict:
    """Keep only the generation kwargs the installed framework's signature accepts.

    Drop unsupported sampling options with a diagnostic. Required multimodal
    conditioning arguments must be supported; otherwise raise an error.
    """
    import inspect

    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins / C functions with no signature
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return kwargs
    accepted = set(parameters)
    missing_core = [name for name in CORE_GENERATION_ARGS if name not in accepted]
    if missing_core:
        raise RuntimeError(
            f"{getattr(fn, '__qualname__', fn)} does not accept {missing_core}; this "
            "reasoner engine expects a multimodal generate_reasoner_text entry point."
        )
    dropped = sorted(set(kwargs) - accepted)
    if dropped:
        print(f"PlanReasonerEngine: framework lacks {dropped}; using its own defaults")
    return {key: value for key, value in kwargs.items() if key in accepted}


class PlanReasonerEngine:
    """Reasoner inference using the und branch of OmniMoTModel.

    Format initial and current/history observations as multi-image chat
    messages with ``vlm_processor.apply_chat_template``, then pass the resulting
    tokens and image tensors to ``model.net.generate_reasoner_text``.
    """

    def __init__(
        self,
        model,
        resolution: str = "256",
        reasoner_input_size: tuple[int, int] | None = None,
        processor_path: str | None = None,
    ):
        self.model = model
        self.processor = model.vlm_processor
        self._hf_processor_mode = False
        if type(self.processor).__name__ == "LLMTokenizerProcessor" or not callable(
            getattr(self.processor, "apply_chat_template", None)
        ):
            # A text-only processor cannot supply image tensors for multi-image
            # prompts. Load an explicitly specified, training-compatible processor.
            from transformers import AutoProcessor

            processor_path = processor_path or os.environ.get(PROCESSOR_PATH_ENV)
            if not processor_path:
                raise ValueError(
                    "this checkpoint exposes a text-only tokenizer processor; a multi-image "
                    f"rollout needs the multimodal one. Pass --vlm-processor-path <HF dir> "
                    f"(or set {PROCESSOR_PATH_ENV})."
                )
            print(f"PlanReasonerEngine: vlm_processor is text-only; using AutoProcessor from {processor_path}")
            self.processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
            self._hf_processor_mode = True
        self.output_sizes = VIDEO_RES_SIZE_INFO[resolution]
        self.reasoner_input_size = reasoner_input_size

        eos_raw = model.llm_special_tokens.get("eos_token_id")
        self.eos_id = int(eos_raw) if eos_raw is not None else None
        pad_raw = getattr(model.vlm_tokenizer, "pad_token_id", None)
        self.pad_id = int(pad_raw) if pad_raw is not None else self.eos_id
        self.device = model.tensor_kwargs.get("device", "cuda")

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """Resize and center crop to match Reasoner training preprocessing."""
        if self.reasoner_input_size is not None:
            width, height = self.reasoner_input_size
            return resize_and_center_crop(img, width, height)
        ar_key = get_aspect_ratio(img.size[0], img.size[1])
        target_w, target_h = self.output_sizes[ar_key]
        return resize_and_center_crop(img, target_w, target_h)

    def _needs_model_parallel_broadcast(self) -> bool:
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return False
        parallel_dims = getattr(self.model, "parallel_dims", None)
        if parallel_dims is None:
            return False
        return parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled or parallel_dims.cfgp_enabled

    @torch.no_grad()
    def predict_next_step(
        self,
        task: str,
        initial_state_img: Image.Image,
        history: list[tuple[str, Image.Image]],
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        seed: int | None = None,
        with_duration: bool = False,
        prompt_format: str = "bare",
        current_obs: Image.Image | None = None,
        sync_eos_across_ranks: bool = True,
    ) -> str:
        """Predict the next step from the initial state and completed trajectory.

        ``prompt_format="tags"`` switches to the training-native three-tag conversation
        (system format instruction + text-only [PREVIOUS STEPS] + a dedicated
        [CURRENT OBSERVATION] image); ``current_obs`` is then required and the tag format
        always asks for ``<duration>`` regardless of ``with_duration``. The raw text is
        returned; the caller parses it (see :mod:`interleave_vp.text_parsing`).
        """
        initial_img = self._preprocess_image(initial_state_img)
        if prompt_format == "tags":
            if current_obs is None:
                raise ValueError("prompt_format='tags' requires current_obs")
            user_content = _build_user_content_tags(
                task,
                initial_img,
                [instruction for instruction, _ in history],
                self._preprocess_image(current_obs),
            )
            messages = [
                {"role": "system", "content": [{"type": "text", "text": REASONER_TAGS_SYSTEM_PROMPT}]},
                {"role": "user", "content": user_content},
            ]
        else:
            preprocessed_history = [
                (instruction, self._preprocess_image(result_image))
                for instruction, result_image in history
            ]
            user_content = _build_user_content(task, initial_img, preprocessed_history, with_duration=with_duration)
            messages = [{"role": "user", "content": user_content}]

        if self._hf_processor_mode:
            # HF AutoProcessor.apply_chat_template only renders text (bare ids
            # tensor); images must go through the processor call separately.
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images = [
                item["image"]
                for message in messages
                for item in message["content"]
                if isinstance(item, dict) and item.get("type") == "image"
            ]
            inputs = self.processor(text=[text], images=images, return_tensors="pt")
        else:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )

        # Some Qwen3VLProcessor implementations return unbatched [L] ids while the HF
        # AutoProcessor fallback returns batched [1, L]; normalise to [1, L].
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        request = {
            "input_ids": input_ids.to(self.device),
            "pixel_values": inputs["pixel_values"].to(self.device),
            "image_grid_thw": inputs["image_grid_thw"].to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.eos_id,
            "pad_token_id": self.pad_id,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "seed": seed,
            "return_only_new_tokens": True,
            "sync_eos_across_ranks": sync_eos_across_ranks,
        }
        out_ids = self.model.net.generate_reasoner_text(**_supported_kwargs(
            self.model.net.generate_reasoner_text, request
        ))

        decoded = self.model.detokenize_text(out_ids, skip_special_tokens=True)
        if isinstance(decoded, list):
            decoded = decoded[0]
        result = decoded.strip()

        if self._needs_model_parallel_broadcast():
            result_list = [result]
            dist.broadcast_object_list(result_list, src=0)
            result = result_list[0]

        return result
