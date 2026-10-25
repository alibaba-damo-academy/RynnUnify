#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Plan-conditioned video generation through the ``gen`` branch of OmniMoTModel.

Caption JSON carries the action, optional future-scene description, and media
metadata. Scene-memory images precede the main video; an optional previous-chunk
prefix conditions continuous generation. Both layouts must match training.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from cosmos_framework.data.vfm.local_datasets.helper import get_aspect_ratio
from cosmos_framework.data.vfm.sequence_packing import SequencePlan
from cosmos_framework.data.vfm.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.inference.structured_caption import caption_json_to_prompt
from cosmos_framework.tools.visualize.video import save_img_or_video
from interleave_vp.utils import build_caption_json, extract_last_frame, is_main_rank

# Wan2.2 VAE temporal compression: k clean latents cover pixels 0..4(k-1), latent 0 is
# pixel 0 alone, and a clip must therefore be 4n+1 frames long.
TEMPORAL_COMPRESSION = 4


def _pil_to_conditioning_tensor(img: Image.Image, target_h: int, target_w: int) -> torch.Tensor:
    """PIL → conditioning tensor [3,1,H,W] in [-1,1].

    Uses tensor-level resize and center crop to match training preprocessing.
    """
    import numpy as np

    img_tensor = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).float().unsqueeze(0)  # [1,3,H,W]
    orig_h, orig_w = img_tensor.shape[2], img_tensor.shape[3]
    scaling_ratio = max(target_w / orig_w, target_h / orig_h)
    resize_h = int(math.ceil(scaling_ratio * orig_h))
    resize_w = int(math.ceil(scaling_ratio * orig_w))
    img_tensor = TF.resize(img_tensor, [resize_h, resize_w])
    img_tensor = TF.center_crop(img_tensor, [target_h, target_w])
    img_tensor = img_tensor.squeeze(0).round().clamp(0, 255).to(torch.uint8).float()  # [3,H,W]
    img_tensor = img_tensor / 127.5 - 1.0
    return img_tensor.unsqueeze(1)  # [3,1,H,W]


def _resolve_caption_num_frames(num_frames: int, caption_num_frames: int | None) -> int:
    resolved = num_frames if caption_num_frames is None else int(caption_num_frames)
    if resolved <= 0:
        raise ValueError(f"caption_num_frames must be positive, got {resolved}")
    return resolved


class PlanGeneratorEngine:
    """Generator inference using the gen branch of OmniMoTModel.

    Builds the conditioned video batch and calls
    ``model.generate_samples_from_batch``.
    """

    def __init__(
        self,
        model,
        resolution: str = "480",
        num_frames: int = 81,
        fps: int = 24,
        num_steps: int = 50,
        guidance: float = 3.0,
        shift: float = 3.0,
        anchor_resolution: str = "",
    ):
        self.model = model
        self.resolution = resolution
        self.num_frames = num_frames
        self.fps = fps
        self.num_steps = num_steps
        self.guidance = guidance
        self.shift = shift
        self.output_sizes = VIDEO_RES_SIZE_INFO[resolution]
        # Resolution tier for the history anchor; must match the training
        # side's history_anchor_resolution. Empty = same tier as the main video.
        if anchor_resolution and anchor_resolution not in VIDEO_RES_SIZE_INFO:
            raise ValueError(f"anchor_resolution={anchor_resolution!r} not in {list(VIDEO_RES_SIZE_INFO)}")
        self.anchor_resolution = anchor_resolution
        self.device = model.tensor_kwargs.get("device", "cuda")

    @torch.no_grad()
    def generate_visual_plan(
        self,
        text_plan: str,
        current_frame: Image.Image,
        output_dir: Path,
        aspect_ratio: str | None = None,
        seed: int | None = None,
        write_output: bool | None = None,
        context_images: list[Image.Image] | None = None,
        keep_frames: bool = False,
        prefix_frames: list[Image.Image] | None = None,
        target_frame: Image.Image | None = None,
        num_frames: int | None = None,
        caption_num_frames: int | None = None,
        temporal_caption: str | None = None,
    ) -> tuple[Image.Image, Path]:
        """Generate visual plan video from text plan + current frame.

        Args:
            text_plan: Predicted text plan from Reasoner.
            current_frame: Current observation frame (PIL).
            output_dir: Directory to save outputs.
            aspect_ratio: Fixed AR bucket key (e.g. "16,9"). When None,
                auto-detected from current_frame dimensions. Pass a fixed
                value during interleaved rolling to avoid AR drift when
                generated frames have target-resolution dimensions.
            seed: Random seed for diffusion sampling.
            write_output: Whether to save the generated video on this rank.
                Defaults to the previous rank-0-only behavior.
            context_images: Optional scene-memory frames (PIL) packed as
                fully-conditioned vision items before the main video, matching the
                --context-frames training format. Index 0 is the episode anchor and may
                use ``anchor_resolution``; the rest use the 256 tier.
                Only use with a model trained on this context layout.
            keep_frames: Expose ``last_video_frames_pil`` (this chunk's decoded frames) so
                the rollout can feed --context-fill top-up and the cross-chunk prefix
                without re-decoding the mp4.
            prefix_frames: Optional V2V video prefix (PIL) handed over clean as
                the leading latent frames of the main video, matching a
                ``conditioning_config`` trained with k > 1. The count must be
                exactly ``4k - 3`` for the intended k latent frames (the Wan2.2
                VAE is causal, so k latents span pixels 0..4(k-1)). Composes with
                context_images; mutually exclusive with target_frame.
            caption_num_frames: Optional action duration used in the caption.
                Defaults to the full generated timeline length. Cross-chunk V2V
                uses this to exclude the preceding-chunk prefix from the current
                action's stated duration.
            temporal_caption: Optional dense visual-evolution description added
                to the caption_json (the field the training plan sources carry;
                feed the Reasoner's predicted <world_modeling> text to match the
                training prompt distribution).

        Returns:
            (last_frame_pil, video_path).
        """
        w, h = current_frame.size
        ar_bucket = aspect_ratio if aspect_ratio is not None else get_aspect_ratio(w, h)
        target_w, target_h = self.output_sizes[ar_bucket]

        temporal_compression_factor = TEMPORAL_COMPRESSION
        # Per-call length override: the goal-conditioned single-step evaluator
        # generates each clip at its own GT length, because pinning the goal far
        # past where the action actually ends leaves the model with nothing to do
        # in between. Snapped to 4n+1, which the VAE requires.
        num_frames = int(num_frames) if num_frames else self.num_frames
        num_frames = (num_frames - 1) // temporal_compression_factor * temporal_compression_factor + 1
        caption_num_frames = _resolve_caption_num_frames(num_frames, caption_num_frames)
        cond_duration = caption_num_frames / self.fps

        caption_json = build_caption_json(
            plan=text_plan,
            target_w=target_w, target_h=target_h,
            ar_bucket=ar_bucket,
            duration_sec=cond_duration, fps=self.fps,
            temporal_caption=temporal_caption,
        )
        prompt = caption_json_to_prompt(caption_json)

        conditioning_frames = _pil_to_conditioning_tensor(current_frame, target_h, target_w)

        video_data = torch.zeros(1, 3, num_frames, target_h, target_w, dtype=torch.bfloat16)
        video_data[0, :, :1, :, :] = conditioning_frames.to(dtype=torch.bfloat16)
        video_data[0, :, 1:, :, :] = video_data[0, :, 0:1, :, :].expand(-1, num_frames - 1, -1, -1)

        cond_indexes = [0]
        if prefix_frames:
            # V2V prefix: hand over the first k latent frames clean and continue.
            # Composes with context_images (the cross-chunk recipe): the prefix lives
            # in the main video, which is the last packed item, so the context items
            # stay clean ahead of it. target_frame owns the last latent, so it stays
            # exclusive.
            if target_frame is not None:
                raise ValueError("prefix_frames and target_frame are mutually exclusive")
            n_prefix = len(prefix_frames)
            k = 1 + (n_prefix - 1) // temporal_compression_factor
            if n_prefix != temporal_compression_factor * (k - 1) + 1:
                raise ValueError(f"prefix_frames must be 4k-3 frames for k clean latents, got {n_prefix}")
            if n_prefix > num_frames:
                raise ValueError(f"prefix ({n_prefix}) longer than the video ({num_frames})")
            prefix_tensor = torch.cat(
                [_pil_to_conditioning_tensor(f, target_h, target_w) for f in prefix_frames], dim=1
            )  # [3,n_prefix,H,W]
            video_data[0, :, :n_prefix, :, :] = prefix_tensor.to(dtype=torch.bfloat16)
            if n_prefix < num_frames:
                # Hold the last prefix frame over the tail. The causal VAE's
                # conditioned latents depend only on prefix pixels, so this
                # placeholder cannot affect them.
                video_data[0, :, n_prefix:, :, :] = video_data[0, :, n_prefix - 1 : n_prefix, :, :].expand(
                    -1, num_frames - n_prefix, -1, -1
                )
            cond_indexes = list(range(k))
        elif target_frame is not None:
            # Goal-conditioned chunk (training-side target_frame_condition): the
            # LAST latent frame is handed over clean, so the model interpolates
            # between the start and the goal instead of extrapolating.
            #
            # Fill the entire tail with the goal: the causal VAE's final latent
            # depends on all preceding frames. Latent 0 stays unchanged because
            # it depends only on the initial frame; intermediate latents are noised.
            target_tensor = _pil_to_conditioning_tensor(target_frame, target_h, target_w)
            video_data[0, :, 1:, :, :] = target_tensor.to(dtype=torch.bfloat16)
            t_latent = 1 + (num_frames - 1) // temporal_compression_factor
            cond_indexes = [0, t_latent - 1]

        sequence_plan = SequencePlan(
            has_text=True, has_vision=True, condition_frame_indexes_vision=cond_indexes,
        )

        if context_images:
            # Multi-vision-item batch: the sample's video entry is a LIST of
            # batch-less [3,T,H,W] items (the model-side flatten unsqueezes
            # each); all items except the last are packed fully conditioned.
            # The anchor (index 0) may use a higher tier than the rolling
            # keyframes, which stay at 256 — mirroring the training-side
            # history_anchor_resolution.
            base_w, base_h = VIDEO_RES_SIZE_INFO["256"][ar_bucket]
            if self.anchor_resolution:
                anchor_w, anchor_h = VIDEO_RES_SIZE_INFO[self.anchor_resolution][ar_bucket]
            else:
                anchor_w, anchor_h = base_w, base_h
            ctx_items = []
            ctx_sizes = []
            for ctx_idx, img in enumerate(context_images):
                cw, ch = (anchor_w, anchor_h) if ctx_idx == 0 else (base_w, base_h)
                ctx_items.append(
                    _pil_to_conditioning_tensor(img, ch, cw).to(dtype=torch.bfloat16, device=self.device)
                )  # [3,1,ch,cw] in [-1,1]
                ctx_sizes.append(
                    torch.tensor([ch, cw, ch, cw], dtype=torch.float32, device=self.device)
                )
            video_entry = ctx_items + [video_data[0].to(self.device)]
            image_size_entry = ctx_sizes + [
                torch.tensor([target_h, target_w, target_h, target_w], dtype=torch.float32, device=self.device)
            ]
        else:
            video_entry = video_data.to(self.device)
            image_size_entry = torch.tensor(
                [[target_h, target_w, target_h, target_w]], dtype=torch.float32
            ).to(self.device)

        data_batch = {
            "dataset_name": "video_data",
            self.model.input_video_key: [video_entry],
            "image_size": [image_size_entry],
            "t5_text_embeddings": torch.randn(1, 512, 1024, dtype=torch.bfloat16, device=self.device),
            "fps": torch.full((1,), float(self.fps), device=self.device),
            "conditioning_fps": torch.full((1,), float(self.fps), device=self.device),
            "num_frames": torch.full((1,), num_frames, device=self.device),
            "is_preprocessed": True,
            "sequence_plan": [sequence_plan],
            self.model.input_caption_key: [prompt],
        }

        actual_seed = seed if seed is not None else 42
        outputs = self.model.generate_samples_from_batch(
            data_batch,
            guidance=self.guidance,
            seed=[actual_seed],
            num_steps=self.num_steps,
            shift=self.shift,
            n_sample=1,
            has_negative_prompt=False,
        )

        vision_latent = outputs["vision"][0]
        vision = self.model.decode(vision_latent)  # [1,3,T,H,W]
        video_cthw = ((1.0 + vision[0]) / 2).clamp(0, 1)  # [3,T,H,W]

        video_path = output_dir / "visual_plan.mp4"
        if write_output is None:
            write_output = is_main_rank()
        if write_output:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_img_or_video(video_cthw, str(video_path.with_suffix("")), fps=self.fps)

        last_frame = extract_last_frame(video_cthw)

        # Keep this chunk's decoded frames for the eval loop (avoids re-decoding the
        # mp4): --context-fill tops the first steps up from them, and --cross-chunk-v2v
        # slices the V2V prefix out of them. Populated on request only, since a chunk's
        # full frames cost ~100 MB.
        if keep_frames:
            frames_np = (video_cthw.permute(1, 2, 3, 0) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            self.last_video_frames_pil = [Image.fromarray(frames_np[t]) for t in range(frames_np.shape[0])]
        else:
            self.last_video_frames_pil = None

        return last_frame, video_path
