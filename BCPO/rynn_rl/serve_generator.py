"""Serve frozen video generation with validated, portable MP4 responses."""
import argparse
import base64
import hashlib
import importlib
import io
import math
import os
import secrets
import threading
from pathlib import Path

import av
import numpy as np
from PIL import Image
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


class Request(BaseModel):
    prompt: str = Field(min_length=1, max_length=32768)
    mode: str
    images: list[str] = Field(min_length=1, max_length=2)
    duration_seconds: float = Field(gt=0, le=10)
    fps: int = Field(default=24, ge=1, le=60)
    resolution: str = "480"
    num_inference_steps: int = Field(default=25, ge=1, le=100)
    guidance_scale: float = Field(default=5, ge=0, le=30)
    return_frames_per_second: int = Field(default=2, ge=1, le=24)
    seed: int = Field(default=42, ge=0, le=2**32-1)
    return_video: bool = True
    persist_video: bool = True


def decode_images(request):
    expected = {"i2v": 1, "fl2v": 2}.get(request.mode)
    if expected is None or len(request.images) != expected:
        raise ValueError("i2v requires one image; fl2v requires two")
    images = []
    for value in request.images:
        if len(value) > 24 * 1024 * 1024:
            raise ValueError("Conditioning image exceeds 24 MiB encoded")
        with Image.open(io.BytesIO(base64.b64decode(value, validate=True))) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("Conditioning image exceeds 16 megapixels")
            images.append(image.convert("RGB"))
    return images


def pack_video(frames, fps, sample_fps):
    if not frames:
        raise RuntimeError("Backend returned no frames")
    frames = [frame.convert("RGB") for frame in frames]
    width, height = frames[0].size
    if width % 2 or height % 2 or any(f.size != (width, height) for f in frames):
        raise RuntimeError("Video frames must share even dimensions")
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for image in frames:
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    raw = buffer.getvalue()
    sampled = []
    for index in range(0, len(frames), max(1, round(fps / sample_fps))):
        png = io.BytesIO()
        frames[index].save(png, format="PNG")
        sampled.append(base64.b64encode(png.getvalue()).decode())
    return dict(ok=True, video_base64=base64.b64encode(raw).decode(),
                video_sha256=hashlib.sha256(raw).hexdigest(), frames=sampled,
                fps=fps, num_frames=len(frames), duration_seconds=len(frames)/fps)


class DiffusersBackend:
    """Explicit adapters; arbitrary pipelines are not assumed compatible."""
    def __init__(self, model, backend, device, dtype, fl2v=False):
        import torch
        import diffusers
        self.backend, self.device = backend, device
        self.modes = {"i2v"}
        if backend == "cosmos3":
            cls = getattr(diffusers, "Cosmos3OmniPipeline", None)
            if cls is None:
                raise RuntimeError("Install a Diffusers version providing Cosmos3OmniPipeline; see docs/deployment.md")
            if fl2v:
                raise ValueError("The Cosmos3 adapter exposes i2v only; use a native adapter for fl2v")
            self.pipe = cls.from_pretrained(model, dtype=getattr(torch, dtype), device_map=device)
        else:
            self.pipe = diffusers.DiffusionPipeline.from_pretrained(model, torch_dtype=getattr(torch, dtype))
            # Keep behavior explicit and tied to a documented pipeline contract.
            if self.pipe.__class__.__name__ != "WanImageToVideoPipeline":
                raise ValueError("diffusers backend currently supports WanImageToVideoPipeline directories")
            self.pipe.to(device)
            if fl2v:
                self.modes.add("fl2v")

    def generate(self, request, images):
        import torch
        short = int(request.resolution)
        if short not in (256, 480, 720):
            raise ValueError("Supported resolutions: 256, 480, 720")
        ratio = images[0].width / images[0].height
        height = short if ratio >= 1 else round(short / ratio)
        width = round(short * ratio) if ratio >= 1 else short
        if max(width, height) > 2048:
            raise ValueError("Requested aspect ratio exceeds output dimension limit")
        height, width = max(32, height // 32 * 32), max(32, width // 32 * 32)
        # Temporal VAE uses 4n+1 frames; report actual duration in the response.
        count = 4 * math.ceil((request.duration_seconds * request.fps - 1) / 4) + 1
        kwargs = dict(prompt=request.prompt, image=images[0], height=height, width=width,
                      num_frames=count, num_inference_steps=request.num_inference_steps,
                      guidance_scale=request.guidance_scale,
                      generator=torch.Generator(device=self.device).manual_seed(request.seed))
        if request.mode == "fl2v":
            kwargs["last_image"] = images[1]
        if self.backend == "cosmos3":
            kwargs["fps"] = float(request.fps)
        with torch.inference_mode():
            result = self.pipe(**kwargs)
        frames = result.video if self.backend == "cosmos3" else result.frames[0]
        return [f if isinstance(f, Image.Image) else Image.fromarray(
            (np.clip(f, 0, 1)*255).round().astype(np.uint8) if np.asarray(f).dtype.kind == "f"
            else np.asarray(f).astype(np.uint8)) for f in frames]


def create_app(backend, api_key=None):
    app = FastAPI(title="RynnUnify frozen video generator")
    lock = threading.Lock()

    def authorize(value):
        if api_key and not secrets.compare_digest(value or "", "Bearer " + api_key):
            raise HTTPException(401, "Invalid bearer token")

    @app.get("/health")
    def health():
        return {"ready": True, "modes": sorted(backend.modes)}

    @app.post("/generate_video")
    def generate(request: Request, authorization: str | None = Header(default=None)):
        authorize(authorization)
        if request.mode not in backend.modes:
            raise HTTPException(422, "Backend does not support requested conditioning mode")
        try:
            images = decode_images(request)
        except (ValueError, OSError) as error:
            raise HTTPException(422, str(error)) from error
        # One GPU pipeline is not reentrant. Queue requests without concurrent model mutation.
        with lock:
            try:
                frames = backend.generate(request, images)
                return pack_video(frames, request.fps, request.return_frames_per_second)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["cosmos3", "diffusers", "native"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--native-factory", help="Installed module:factory for the checkpoint's native runtime")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--enable-fl2v", action="store_true", help="Requires a Wan FLF2V-compatible checkpoint")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18900)
    args = p.parse_args()
    if not Path(args.model).is_dir():
        p.error("--model must be a local model directory; download weights before startup")
    if args.backend == "native":
        if not args.native_factory or ":" not in args.native_factory:
            p.error("native requires --native-factory module:function supplied by the native runtime")
        module, name = args.native_factory.split(":", 1)
        backend = getattr(importlib.import_module(module), name)(
            model_path=args.model, device=args.device, dtype=args.dtype)
        if not getattr(backend, "modes", None) or not callable(getattr(backend, "generate", None)):
            p.error("Native factory must return an object with modes and generate(request, images)")
    else:
        backend = DiffusersBackend(args.model, args.backend, args.device, args.dtype, args.enable_fl2v)
    import uvicorn
    uvicorn.run(create_app(backend, os.getenv("GENERATOR_API_KEY")), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
