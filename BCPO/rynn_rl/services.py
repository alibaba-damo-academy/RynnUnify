"""Frozen generator and judge clients. No infrastructure lifecycle operations."""

import asyncio
import base64
import hashlib
import io
import random
import os
from pathlib import Path
import uuid
import httpx
from .rewards import FeedbackError


class Judge:
    def __init__(self, endpoint, model, timeout=120, max_tokens=512, concurrency=8):
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.model, self.timeout, self.max_tokens = model, timeout, max_tokens
        self.semaphore = asyncio.Semaphore(concurrency)

    async def __call__(self, messages):
        headers = {}
        key = os.environ.get("JUDGE_API_KEY")
        if key:
            headers["Authorization"] = "Bearer " + key
        async with self.semaphore, httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json={"model": self.model, "messages": messages, "temperature": 0, "max_tokens": self.max_tokens},
                )
                response.raise_for_status()
                choice = response.json()["choices"][0]
                if choice["finish_reason"] != "stop" or not isinstance(choice["message"]["content"], str):
                    raise ValueError("Incomplete judge output")
                return choice["message"]["content"]
            except (httpx.HTTPError, KeyError, ValueError, TypeError, IndexError):
                raise FeedbackError("Judge request failed or returned incomplete feedback") from None


def last_frame(path):
    import av
    from PIL import Image

    last = None
    with av.open(path) as container:
        for frame in container.decode(video=0):
            last = frame.to_ndarray(format="rgb24")
    if last is None:
        raise ValueError("Observed video has no decodable frames")
    return Image.fromarray(last)


class Generator:
    def __init__(self, endpoint, output, timeout=7200, concurrency=4):
        self.endpoint, self.output, self.timeout = endpoint, Path(output), timeout
        self.output.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(concurrency)

    async def __call__(self, question, prompt, duration):
        from PIL import Image, ImageOps

        if question.videos:
            images = [last_frame(question.videos[0])]
        else:
            images = []
            for path in question.images:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
        if len(images) not in (1, 2):
            raise ValueError("G requires one/two images or one observed video")
        # Select conditioning images using the request frame references.
        import re

        refs = []
        for token in re.findall(r"<(?:frame|Picture)\s*([^<>]*)>", prompt, re.I):
            if not token.strip().isdigit() or not 1 <= int(token) <= len(images):
                raise ValueError("Invalid conditioning image reference")
            if int(token) not in refs:
                refs.append(int(token))
        refs = refs or list(range(1, len(images) + 1))
        images = [images[i - 1] for i in refs]
        prompt = re.sub(r"<(?:frame|Picture)\s*\d+\s*>", "", prompt, flags=re.I).strip()
        mode = "i2v" if len(images) == 1 else "fl2v"
        prompt += (
            "\nUse <frame 1> as the conditioning image."
            if mode == "i2v"
            else "\nUse <frame 1> as the first frame and <frame 2> as the last frame."
        )
        target, encoded = images[0].size, []
        for image in images:
            fitted = ImageOps.contain(image, target)
            canvas = Image.new("RGB", target)
            canvas.paste(fitted, ((target[0] - fitted.width) // 2, (target[1] - fitted.height) // 2))
            buf = io.BytesIO()
            canvas.save(buf, format="PNG")
            encoded.append(base64.b64encode(buf.getvalue()).decode())
        payload = dict(
            prompt=prompt,
            mode=mode,
            images=encoded,
            duration_seconds=duration,
            fps=24,
            resolution="480",
            num_inference_steps=25,
            guidance_scale=5.0,
            return_frames_per_second=2,
            persist_video=True,
            return_video=True,
            seed=random.getrandbits(32),
        )
        key = os.environ.get("GENERATOR_API_KEY")
        headers = {"Authorization": "Bearer " + key} if key else {}
        async with self.semaphore, httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(3):
                try:
                    response = await client.post(self.endpoint, json=payload, headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    raw = base64.b64decode(result["video_base64"], validate=True)
                    digest = hashlib.sha256(raw).hexdigest()
                    if result.get("ok") is not True or not raw or digest != result["video_sha256"]:
                        raise ValueError("Invalid video checksum")
                    path = self.output / (uuid.uuid4().hex + ".mp4")
                    path.write_bytes(raw)
                    try:
                        last_frame(str(path))  # Decode before admitting it as visual evidence.
                    except Exception:
                        path.unlink(missing_ok=True)
                        raise ValueError("Undecodable generated video") from None
                    return str(path.resolve())
                except (httpx.HTTPError, KeyError, ValueError, TypeError):
                    if attempt == 2:
                        raise FeedbackError("Generator failed after 3 technical attempts") from None
                    await asyncio.sleep(1 + attempt)
