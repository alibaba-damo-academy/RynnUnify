import base64
import hashlib
import io

import av
from fastapi.testclient import TestClient
from PIL import Image

from rynn_rl.serve_generator import create_app, DiffusersBackend, Request


class Backend:
    modes = {"i2v"}

    def generate(self, request, images):
        return [images[0].resize((16, 16)) for _ in range(5)]


def payload():
    out = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(out, format="PNG")
    return {"prompt": "Continue motion", "mode": "i2v", "duration_seconds": 1,
            "images": [base64.b64encode(out.getvalue()).decode()], "fps": 4}


def test_generator_http_auth_video_and_modes():
    with TestClient(create_app(Backend(), "test-secret")) as client:
        assert client.get("/health").json()["modes"] == ["i2v"]
        assert client.post("/generate_video", json=payload()).status_code == 401
        headers = {"Authorization": "Bearer test-secret"}
        result = client.post("/generate_video", json=payload(), headers=headers)
        assert result.status_code == 200
        data = result.json()
        raw = base64.b64decode(data["video_base64"])
        assert hashlib.sha256(raw).hexdigest() == data["video_sha256"]
        with av.open(io.BytesIO(raw)) as video:
            assert len(list(video.decode(video=0))) == 5
        bad = payload()
        bad["mode"] = "fl2v"
        assert client.post("/generate_video", json=bad, headers=headers).status_code == 422
        bad = payload()
        bad["images"] = ["invalid"]
        assert client.post("/generate_video", json=bad, headers=headers).status_code == 422


def test_pipeline_call_preserves_last_frame_and_reports_frames():
    from types import SimpleNamespace
    backend = object.__new__(DiffusersBackend)
    backend.backend, backend.device = "diffusers", "cpu"
    calls = []

    def pipe(**kw):
        calls.append(kw)
        return SimpleNamespace(frames=[[Image.new("RGB", (16, 16))]])

    backend.pipe = pipe
    images = [Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))]
    request = Request(**{**payload(), "mode": "fl2v", "images": payload()["images"]*2})
    assert len(backend.generate(request, images)) == 1
    assert calls[0]["image"] is images[0] and calls[0]["last_image"] is images[1]
    assert calls[0]["num_frames"] % 4 == 1


def test_judge_auth_model_and_finish_reason():
    import torch
    from types import SimpleNamespace
    from rynn_rl.serve_judge import create_app as judge_app

    class Tokenizer:
        pad_token_id = 0
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}
        def decode(self, tokens, **kwargs):
            return "1"

    class Model:
        generation_config = SimpleNamespace(eos_token_id=3)
        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 4, 3]])

    with TestClient(judge_app(Model(), Tokenizer(), "judge", "cpu", api_key="key")) as client:
        request = {"model": "judge", "messages": [{"role": "user", "content": "Score"}]}
        assert client.post("/v1/chat/completions", json=request).status_code == 401
        result = client.post("/v1/chat/completions", json=request,
                             headers={"Authorization": "Bearer key"}).json()
        assert result["choices"][0]["finish_reason"] == "stop"
