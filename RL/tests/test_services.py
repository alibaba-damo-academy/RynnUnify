import asyncio
import base64
import hashlib
import json
import av
import httpx
import numpy as np
import pytest
from PIL import Image
from rynn_rl.data import Question
from rynn_rl.services import Generator, Judge, last_frame
from rynn_rl.rewards import FeedbackError


def video(path):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for value in (0, 200):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), value, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path.read_bytes()


def client_mock(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    )


def test_full_video_request_roundtrip_and_observed_last_frame(tmp_path, monkeypatch):
    observed = tmp_path / "observed.mp4"
    raw = video(observed)
    assert np.asarray(last_frame(str(observed))).mean() > 190
    received = []

    def handler(request):
        payload = json.loads(request.content)
        received.append(payload)
        return httpx.Response(
            200,
            json={
                "ok": True,
                "video_base64": base64.b64encode(raw).decode(),
                "video_sha256": hashlib.sha256(raw).hexdigest(),
            },
        )

    client_mock(monkeypatch, handler)
    generator = Generator("http://test/generate_video", tmp_path / "outputs")
    q = Question("q", "question", "10 cm", "phys", videos=[str(observed)])
    result = asyncio.run(generator(q, "Continue", 4))
    assert hashlib.sha256(open(result, "rb").read()).digest() == hashlib.sha256(raw).digest()
    import io

    image = Image.open(io.BytesIO(base64.b64decode(received[0]["images"][0])))
    assert np.asarray(image).mean() > 190
    assert received[0]["mode"] == "i2v" and received[0]["return_video"] is True
    assert "reference" not in received[0]


def test_bad_checksum_is_technical_failure(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(image)
    calls = []

    def handler(request):
        calls.append(request.content)
        return httpx.Response(
            200, json={"ok": True, "video_base64": base64.b64encode(b"bad").decode(), "video_sha256": "wrong"}
        )

    client_mock(monkeypatch, handler)

    async def no_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    with pytest.raises(FeedbackError):
        asyncio.run(
            Generator("http://test/generate_video", tmp_path / "out")(
                Question("q", "q", "A", "choice", images=[str(image)]), "continue", 4
            )
        )
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]


def test_judge_rejects_truncated_verdict(monkeypatch):
    client_mock(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": "###Judge: True"}, "finish_reason": "length"}]}
        ),
    )
    with pytest.raises(FeedbackError):
        asyncio.run(Judge("http://test/v1", "model")([{"role": "user", "content": "test"}]))
