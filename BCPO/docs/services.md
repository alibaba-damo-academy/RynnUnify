# Frozen service interfaces

Start the included Generator and Judge servers using [Deployment](deployment.md), or configure compatible external endpoints.

## Video generator

POST to your configured `--generator-url`, for example `http://127.0.0.1:18900/generate_video`. JSON request fields:

```json
{"prompt":"Continue the interaction. Use <frame 1> as the conditioning image.","mode":"i2v","images":["BASE64_PNG"],"duration_seconds":4.0,"fps":24,"resolution":"480","num_inference_steps":25,"guidance_scale":5.0,"return_frames_per_second":2,"persist_video":true,"return_video":true,"seed":42}
```

Use `fl2v` for two first/last conditioning frames. Different image sizes are letterboxed to the first image's dimensions. Frame references in the sampled prompt select the supplied observed images. Phys conditioning uses the actual final decoded frame of its observed video.

Required response:

```json
{"ok":true,"video_base64":"BASE64_COMPLETE_MP4","video_sha256":"SHA256_OF_DECODED_MP4"}
```

The client verifies the checksum and decodes the MP4 before returning a local path to the reasoner. A filesystem path on the server alone is not sufficient. The server also returns a `frames` array of sampled base64 PNGs.

Authentication: optional `GENERATOR_API_KEY`, sent as a bearer token. No server creation, shutdown or model download is performed. Generation is retried up to three times on technical failures using the same request. A server may not provide idempotency, so a transport timeout can lead to duplicated compute on retry; configure service-side request deduplication for production use.

## Text judge

`--judge-url` is an OpenAI-compatible base URL ending in `/v1`; requests go to `/chat/completions`. Specify `--judge-model`. Authentication comes from `JUDGE_API_KEY`. The judge must support `messages`, `temperature`, `max_tokens`, and a response containing `choices[0].message.content` and `finish_reason="stop"`.

Return the exact verdict format requested by the included task rubric. Structured/length-limited/incomplete responses are retried and then raise a feedback error. They are never coerced into zero reward.
