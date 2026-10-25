"""Serve a frozen Hugging Face text judge through the RL chat-completions contract."""
import argparse
import os
import secrets
import threading

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class Completion(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1, max_length=32)
    temperature: float = Field(default=0, ge=0, le=0)
    max_tokens: int = Field(default=512, ge=1, le=2048)


def create_app(model, tokenizer, model_name, device, max_context=32768, api_key=None):
    import torch
    app = FastAPI(title="RynnUnify frozen text judge")
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"ready": True, "model": model_name}

    @app.post("/v1/chat/completions")
    def complete(request: Completion, authorization: str | None = Header(default=None)):
        if api_key and not secrets.compare_digest(authorization or "", "Bearer " + api_key):
            raise HTTPException(401, "Invalid bearer token")
        if request.model != model_name:
            raise HTTPException(404, "Unknown judge model")
        with lock, torch.inference_mode():
            encoded = tokenizer.apply_chat_template(
                [m.model_dump() for m in request.messages], tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            prefix = encoded["input_ids"].shape[-1]
            if prefix + request.max_tokens > max_context:
                raise HTTPException(422, "Judge input plus output budget exceeds max-context")
            from transformers import GenerationConfig
            eos = model.generation_config.eos_token_id
            config = GenerationConfig(do_sample=False, max_new_tokens=request.max_tokens,
                                      eos_token_id=eos, pad_token_id=tokenizer.pad_token_id)
            tokens = model.generate(**encoded, generation_config=config)[0, prefix:]
            eos_ids = eos if isinstance(eos, list) else [eos]
            stopped = bool(len(tokens) and int(tokens[-1]) in eos_ids)
            content = tokenizer.decode(tokens, skip_special_tokens=True)
            return {"choices": [{"message": {"role": "assistant", "content": content},
                                 "finish_reason": "stop" if stopped else "length"}]}
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--served-model-name", default="judge")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--max-context", type=int, default=32768)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    args = p.parse_args()
    import torch
    import uvicorn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    model.to(args.device).eval().requires_grad_(False)
    uvicorn.run(create_app(model, tokenizer, args.served_model_name, args.device,
                          args.max_context, os.getenv("JUDGE_API_KEY")), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
