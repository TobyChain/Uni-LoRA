#!/usr/bin/env python
"""
FastChat-compatible API Server for Uni-LoRA Model

This script starts an OpenAI-compatible API server serving the Uni-LoRA
fine-tuned Qwen1.5-MoE model.

Usage:
    python serve_unilora_api.py --stage2_path <path> --port 8000

Then use with OpenAI client:
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "unilora-qwen-moe", "messages": [{"role": "user", "content": "Hello"}]}'
"""

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add modeling path
sys.path.insert(0, str(Path(__file__).parent.parent / "modeling"))
from modeling_unilora_moe import apply_unilora_to_qwen_moe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model and tokenizer
model = None
tokenizer = None


def load_model_with_unilora(
    base_model_path: str,
    unilora_params_path: str,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load base model and apply trained Uni-LoRA parameters."""
    global model, tokenizer

    logger.info(f"Loading base model from {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Applying Uni-LoRA architecture...")
    model = apply_unilora_to_qwen_moe(
        model,
        rank=64,
        alpha=16.0,
        use_rank1=True,
    )

    logger.info(f"Loading Uni-LoRA parameters from {unilora_params_path}")
    unilora_params = torch.load(unilora_params_path, map_location="cpu")

    if "unilora_shared_vector" in unilora_params:
        model.unilora_shared_vector.data = unilora_params["unilora_shared_vector"].to(
            model.unilora_shared_vector.device
        )

    loaded_count = 0
    for name, param in model.named_parameters():
        if name in unilora_params:
            param.data = unilora_params[name].to(param.device)
            loaded_count += 1

    logger.info(f"Loaded {loaded_count} projection parameters")
    model.eval()

    return model, tokenizer


def generate_completion(
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = False,
) -> Dict[str, Any]:
    """Generate completion for chat messages."""
    global model, tokenizer

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - start_time

    response_text = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    )

    output_tokens = outputs.shape[1] - input_length

    # OpenAI-compatible response format
    completion = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "unilora-qwen-moe",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_length,
            "completion_tokens": output_tokens,
            "total_tokens": input_length + output_tokens,
        },
        "_generation_time": round(elapsed, 2),
    }

    return completion


# Flask/FastAPI server
try:
    from flask import Flask, request, jsonify

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def create_flask_app():
    """Create Flask-based API server."""
    app = Flask(__name__)

    @app.route("/v1/models", methods=["GET"])
    def list_models():
        return jsonify(
            {
                "object": "list",
                "data": [
                    {
                        "id": "unilora-qwen-moe",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "user",
                    }
                ],
            }
        )

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions():
        data = request.json
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 512)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)

        try:
            completion = generate_completion(messages, max_tokens, temperature, top_p)
            return jsonify(completion)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    return app


def create_fastapi_app():
    """Create FastAPI-based API server."""
    app = FastAPI(title="Uni-LoRA API Server")

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str = "unilora-qwen-moe"
        messages: List[ChatMessage]
        max_tokens: int = 512
        temperature: float = 0.7
        top_p: float = 0.9
        stream: bool = False

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "unilora-qwen-moe",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "user",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        try:
            completion = generate_completion(
                messages,
                request.max_tokens,
                request.temperature,
                request.top_p,
            )
            return JSONResponse(content=completion)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def main():
    parser = argparse.ArgumentParser(description="Uni-LoRA API Server")
    parser.add_argument(
        "--base_model",
        type=str,
        default="/root/autodl-tmp/model/Qwen/Qwen1.5-MoE-A2.7B-Chat",
        help="Path to base Qwen1.5-MoE model",
    )
    parser.add_argument(
        "--stage2_path",
        type=str,
        default="../training/output/qwen_moe_unilora_pipeline_0107/stage2",
        help="Path to stage2 output directory",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["flask", "fastapi"],
        default="fastapi" if FASTAPI_AVAILABLE else "flask",
        help="Web framework backend",
    )

    args = parser.parse_args()

    # Load model
    unilora_params_path = Path(args.stage2_path) / "unilora_params.pt"
    if not unilora_params_path.exists():
        logger.error(f"unilora_params.pt not found at {unilora_params_path}")
        sys.exit(1)

    load_model_with_unilora(args.base_model, str(unilora_params_path))

    # Start server
    logger.info(f"Starting {args.backend} server on {args.host}:{args.port}")

    if args.backend == "fastapi" and FASTAPI_AVAILABLE:
        app = create_fastapi_app()
        uvicorn.run(app, host=args.host, port=args.port)
    elif FLASK_AVAILABLE:
        app = create_flask_app()
        app.run(host=args.host, port=args.port, threaded=False)
    else:
        logger.error("No web framework available. Install flask or fastapi+uvicorn")
        sys.exit(1)


if __name__ == "__main__":
    main()
