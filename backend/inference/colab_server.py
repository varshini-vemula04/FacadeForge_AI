"""Authenticated Colab HTTP server for FacadeForge SD 1.5 + Canny ControlNet.

In a Colab notebook that already has `pipe` loaded, import this module and call
set_pipeline(pipe) before starting `app`. That reuses the notebook's in-memory
pipeline instead of loading model weights a second time. If no pipeline is passed,
the server loads both models once from the local Hugging Face cache only.
"""

from __future__ import annotations

import hmac
import io
import logging
import os
import secrets
import socket
import threading

import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
)
from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

STYLES = {
    "modern": "modern contemporary residential architecture, realistic concrete, glass windows, natural stone and wood materials, professionally designed facade",
    "luxury": "luxury contemporary architecture, refined natural stone, warm timber, premium glass, elegant exterior lighting",
    "indian": "contemporary traditional Indian residential architecture, warm sandstone, subtle jaali screens, shaded verandah, climate-aware detailing",
    "industrial": "industrial modern architecture, exposed concrete, dark steel framing, textured masonry, large factory-style windows",
    "scandinavian": "Scandinavian architecture, pale timber, soft neutral stone, simple roof details, calm natural daylight",
}
BASE_PROMPT = (
    "Photorealistic architectural exterior visualization of the exact building shown in the reference sketch, "
    "realistic materials, realistic daylight, detailed textures, professional architectural photography, high quality, sharp focus. "
    "Preserve the original building geometry, floor count, roofline, window positions, door positions, balconies and proportions."
)
NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, distorted architecture, extra floors, missing floors, extra windows, missing windows, "
    "deformed building, unrealistic geometry, blurry, low quality, watermark, text, people"
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
logging.getLogger("werkzeug").setLevel(logging.ERROR)
_pipeline = None
_pipeline_lock = threading.Lock()
_inference_lock = threading.Lock()


class ServerConfigurationError(RuntimeError):
    pass


def set_pipeline(pipeline) -> None:
    """Use an already loaded pipeline from the notebook in this Python process."""
    global _pipeline
    with _pipeline_lock:
        _pipeline = pipeline
    print("FacadeForge pipeline registered successfully with the inference server.")


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        if not torch.cuda.is_available():
            raise ServerConfigurationError("This Colab runtime has no CUDA GPU. Select a GPU runtime and rerun the model setup cells.")
        try:
            controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_MODEL,
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=True,
            )
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                BASE_MODEL,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=True,
            )
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline = pipeline.to("cuda")
            pipeline.enable_attention_slicing()
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
            _pipeline = pipeline
            return _pipeline
        except Exception as exc:
            raise ServerConfigurationError(
                "Could not load SD 1.5 and Canny ControlNet from this Colab session's local Hugging Face cache. "
                "Run the notebook's existing model-loading cell first; the server does not download model weights."
            ) from exc


def _authorized() -> bool:
    expected = os.getenv("FACADEFORGE_REMOTE_TOKEN", "")
    supplied = request.headers.get("Authorization", "")
    if not expected:
        return False
    return hmac.compare_digest(supplied, f"Bearer {expected}")


def _prepare_sketch(image: Image.Image) -> Image.Image:
    """Keep the notebook's contain-on-white 512x512 + Canny(50, 150) path."""
    image = ImageOps.contain(image.convert("RGB"), (512, 512))
    canvas = Image.new("RGB", (512, 512), "white")
    canvas.paste(image, ((512 - image.width) // 2, (512 - image.height) // 2))
    gray = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return Image.fromarray(edges).convert("RGB")


@app.get("/health")
def health():
    if not _authorized():
        return jsonify(error="Unauthorized. Supply a valid bearer token."), 401
    return jsonify(
        available=_pipeline is not None or torch.cuda.is_available(),
        model_ready=_pipeline is not None,
        cuda_available=torch.cuda.is_available(),
        base_model=BASE_MODEL,
        controlnet=CONTROLNET_MODEL,
        message="Colab is connected; the pipeline will load from the local model cache on the first generation." if _pipeline is None else "The SD 1.5 + Canny ControlNet pipeline is ready.",
    )


@app.post("/generate")
def generate():
    if not _authorized():
        return jsonify(error="Unauthorized. Supply a valid bearer token."), 401
    upload = request.files.get("sketch")
    if upload is None or not upload.filename:
        return jsonify(error="A sketch image is required."), 400
    style = request.form.get("style", "").strip().lower()
    if style not in STYLES:
        return jsonify(error="Unknown style. Choose modern, luxury, indian, industrial, or scandinavian."), 400
    source = upload.read(MAX_UPLOAD_BYTES + 1)
    if len(source) > MAX_UPLOAD_BYTES:
        return jsonify(error="Sketch exceeds the 8 MB upload limit."), 413
    try:
        original = Image.open(io.BytesIO(source))
        if original.format not in {"PNG", "JPEG"}:
            return jsonify(error="Upload a PNG or JPG sketch."), 415
        if original.width * original.height > MAX_IMAGE_PIXELS:
            return jsonify(error="Sketch has too many pixels; resize it and try again."), 413
        original.load()
        control_image = _prepare_sketch(original)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return jsonify(error="The uploaded file is not a valid PNG or JPG image."), 400

    try:
        pipeline = _get_pipeline()
    except ServerConfigurationError as exc:
        return jsonify(error=str(exc)), 503

    prompt = f"{BASE_PROMPT} Architectural direction: {STYLES[style]}."
    generator = torch.Generator(device="cpu").manual_seed(secrets.randbelow(2**31))
    try:
        with _inference_lock:
            result = pipeline(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=control_image,
                width=512,
                height=512,
                num_inference_steps=20,
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0,
                generator=generator,
            )
        image_buffer = io.BytesIO()
        result.images[0].save(image_buffer, format="PNG")
        image_buffer.seek(0)
        return send_file(image_buffer, mimetype="image/png", download_name="facadeforge-render.png", max_age=0)
    except Exception:
        app.logger.exception("Colab facade inference failed")
        return jsonify(error="The Colab model failed during generation. Check the Colab server output and try again."), 500


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify(error="Sketch exceeds the 8 MB upload limit."), 413


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


_server_thread: threading.Thread | None = None


def run_server(host: str = "0.0.0.0", port: int = 7860) -> None:
    if not os.getenv("FACADEFORGE_REMOTE_TOKEN"):
        raise ServerConfigurationError("Set FACADEFORGE_REMOTE_TOKEN in the Colab runtime before starting the server.")
    if not torch.cuda.is_available():
        raise ServerConfigurationError("Colab has no CUDA GPU. Select Runtime > Change runtime type > T4 GPU first.")
    if is_port_in_use(port):
        return
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)


def start_server(host: str = "0.0.0.0", port: int = 7860) -> threading.Thread | None:
    """Safely start the server in a background daemon thread, preventing duplicates."""
    global _server_thread
    if _server_thread is not None and _server_thread.is_alive():
        print("FacadeForge server thread is already active.")
        return _server_thread
    if is_port_in_use(port):
        print(f"FacadeForge server is already listening on port {port}.")
        return None
    _server_thread = threading.Thread(
        target=run_server,
        kwargs={"host": host, "port": port},
        daemon=True,
    )
    _server_thread.start()
    print(f"FacadeForge server thread started on {host}:{port}.")
    return _server_thread


if __name__ == "__main__":
    run_server()
