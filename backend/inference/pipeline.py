"""Lazy, local-only adapter for the model pipeline used in facade_forge.ipynb.

Model weights are never downloaded by the application. Put compatible model files in
the Hugging Face cache first, then set FACADEFORGE_ALLOW_LOCAL_MODEL=1. This mirrors
the notebook's pipeline while keeping the web app usable when inference is unavailable.
"""

from __future__ import annotations

import os
import secrets
import threading
import importlib.util

import numpy as np
from PIL import Image, ImageOps

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
_pipeline = None
_load_lock = threading.Lock()
_pipeline_error: str | None = None
_inference_lock = threading.Lock()

STYLES = {
    "modern": "modern minimalist architecture, clean geometric lines, smooth white concrete, expansive glazing, restrained landscaping",
    "luxury": "luxury contemporary architecture, refined natural stone, warm timber, premium glass, elegant exterior lighting",
    "indian": "contemporary traditional Indian residential architecture, warm sandstone, subtle jaali screens, shaded verandah, climate-aware detailing",
    "industrial": "industrial modern architecture, exposed concrete, dark steel framing, textured masonry, large factory-style windows",
    "scandinavian": "Scandinavian architecture, pale timber, soft neutral stone, simple pitched roof details, calm natural daylight",
}

BASE_PROMPT = (
    "Photorealistic architectural exterior visualization of the exact building shown in the reference sketch. "
    "Professional architectural photography, realistic materials, natural daylight, detailed facade textures. "
    "Preserve the original building geometry, floor count, roofline, window positions, door positions, balconies and proportions."
)
NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, distorted architecture, extra floors, missing floors, extra windows, "
    "missing windows, deformed building, unrealistic geometry, blurry, low quality, watermark, text"
)


class InferenceUnavailable(RuntimeError):
    """Raised when this machine cannot run the local pretrained pipeline."""


def pipeline_status() -> dict:
    """Report prerequisites without loading weights or attempting a download."""
    required_modules = {"diffusers": "Diffusers", "cv2": "OpenCV", "torch": "PyTorch"}
    missing = [label for module, label in required_modules.items() if importlib.util.find_spec(module) is None]
    cuda_available = False
    if importlib.util.find_spec("torch") is not None:
        import torch

        cuda_available = torch.cuda.is_available()
    if missing or not cuda_available:
        details = []
        if missing:
            details.append("missing packages: " + ", ".join(missing))
        if not cuda_available:
            details.append("PyTorch does not have an available CUDA GPU")
        return {
            "available": False,
            "state": "gpu_unavailable" if not cuda_available else "missing_dependencies",
            "message": "Local inference is unavailable: " + "; ".join(details) + ". The notebook's SD 1.5 + Canny ControlNet pipeline needs these local prerequisites.",
            "models": [BASE_MODEL, CONTROLNET_MODEL],
            "model_loading": False,
        }
    if os.getenv("FACADEFORGE_ALLOW_LOCAL_MODEL") != "1":
        return {
            "available": False,
            "state": "not_configured",
            "message": "CUDA and inference packages are available, but local model inference is not enabled. Cache both model weights, then set FACADEFORGE_ALLOW_LOCAL_MODEL=1.",
            "models": [BASE_MODEL, CONTROLNET_MODEL],
            "model_loading": False,
        }
    return {
        "available": _pipeline is not None,
        "state": "ready" if _pipeline is not None else "configured",
        "message": _pipeline_error or ("The local model is ready." if _pipeline is not None else "CUDA is available. The model will load from the local cache on the first request; no weights will be downloaded."),
        "models": [BASE_MODEL, CONTROLNET_MODEL],
        "model_loading": False,
    }


def _get_pipeline():
    global _pipeline, _pipeline_error
    status = pipeline_status()
    if status["state"] in {"not_configured", "missing_dependencies", "gpu_unavailable"}:
        raise InferenceUnavailable(status["message"])
    if _pipeline is not None:
        return _pipeline
    with _load_lock:
        if _pipeline is not None:
            return _pipeline
        try:
            import torch
            from diffusers import (
                ControlNetModel,
                DPMSolverMultistepScheduler,
                StableDiffusionControlNetPipeline,
            )

            # local_files_only prevents accidental multi-gigabyte downloads on app startup.
            controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_MODEL,
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=True,
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                BASE_MODEL,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=True,
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            _pipeline = pipe
            _pipeline_error = None
            return _pipeline
        except Exception as exc:
            _pipeline_error = (
                f"Could not load the SD 1.5 + Canny ControlNet weights from the local Hugging Face cache. "
                f"No download was attempted ({exc.__class__.__name__})."
            )
            raise InferenceUnavailable(_pipeline_error) from exc


def prepare_sketch(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Match notebook preprocessing: contain on white 512px canvas, Canny 50/150."""
    import cv2

    contained = ImageOps.contain(image.convert("RGB"), (512, 512))
    canvas = Image.new("RGB", (512, 512), "white")
    canvas.paste(contained, ((512 - contained.width) // 2, (512 - contained.height) // 2))
    gray = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return canvas, Image.fromarray(edges).convert("RGB")


def render_facade(image: Image.Image, style: str) -> Image.Image:
    pipe = _get_pipeline()
    _original, edge_image = prepare_sketch(image)
    import torch

    generator = torch.Generator(device="cpu").manual_seed(secrets.randbelow(2**31))
    prompt = f"{BASE_PROMPT} Architectural direction: {STYLES[style]}"
    # Keep the single shared pipeline safe from simultaneous GPU inference calls.
    with _inference_lock:
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=edge_image,
            width=512,
            height=512,
            num_inference_steps=20,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            generator=generator,
        )
    return result.images[0]
