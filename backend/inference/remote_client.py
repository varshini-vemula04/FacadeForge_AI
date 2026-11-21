"""Authenticated HTTP client for the Colab-hosted FacadeForge model server."""

from __future__ import annotations

import io
import json
import os
import socket
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

MAX_GENERATED_BYTES = 24 * 1024 * 1024


class RemoteInferenceError(RuntimeError):
    """A useful, user-safe remote connection or inference error."""


def _configuration() -> tuple[str, str]:
    base_url = os.getenv("FACADEFORGE_REMOTE_URL", "").strip().rstrip("/")
    token = os.getenv("FACADEFORGE_REMOTE_TOKEN", "").strip()
    if not base_url or not token:
        raise RemoteInferenceError(
            "Colab is not connected. Set FACADEFORGE_REMOTE_URL and FACADEFORGE_REMOTE_TOKEN in the local .env file, then restart Flask."
        )
    parsed = urlparse(base_url)
    if not parsed.hostname or parsed.scheme not in {"https", "http"} or parsed.query or parsed.fragment:
        raise RemoteInferenceError("FACADEFORGE_REMOTE_URL must be a valid HTTPS base URL without a query string or fragment.")
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise RemoteInferenceError("The remote inference endpoint must use HTTPS so the shared token is not sent over plain HTTP.")
    return base_url, token


def _request(path: str, *, data: bytes | None = None, content_type: str | None = None, timeout: float = 8.0):
    base_url, token = _configuration()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json, image/png"}
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(f"{base_url}{path}", data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        return urlopen(request, timeout=timeout)
    except HTTPError as exc:
        detail = ""
        try:
            payload = json.loads(exc.read().decode("utf-8", errors="replace"))
            detail = str(payload.get("error", payload.get("message", "")))
        except (ValueError, AttributeError):
            pass
        if exc.code == 401:
            raise RemoteInferenceError("Colab rejected the shared token. Check that the same FACADEFORGE_REMOTE_TOKEN is set in Colab and local .env.") from exc
        if exc.code in {502, 503, 504}:
            raise RemoteInferenceError(detail or "The Colab model is not ready. Confirm the notebook pipeline is loaded and try again.") from exc
        raise RemoteInferenceError(detail or f"Colab returned HTTP {exc.code}.") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RemoteInferenceError("Timed out waiting for the Colab inference server. Generation can take several minutes; check that Colab is still running.") from exc
    except URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            message = "Timed out connecting to Colab. Check that the Quick Tunnel is running."
        else:
            message = "Could not reach Colab. The runtime may have disconnected or the Quick Tunnel URL may have changed."
        raise RemoteInferenceError(message) from exc


def check_remote_health() -> dict:
    """Authenticated health check for the configured tunnel and Colab worker."""
    try:
        with _request("/health", timeout=8.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("available") is True and payload.get("cuda_available") is True:
            if payload.get("model_ready") is True:
                message = "Colab is connected and the SD 1.5 + ControlNet pipeline is loaded."
            else:
                message = "Colab and its GPU are connected. The first generation will load or verify the models from the existing Colab cache."
            return {"available": True, "state": "connected", "message": message}
        return {
            "available": False,
            "state": "model_not_ready",
            "message": payload.get("message", "The Colab server is reachable, but its model pipeline is not loaded."),
        }
    except RemoteInferenceError as exc:
        return {"available": False, "state": "disconnected", "message": str(exc)}
    except (ValueError, OSError):
        return {"available": False, "state": "invalid_response", "message": "Colab responded with an unreadable health check. Restart the Colab server cell."}


def _multipart(sketch: bytes, style: str) -> tuple[bytes, str]:
    boundary = "----FacadeForge" + uuid.uuid4().hex
    marker = boundary.encode("ascii")
    try:
        with Image.open(io.BytesIO(sketch)) as image:
            image_format = image.format
    except (UnidentifiedImageError, OSError) as exc:
        raise RemoteInferenceError("The uploaded sketch could not be read before sending it to Colab.") from exc
    if image_format not in {"PNG", "JPEG"}:
        raise RemoteInferenceError("The remote service accepts PNG and JPG sketches only.")
    filename = "sketch.jpg" if image_format == "JPEG" else "sketch.png"
    mime_type = "image/jpeg" if image_format == "JPEG" else "image/png"
    chunks = [
        b"--" + marker + b"\r\nContent-Disposition: form-data; name=\"style\"\r\n\r\n" + style.encode("ascii") + b"\r\n",
        b"--" + marker + f"\r\nContent-Disposition: form-data; name=\"sketch\"; filename=\"{filename}\"\r\nContent-Type: {mime_type}\r\n\r\n".encode("ascii") + sketch + b"\r\n",
        b"--" + marker + b"--\r\n",
    ]
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def generate_remote_image(sketch: bytes, style: str) -> bytes:
    """Send a sketch to Colab and return only a verified PNG response."""
    try:
        timeout = max(30.0, min(float(os.getenv("FACADEFORGE_REMOTE_TIMEOUT_SECONDS", "300")), 900.0))
    except ValueError:
        timeout = 300.0
    body, content_type = _multipart(sketch, style)
    try:
        with _request("/generate", data=body, content_type=content_type, timeout=timeout) as response:
            image_bytes = response.read(MAX_GENERATED_BYTES + 1)
            response_type = response.headers.get_content_type()
    except RemoteInferenceError:
        raise
    except OSError as exc:
        raise RemoteInferenceError("The Colab connection closed during generation. Check the runtime and Quick Tunnel, then try again.") from exc

    if response_type != "image/png" or not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RemoteInferenceError("Colab did not return a PNG image. Check the Colab server log; no demonstration image was substituted.")
    if len(image_bytes) > MAX_GENERATED_BYTES:
        raise RemoteInferenceError("Colab returned an image larger than the 24 MB response limit.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as result:
            result.verify()
            if result.format != "PNG":
                raise RemoteInferenceError("Colab returned an image in an unexpected format; expected PNG.")
    except (UnidentifiedImageError, OSError) as exc:
        raise RemoteInferenceError("Colab returned a damaged PNG image. Check the Colab server log and try again.") from exc
    return image_bytes
