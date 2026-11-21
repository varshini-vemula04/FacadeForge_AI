"""FacadeForge web application and asynchronous generation endpoints."""

from __future__ import annotations

import io
import os
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]


def _load_environment_file(path: Path) -> None:
    """Load the simple KEY=value format used in .env.example without another dependency."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


_load_environment_file(ROOT / ".env")

from inference.remote_client import RemoteInferenceError, check_remote_health, generate_remote_image

GENERATED_DIR = Path(__file__).resolve().parent / "generated"
GENERATED_DIR.mkdir(exist_ok=True)
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_FORMATS = {"PNG", "JPEG"}

app = Flask(
    __name__,
    template_folder=str(ROOT / "frontend" / "templates"),
    static_folder=str(ROOT / "frontend" / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/assets/<path:filename>")
def project_asset(filename: str):
    return send_from_directory(ROOT / "assets", filename)


@app.get("/workspace")
def workspace():
    return render_template("workspace.html")


@app.get("/api/status")
def status():
    return jsonify(check_remote_health())


def _run_job(job_id: str, source: bytes, style: str) -> None:
    with _jobs_lock:
        _jobs[job_id]["state"] = "running"
        _jobs[job_id]["message"] = "Sending the sketch to the Colab inference service."
    try:
        rendered = generate_remote_image(source, style)
        output_path = GENERATED_DIR / f"{job_id}.png"
        output_path.write_bytes(rendered)
        with _jobs_lock:
            _jobs[job_id].update(
                state="complete",
                message="Facade visualization generated.",
                result_url=f"/api/jobs/{job_id}/image",
                download_url=f"/api/jobs/{job_id}/download",
            )
    except RemoteInferenceError as exc:
        with _jobs_lock:
            _jobs[job_id].update(state="error", message=str(exc))
    except Exception as exc:  # Keep internal details in the server log, not user-facing text.
        app.logger.exception("Facade generation failed")
        with _jobs_lock:
            _jobs[job_id].update(
                state="error",
                message=f"Generation failed: {exc.__class__.__name__}. Check the backend terminal for details.",
            )


@app.post("/api/generate")
def generate():
    upload = request.files.get("sketch")
    if upload is None or not upload.filename:
        return jsonify(error="Choose a sketch image before generating."), 400
    source = upload.read(MAX_UPLOAD_BYTES + 1)
    if len(source) > MAX_UPLOAD_BYTES:
        return jsonify(error="The image is larger than 8 MB. Please choose a smaller file."), 413
    try:
        image = Image.open(io.BytesIO(source))
        if image.width * image.height > 40_000_000:
            return jsonify(error="That image has too many pixels to process safely. Resize it and try again."), 413
        image.verify()
        image_format = image.format
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return jsonify(error="That file could not be read as a valid PNG or JPG image."), 400
    if image_format not in ALLOWED_FORMATS:
        return jsonify(error="Upload a PNG or JPG image."), 415
    style = request.form.get("style", "modern").strip().lower()
    allowed_styles = {"modern", "luxury", "indian", "industrial", "scandinavian"}
    if style not in allowed_styles:
        return jsonify(error="Select one of the available architectural styles."), 400

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"id": job_id, "state": "queued", "message": "Waiting to contact Colab."}
    # The HTTP request to Colab runs in a worker thread, keeping Flask and the page responsive.
    threading.Thread(target=_run_job, args=(job_id, source, style), daemon=True).start()
    return jsonify(id=job_id, state="queued"), 202


@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify(error="This generation request was not found. It may have expired when the server restarted."), 404
        return jsonify(job)


@app.get("/api/jobs/<job_id>/image")
def get_job_image(job_id: str):
    path = GENERATED_DIR / f"{job_id}.png"
    if not path.is_file():
        return jsonify(error="The generated image is not ready."), 404
    return send_file(path, mimetype="image/png", max_age=0)


@app.get("/api/jobs/<job_id>/download")
def download_job_image(job_id: str):
    path = GENERATED_DIR / f"{job_id}.png"
    if not path.is_file():
        return jsonify(error="The generated image is not ready."), 404
    return send_file(path, mimetype="image/png", as_attachment=True, download_name="facadeforge-render.png")


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="The image is larger than 8 MB. Please choose a smaller file."), 413


if __name__ == "__main__":
    print("FacadeForge is starting at http://127.0.0.1:5000")
    print("Demo portfolio images are examples. Workspace outputs are only created by a successful inference run.")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
