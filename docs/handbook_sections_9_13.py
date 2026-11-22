# -*- coding: utf-8 -*-
"""Chapters 9 to 13 for FacadeForge AI Viva Preparation Handbook."""

SECTION_9_TO_13 = """
## 9. Installation, Setup, & Step-by-Step Execution Guide

This section provides reproducible, step-by-step instructions for bringing up both the local web gateway and the remote Google Colab GPU inference worker.

### 9.1 Prerequisites
- **Local Machine:** Windows 10/11 (or macOS / Linux), Python 3.10+ installed, active Internet connection.
- **Remote Machine:** Google account with access to Google Colab (Free tier or Colab Pro).
- **Git:** Git CLI installed locally.

---

### 9.2 Part 1: Setting up the Remote GPU Server in Google Colab

1. **Open the Notebook:**  
   Navigate to [Google Colab](https://colab.research.google.com/) and upload `facade_forge.ipynb` from the project root.
2. **Select GPU Hardware Accelerator:**  
   In the Colab menu, select **Runtime → Change runtime type**. Under *Hardware accelerator*, choose **T4 GPU** (Standard). Click **Save**.
3. **Run Pipeline Initialization Cells (Cells 0 to 6):**  
   Execute cells sequentially:
   - Cell 0: Verifies PyTorch and detects `Nvidia Tesla T4`.
   - Cell 2: Installs `diffusers`, `transformers`, `accelerate`, and `safetensors`.
   - Cell 5: Loads `lllyasviel/sd-controlnet-canny` and `runwayml/stable-diffusion-v1-5` into GPU memory in `torch.float16`, attaches `DPMSolverMultistepScheduler`, and enables attention slicing. Confirm output displays: *"Both models loaded successfully!"*
   - Cell 6: Confirms GPU allocation and pipeline readiness.
4. **Execute the Remote Server & Tunnel Cell (Cell 12):**  
   Run the final integrated server cell. This cell:
   - Installs `Flask>=3.0,<4.0` if not present.
   - Downloads the `cloudflared` Debian package and installs it via `dpkg`.
   - Downloads the latest `colab_server.py` from the official GitHub repository.
   - Prompts the user securely for the shared secret token via `getpass.getpass()`.
   - Registers the existing in-memory `pipe` with `colab_server.set_pipeline(pipe)`.
   - Starts the server on `0.0.0.0:7860` in a background daemon thread.
   - Spawns `cloudflared tunnel --url http://127.0.0.1:7860` in a detached daemon session.
   - Monitors `/content/tunnel.log` and prints the active public URL:
     ```
     ============================================================
     SUCCESS! Your Cloudflare Tunnel is live at:
     https://achieving-possible-ground-record.trycloudflare.com
     ============================================================
     ```
   **Important:** Copy this `https://*.trycloudflare.com` URL. Keep this Colab tab open.

---

### 9.3 Part 2: Setting up the Local Web Gateway (Windows)

1. **Clone the Repository & Navigate to Directory:**
   ```powershell
   git clone https://github.com/varshini-vemula04/FacadeForge_AI.git
   cd FacadeForge_AI
   ```
2. **Create and Activate Virtual Environment:**
   ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. **Install Minimal Gateway Requirements:**
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
   *(Note: Do NOT install `requirements-ai.txt` on the local laptop; that is the legacy local-inference file. Only `requirements.txt` containing Flask, Pillow, and NumPy is needed).*
4. **Configure Environment Variables (`.env`):**
   Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```
   Populate `.env` with the exact tunnel URL from Colab and the shared token:
   ```dotenv
   FACADEFORGE_REMOTE_URL=https://achieving-possible-ground-record.trycloudflare.com
   FACADEFORGE_REMOTE_TOKEN=your-exact-generated-shared-secret-token
   FACADEFORGE_REMOTE_TIMEOUT_SECONDS=300
   ```
   Save and close the file.
5. **Start the Flask Web Gateway:**
   ```powershell
   $env:PYTHONPATH="backend"
   python backend\app.py
   ```
   The terminal will output:
   ```
   FacadeForge is starting at http://127.0.0.1:5000
   Demo portfolio images are examples. Workspace outputs are only created by a successful inference run.
   * Serving Flask app 'app'
   * Running on http://127.0.0.1:5000
   ```
6. **Open in Browser:**  
   Navigate to `http://127.0.0.1:5000/workspace`. Confirm the status indicator turns green: *"Colab is connected and the SD 1.5 + ControlNet pipeline is loaded."*

---

### 9.4 Verification & Troubleshooting Checklist

| Symptom / Error | Root Cause | Exact Remediation |
|---|---|---|
| **Status: Disconnected / "Could not reach Colab"** | The Cloudflare tunnel is not running, Colab runtime disconnected, or the URL in `.env` has an error or extra slash. | Check Colab tab. Confirm tunnel cell is active. Re-copy the `https://*.trycloudflare.com` URL into `.env`. No Flask restart is required because `.env` auto-reloads on each request. |
| **Status: Unauthorized (HTTP 401)** | Token mismatch between local `.env` and Colab's `FACADEFORGE_REMOTE_TOKEN`. | Ensure the exact same string is set in both locations without leading or trailing spaces. |
| **Colab Cell Crashes: "Your session crashed after using all available RAM"** | Occurs when using `pipe.enable_model_cpu_offload()` which attempts to offload multi-gigabyte models into Colab's 12 GB system RAM. | In Cell 5 of the notebook, ensure `pipe = pipe.to("cuda")` is used instead of CPU offload. Keep the entire pipeline inside the 16 GB GPU VRAM. |
| **HTTP 530 / Error 1033 in Browser Console** | The Cloudflare tunnel terminated because it was bound to a foreground subshell that closed when another cell executed. | Launch `cloudflared` using `subprocess.Popen(..., start_new_session=True)` in Python so the tunnel runs in an isolated process group immune to Jupyter kernel interrupts. |
| **Port 7860: "Address already in use"** | Rerunning Colab server cell without terminating the previous Flask process. | `colab_server.py` includes `is_port_in_use(port)`. It safely reuses the active thread or checks port availability before binding. |
| **"RuntimeError: Model pipeline 'pipe' not found in globals"** | The user executed Cell 12 (the server cell) before running Cells 0–6. | Run cells 0 through 6 first so that `pipe` is fully loaded into Colab's global memory. |

---

## 10. Key Technical Design Decisions & Justifications

Examiners frequently ask **"Why did you choose X instead of Y?"** This section arms the team with solid technical justifications.

### 10.1 Why Stable Diffusion 1.5 instead of SDXL or Midjourney?
| Criteria | Stable Diffusion 1.5 (Chosen) | SDXL (Alternative) | Midjourney / DALL-E 3 (Alternative) |
|---|---|---|---|
| **Hardware & VRAM** | Runs comfortably in ~7 GB VRAM (`float16`). Perfectly fits Colab's free 16 GB T4 GPU. | Requires >16 GB VRAM for stable ControlNet execution; frequently triggers OOM crashes on free Colab. | Proprietary cloud APIs with zero control over neural weights or custom adapters. |
| **Spatial Control** | Mature, highly stable ControlNet v1.1 checkpoints available natively. | ControlNet adapters for SDXL are significantly heavier and less stable. | Pure prompt-based; impossible to pass exact Canny edge tensors to preserve architectural blueprints. |
| **Cost & Independence** | 100% open-source, reproducible, zero API subscription costs. | Open-source, but requires costly paid GPU compute (A100 / V100). | Recurring per-image API billing; violates academic project self-containment requirements. |
| **Inference Latency** | **8–12 seconds** with DPMSolver++ (20 steps). | 35–60 seconds per image. | Variable network queuing. |

### 10.2 Why ControlNet Canny instead of MLSD, Depth, or Pure Text-to-Image?
| Method | Strengths | Weaknesses | Why Canny Won for FacadeForge AI |
|---|---|---|---|
| **Pure Text-to-Image** | High material creativity. | Zero structural control; hallucinates random doors, windows, and floors. | Completely disqualified for architecture; clients require exact floor and window alignment. |
| **ControlNet MLSD (Line Detection)** | Detects clean straight geometric lines. | Fails completely on curved arches, organic masonry, terrain, and hand-drawn perspective sketches. | Too rigid; fails on real-world conceptual hand sketches with slight curvature or shading. |
| **ControlNet Depth (MiDaS)** | Captures 3D spatial depth. | Cannot extract fine details from 2D flat paper sketches (a piece of paper has no 3D depth field). | Requires existing 3D geometric surfaces; hand sketches produce ambiguous flat depth maps. |
| **ControlNet Canny (Chosen)** | **Extracts all structural contours, window frames, rooflines, and floor slabs across both digital drafts and hand drawings.** | Captures paper smudges if thresholds are too low. | **Selected because its dual-threshold hysteresis ($50/150$) isolates clean structural boundaries while ignoring paper texture.** |

### 10.3 Why Distributed Hybrid Architecture instead of Pure Local or Pure Cloud?
- **Why not Pure Local?** Requiring an Nvidia RTX 3080/4090 GPU on every user's personal machine renders the software inaccessible to 95% of architecture students and small firms who work on lightweight ultrabooks or MacBooks.
- **Why not Pure Cloud (Full Website on AWS/Heroku)?** Hosting full-stack GPU servers on AWS EC2 (`g4dn.xlarge`) costs hundreds of dollars monthly. By hosting the lightweight web gateway locally on Windows and tapping into free Colab T4 GPUs for inference, the system achieves enterprise-grade GPU capability at **zero infrastructure cost**.

### 10.4 Why Flask instead of FastAPI or Django?
- **Django:** Massive monolithic framework with built-in ORM, user auth, and admin panels—completely unnecessary bloat for a focused generative visualization studio.
- **FastAPI:** Highly asynchronous, but introduces complex async/await event loop dynamics with multi-threaded PyTorch tensor operations.
- **Flask (Chosen):** Minimalist, robust, and transparent. Standard synchronous endpoints combined with clean Python `threading.Thread` background tasks allow effortless decoupling of web serving and multi-minute inference calls.

---

## 11. Real Technical Challenges Overcome & Documented Limitations

Honest acknowledgement of challenges and their engineering solutions demonstrates genuine development effort to an examination panel.

### 11.1 Real Engineering Challenges Solved During Development

#### Challenge 1: The Colab System RAM Crash (OOM)
- **The Issue:** Early notebook iterations utilized `pipe.enable_model_cpu_offload()`. While this saves GPU VRAM by moving inactive layers to system memory, it triggered severe system RAM exhaustion in Colab (crashing at 12.2 GB / 12.6 GB system RAM) during the forward pass.
- **The Engineering Fix:** We replaced CPU offloading with direct GPU memory management: `pipe = pipe.to("cuda")` in conjunction with `pipe.enable_attention_slicing()`. This shifted model weight residency entirely into the 16 GB GDDR6 GPU VRAM (utilizing ~7.2 GB), keeping system RAM usage below 4 GB and completely stabilizing the runtime.

#### Challenge 2: Background Cloudflare Tunnel Termination
- **The Issue:** Running `!cloudflared tunnel` directly in a Colab bash cell blocked the Jupyter notebook interface indefinitely. When aborted to run subsequent cells, the subshell was sent a `SIGINT` signal, instantly terminating the public tunnel and breaking remote communication.
- **The Engineering Fix:** We re-engineered tunnel orchestration using Python's `subprocess.Popen` with `start_new_session=True`. This detaches the tunnel daemon into an independent Unix process session immune to Jupyter kernel interrupts, redirecting logs to `/content/tunnel.log` while freeing the notebook interface.

#### Challenge 3: Port 7860 Binding Collisions
- **The Issue:** When developers re-executed notebook cells, Flask attempted to bind to port 7860 while the previous server instance was still listening, raising `OSError: [Errno 98] Address already in use`.
- **The Engineering Fix:** We implemented `is_port_in_use(port)` inside `colab_server.py`. The server checks socket connectivity before attempting to bind, safely reusing the existing thread if active.

#### Challenge 4: The Lingering Empty-State Blueprint Bug
- **The Issue:** In the web workspace, after generation completed, the large 340px empty-state canvas ("YOUR CANVAS IS READY") lingered on screen above the generated comparison grid, forcing the user to scroll awkwardly down the page.
- **The Engineering Fix:** We traced this to CSS specificity. Author stylesheet rules like `.result-empty { display: flex; }` override the browser's default user-agent `[hidden] { display: none; }`. We added explicit global and class-level rules: `[hidden], .result-empty[hidden] { display: none !important; }`. This guarantees that `.result-empty` is instantly excised from the document layout upon generation, snapping the generated render directly beneath the "Visualization" heading.

### 11.2 Documented Project Limitations
1. **Colab Runtime Ephemerality:** Google Colab free tier instances disconnect after 30–90 minutes of user idle time, requiring manual re-initialization of Cell 12.
2. **Tunnel URL Mutability:** Because Cloudflare Quick Tunnels are ephemeral, restarting the tunnel generates a new `trycloudflare.com` subdomain, requiring updating `FACADEFORGE_REMOTE_URL` in `.env`.
3. **Single GPU Concurrency:** The remote inference worker uses a single Tesla T4 GPU with a thread lock (`_inference_lock`). If multiple users submit generation requests simultaneously, they are processed sequentially rather than concurrently.
4. **Input Sketch Quality Sensitivity:** If a sketch is extremely faint (low contrast) or heavily textured with dense cross-hatching, the Canny edge detector may extract unwanted noise. The sketch must have discernible structural line boundaries.

---

## 12. Future Roadmap & Research Extensions

These proposals represent realistic, technically sound extensions for post-graduate development or production scaling:

```
+----------------------------------------------------------------------------------------------------+
|                                    FACADEFORGE AI FUTURE ROADMAP                                   |
+----------------------------------------------------------------------------------------------------+
|  PHASE 1: MULTI-MODAL CONDITIONING                                                                 |
|  - Multi-ControlNet: Dual conditioning with Canny (geometry) + MiDaS Depth (3D depth relief)         |
|  - Segment Anything Model (SAM): Automated architectural semantic segmentation (walls, glass, roof)|
+----------------------------------------------------------------------------------------------------+
|  PHASE 2: PRODUCTION INFRASTRUCTURE                                                                |
|  - Serverless GPU Workers: Migrate from Colab to RunPod / Modal / AWS ECS with auto-scaling        |
|  - Persistent PostgreSQL Database: Historical job tracking, user auth, and cloud S3 asset storage  |
+----------------------------------------------------------------------------------------------------+
|  PHASE 3: DOMAIN FINE-TUNING & HIGH RESOLUTION                                                     |
|  - Architectural LoRA Adapters: Fine-tuned on CMP Facade Dataset and ArchiDaily photography         |
|  - Real-ESRGAN Latent Upscaling: Enhance native 512x512 renders to 2K/4K presentation visuals     |
|  - BIM/CAD Plugin Integrations: Direct export plugins for Autodesk Revit, Rhino, and SketchUp      |
+----------------------------------------------------------------------------------------------------+
```

### 1. Multi-ControlNet Dual Conditioning (Canny + MiDaS Depth)
While Canny edge detection excels at capturing 2D boundary lines, it lacks depth perception (it cannot distinguish between a balcony projecting 2 meters forward versus a window recessed 1 meter backward). Integrating **Multi-ControlNet** will allow the pipeline to condition simultaneously on a Canny edge map (for boundary sharpness) and a monocular depth map generated via **MiDaS or Depth Anything**, producing photorealistic volumetric depth and accurate shadow casting.

### 2. Architectural LoRA Fine-Tuning
The base Stable Diffusion 1.5 model is trained on the general LAION-5B dataset (containing everyday objects, people, and landscapes). Training lightweight **Low-Rank Adaptation (LoRA)** weights on specialized architectural facade datasets (such as the CMP Facade Dataset or high-end architectural photography collections) will significantly improve material fidelity—generating authentic board-formed concrete, curtain wall mullions, and architectural reveals.

### 3. Serverless GPU Deployment (RunPod / Modal / AWS EC2)
Migrate the inference microservice from ephemeral Google Colab runtimes to serverless GPU infrastructure such as **Modal Labs, RunPod Serverless, or AWS EC2 g4dn instances**. Serverless containers spin up on demand in 5–10 seconds, process the inference request on dedicated Nvidia A10G or L4 GPUs, and scale to zero when idle, providing 99.9% uptime with static API endpoints.

### 4. High-Resolution Latent Upscaling (Real-ESRGAN)
The native resolution of SD 1.5 is $512 \times 512$ pixels. Integrating a secondary **Real-ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)** or Latent Upscaler stage will upscale generated facade renders to $2048 \times 2048$ (2K) or $4096 \times 4096$ (4K) print-ready resolution, sharpening glass reflections, brick mortar lines, and wood grain for client presentations.

---

## 13. Individual Module Breakdown & Viva Defense Scripts

Examiners frequently ask individual students: **"What specifically did you work on, and how does your part connect to the rest of the system?"**  
Use these prepared scripts based on your assigned focus area.

### Role 1: Frontend & User Experience Specialist
- **Your Domain:** `frontend/templates/`, `frontend/static/css/site.css`, `frontend/static/js/`
- **What to say in Viva:**  
  *"My primary responsibility was designing and implementing the client-side user experience and asynchronous state engine. I developed the custom architectural editorial design system using semantic HTML5 and responsive CSS3, establishing an intuitive dual-panel layout where the sketch input and rendered visualization sit side-by-side.*  
  *On the scripting side, I implemented `workspace.js`, which handles drag-and-drop sketch ingestion, client-side MIME validation, and non-blocking object URL previews. I also engineered the asynchronous polling engine (`pollJob`) that queries the Flask gateway every 1,200 ms to monitor remote generation progress without freezing the browser interface.*  
  *Crucially, I resolved layout shifts by enforcing strict CSS hidden states so that the empty-state canvas disappears cleanly when the render completes, positioning the generated facade directly beneath the Visualization heading."*

### Role 2: Backend Architecture & API Gateway Specialist
- **Your Domain:** `backend/app.py`, `backend/generated/`, request validation, job queue
- **What to say in Viva:**  
  *"My core responsibility was engineering the local application gateway and asynchronous job lifecycle in `backend/app.py`. Because generative AI inference takes several seconds, standard synchronous HTTP requests would cause browser gateway timeouts.*  
  *I designed an asynchronous architecture: when the client posts a sketch to `/api/generate`, my module validates the image against decompression bombs and size limits, generates a unique UUID4 job token, registers it in a thread-safe in-memory queue protected by `threading.Lock()`, and immediately returns an HTTP 202 Accepted response.*  
  *I implemented background worker threads that handle the network communication with the remote server, write completed PNG files to disk, and expose endpoints for polling status, streaming image bytes, and downloading results. I also implemented dynamic `.env` hot-reloading using Flask's `@app.before_request` hook so the application adapts to new tunnel URLs without restarts."*

### Role 3: AI Model, Computer Vision, & Pipeline Specialist
- **Your Domain:** `facade_forge.ipynb`, `backend/inference/pipeline.py`, Canny edge detection, Diffusers
- **What to say in Viva:**  
  *"My focus was the generative AI and computer vision pipeline. I integrated Stable Diffusion v1.5 with ControlNet Canny using Hugging Face Diffusers. The core of my work was establishing the structure-guided conditioning workflow.*  
  *I implemented the `_prepare_sketch` computer vision module, which takes arbitrary input sketches, uses Pillow's `ImageOps.contain` to preserve exact aspect ratios on a 512×512 white canvas, converts the image to grayscale, and executes multi-stage Canny edge detection with hysteresis thresholds of 50 and 150.*  
  *I configured the diffusion hyperparameters: utilizing the DPMSolverMultistepScheduler to achieve rapid convergence in 20 steps, setting Classifier-Free Guidance to 7.5 to balance material fidelity with visual realism, and enforcing a ControlNet conditioning scale of 1.0 to ensure strict architectural boundary compliance. I also engineered the curated positive and negative prompt templates that drive the five architectural styles."*

### Role 4: Cloud Deployment, Security, & Remote Tunneling Specialist
- **Your Domain:** `backend/inference/colab_server.py`, `backend/inference/remote_client.py`, Cloudflare Tunnel
- **What to say in Viva:**  
  *"My focus was the distributed infrastructure, network tunneling, and remote microservice architecture. Because deep learning models require heavy GPU hardware unavailable on standard laptops, I established a secure remote inference bridge connecting our local gateway to a Google Colab Nvidia Tesla T4 GPU.*  
  *I implemented `colab_server.py`, creating a lightweight Flask microservice on Colab port 7860 that registers the notebook's active in-memory pipeline (`set_pipeline`) to prevent redundant VRAM loading and avoid memory crashes. I engineered the `cloudflared` daemon process in an isolated Unix process session to expose the service over an encrypted HTTPS tunnel.*  
  *On the security front, I implemented cryptographic HMAC-SHA256 bearer token authentication using `hmac.compare_digest` to neutralize timing side-channel attacks, and wrote `remote_client.py` with custom multipart form serialization and multi-layer PNG response verification."*

### Role 5: Integration, QA, & Performance Optimization Specialist
- **Your Domain:** Error handling, hardware optimization, end-to-end testing, `.env`
- **What to say in Viva:**  
  *"My responsibility was end-to-end integration, performance tuning, and failure resilience. I solved the critical GPU Out-of-Memory crashes in Colab by identifying that `enable_model_cpu_offload` was crashing system RAM, and transitioned the pipeline to direct GPU residency (`pipe.to('cuda')`) combined with half-precision `torch.float16` and attention slicing.*  
  *I conducted comprehensive failure mode testing: verifying how the system behaves under network timeouts, closed tunnels, corrupted image uploads, and invalid bearer tokens. I ensured that the local gateway never substitutes mock demo images for genuine inference outputs, guaranteeing that every visualization presented to the user is verified, genuine deep-learning output."*
"""
