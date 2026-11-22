# -*- coding: utf-8 -*-
"""Chapters 1 to 5 for FacadeForge AI Viva Preparation Handbook."""

SECTION_1_TO_5 = """
## 1. Executive Summary & Project Identification

### 1.1 Project Overview
**FacadeForge AI** is an advanced, deep-learning-powered architectural visualization platform designed to bridge the gap between preliminary architectural ideation and realistic visual representation. The platform accepts hand-drawn or digitally drafted 2D architectural building facade sketches and translates them into high-fidelity, photorealistic exterior visualizations while strictly maintaining the building's core structural geometry, floor counts, fenestration (window and door layout), roofline, and spatial proportions.

Architectural design workflows have historically suffered from a severe bottleneck during the concept presentation phase. While conceptual sketches are fast and intuitive for architects to produce, non-technical clients, municipal planning boards, and property investors struggle to visualize finished materiality, environmental lighting, and spatial presence from raw line work. Translating these sketches into presentation-ready visuals traditionally demands labor-intensive 3D CAD/BIM modeling (AutoCAD, Revit, SketchUp), manual UV material mapping, environmental lighting setup, and computationally expensive ray-tracing rendering (V-Ray, Lumion, Corona, Blender), requiring days of effort for early-stage proposals that are subject to frequent revisions.

FacadeForge AI resolves this bottleneck by implementing a **structure-conditioned latent diffusion pipeline**. By coupling **Stable Diffusion v1.5** with **ControlNet Canny edge guidance**, the system transforms rough structural drawings into photorealistic building exteriors in seconds.

### 1.2 Academic & Implementation Identity
| Property | Project Specification |
|---|---|
| **Project Name** | FacadeForge AI |
| **Subtitle** | Sketch-to-Realistic Architectural Building Renderer |
| **Domain** | Generative Artificial Intelligence, Deep Learning, Computer Vision, Architectural Informatics |
| **Primary Technique** | Structure-Conditioned Image-to-Image Translation (Latent Diffusion Models + ControlNet) |
| **Base Generative Model** | Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`) |
| **Spatial Conditioning Model** | ControlNet Canny (`lllyasviel/sd-controlnet-canny`) |
| **Feature Extraction** | Multi-stage Canny Edge Detection (`cv2.Canny`, low threshold = 50, high threshold = 150) |
| **Backend Architecture** | Distributed Hybrid Cloud (Local Flask WSGI Gateway + Remote Cloud GPU Worker) |
| **Remote GPU Infrastructure** | Google Colab Nvidia Tesla T4 GPU (16 GB GDDR6 VRAM, Turing Architecture, Tensor Cores) |
| **Network & Security** | Cloudflare Quick Tunnel (`cloudflared` QUIC/HTTP2) + HMAC-SHA256 Timing-Safe Bearer Token Authentication |
| **Local Web Gateway** | Python 3.10, Flask 3.0.x, Pillow 10.x, NumPy, Werkzeug |
| **Frontend Web App** | Semantic HTML5, Custom Architectural Editorial CSS3 (DM Sans & Playfair Display), Vanilla ES6 JavaScript |
| **Code Repository State** | Main Git branch synchronized at `https://github.com/varshini-vemula04/FacadeForge_AI` |

---

## 2. Project Introduction, Problem Statement, & Core Objectives

### 2.1 The Architectural Visualization Bottleneck
In architectural and civil engineering practice, early design ideation is predominantly visual and analog. Architects begin with quick conceptual facade sketches—either hand-drawn on tracing paper or drafted digitally with vector stylus tools. These sketches communicate essential architectural intent:
- Structural zoning and symmetry.
- Number of storeys / floor-to-floor heights.
- Fenestration rhythm (patterns of windows, curtain walls, and entryways).
- Cantilevers, balconies, pergolas, and rooflines.

However, a fundamental communication gap exists:
1. **Client Cognitive Gap:** Clients and prospective buyers frequently lack the spatial training required to mentally extrapolate a finished building with concrete, timber, textured stone, reflective double-glazed glass, and ambient daylight from black-and-white 2D lines.
2. **The 3D Modeling Overhead:** The standard industry workflow to produce realistic visuals requires modeling every parametric element in 3D CAD/BIM software, applying PBR (Physically Based Rendering) texture maps, placing digital light sources, and running ray-tracers. This process takes anywhere from 8 to 48 hours per design iteration.
3. **High Iteration Cost:** Early-stage designs are inherently fluid. Spending days rendering a concept that the client may reject within minutes is an enormous waste of professional billable hours.

### 2.2 The Proposed Solution: Structure-Guided Generative AI
FacadeForge AI solves this dilemma through **Structure-Guided Image-to-Image Translation**. Instead of generating arbitrary buildings from scratch or forcing architects into 3D modeling, FacadeForge AI takes the architect's actual sketch as a spatial constraint and synthesizes photorealistic architectural finishes directly onto it.

The foundational principle of FacadeForge AI is:
> **"Preserve the original architectural structure while transforming its material and atmospheric reality."**

The platform guarantees that:
- The building's geometry, floor count, and boundaries remain intact.
- Windows, doors, and balconies do not shift or multiply randomly.
- Photorealistic textures (concrete, timber, sandstone, steel, glass) are applied in accordance with curated architectural design movements.
- Realistic environmental daylight, shadow casting, reflections, and atmospheric depth are synthesized automatically.

### 2.3 Core Project Objectives
1. **Automated Sketch-to-Render Pipeline:** Develop an end-to-end automated platform that translates raw facade line sketches into presentation-grade exterior visualizations.
2. **Rigorous Structural Consistency:** Enforce spatial conditioning via ControlNet Canny edge guidance so the output does not deviate from the structural blueprints.
3. **Architectural Style Disentanglement:** Provide distinct, curated style directions (Modern Minimalist, Luxury Contemporary, Traditional Indian, Industrial, Scandinavian) through specialized positive and negative prompt engineering.
4. **Accessible Hybrid System Architecture:** Overcome consumer laptop hardware constraints (lack of high-end dedicated GPUs) by establishing a hybrid architecture where a lightweight local web gateway orchestrates inference across a secure, remote cloud GPU worker (Google Colab T4).
5. **Zero-Hallucination Production Standards:** Implement a strict no-fallback policy: the application displays generated results only when a genuine deep-learning inference run completes successfully on the remote GPU, never substituting preloaded demonstration images.

### 2.4 Target Users & Practical Utility
- **Practicing Architects & Design Studios:** Accelerate concept presentations from days to seconds during initial client alignment meetings.
- **Civil Engineering & Architecture Students:** Visualize design thesis projects and portfolio work without requiring expensive 3D rendering workstations.
- **Real Estate Developers & Marketing Teams:** Generate rapid conceptual variations of residential or commercial facades for market testing before detailed engineering drawings are commissioned.
- **Urban Planners & Municipal Reviewers:** Rapidly assess how a proposed building massing and material palette will harmonize with existing urban streetscapes.

### 2.5 The Non-Technical Analogy (For Viva Panels)
> *"Imagine an architect gives an artist a detailed black-and-white coloring book page of a building. A standard AI model like DALL-E or Midjourney behaves like an artist who looks at the page, throws it in the trash, and paints a completely different building that just looks vaguely similar.*  
> *FacadeForge AI, through ControlNet, acts like a master architectural illustrator who keeps the exact black-and-white coloring book lines pinned to the drafting table, and paints inside those exact structural lines using photorealistic oil paints—applying genuine Italian travertine stone, cedar timber battens, tinted reflective glass, and natural afternoon sun.*  
> *The building lines never move; only the materials, light, and realism are brought to life."*

---

## 3. Complete Technology Stack & Selection Rationale

Every component in FacadeForge AI was selected based on verifiable technical necessity, performance efficiency, and hardware compatibility.

```
+---------------------------------------------------------------------------------------+
|                                    FACADEFORGE AI                                     |
+---------------------------------------------------------------------------------------+
|  FRONTEND LAYER        |  HTML5 (Semantic Layout)                                     |
|                        |  CSS3 (Custom Architectural Editorial Design System)         |
|                        |  Vanilla ES6 JavaScript (Async DOM, Polling Lifecycle)       |
+------------------------+--------------------------------------------------------------+
|  LOCAL GATEWAY LAYER   |  Python 3.10 Runtime Environment                             |
|                        |  Flask 3.0.x (WSGI Routing & Background Worker Threads)      |
|                        |  Pillow 10.x & NumPy (Upload Validation, Sanitization)       |
|                        |  HMAC-SHA256 Token Auth & Subprocess Management              |
+------------------------+--------------------------------------------------------------+
|  SECURE TRANSPORT      |  Cloudflare Quick Tunnel (cloudflared daemon, QUIC / TLS 1.3)|
|                        |  Reverse Proxy & Outbound Tunneling Protocol                 |
+------------------------+--------------------------------------------------------------+
|  REMOTE GPU LAYER      |  Google Colab Runtime (Linux / CUDA 12.x)                    |
|  (INFERENCE WORKER)    |  Nvidia Tesla T4 GPU (16 GB GDDR6 VRAM, Turing Architecture) |
|                        |  PyTorch 2.x (torch.float16 Half-Precision Execution)        |
|                        |  Hugging Face Diffusers (StableDiffusionControlNetPipeline)  |
|                        |  Hugging Face Transformers (CLIP ViT-L/14 Text Encoder)      |
|                        |  DPMSolverMultistepScheduler (Fast 20-Step Diffusion ODE)    |
|                        |  OpenCV (cv2.Canny, cv2.cvtColor)                            |
+---------------------------------------------------------------------------------------+
```

### 3.1 Deep Learning & Computer Vision Stack

#### 1. Python 3.10
- **Role:** Primary development language for both the local gateway (`backend/app.py`, `backend/inference/remote_client.py`) and remote inference server (`backend/inference/colab_server.py`).
- **Why Selected:** Native support across PyTorch, Hugging Face Diffusers, OpenCV, and Flask; excellent handling of binary image streams, thread locks, and asynchronous HTTP clients.

#### 2. PyTorch 2.x (`torch`)
- **Role:** Deep learning computational backbone powering tensor operations, GPU memory allocation, and neural network inference.
- **Why Selected:** PyTorch is the industry-standard framework for generative AI models. On the Colab Tesla T4 GPU, PyTorch executes tensor computations in half-precision (`torch.float16`), slashing VRAM utilization by 50% (from ~14 GB to ~7 GB) without perceptible degradation in visual quality.

#### 3. Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`)
- **Role:** Base text-to-image latent diffusion model responsible for synthesizing realistic architectural materials, lighting, depth, and textures.
- **Why Selected:** 
  - SD 1.5 operates at a native resolution of $512 \\times 512$ pixels, matching our input preprocessing pipeline.
  - Its latent space compression factor ($8\\times$, converting $512 \\times 512 \\times 3$ RGB images into $64 \\times 64 \\times 4$ latents) enables rapid diffusion iteration within the VRAM budget of a single Tesla T4 GPU.
  - SD 1.5 possesses a mature, highly compatible ecosystem with ControlNet checkpoints (`lllyasviel/sd-controlnet-canny`), whereas larger models (e.g., SDXL) require >16 GB VRAM and significantly higher inference latency.

#### 4. ControlNet Canny (`lllyasviel/sd-controlnet-canny`)
- **Role:** Spatial structure-guidance neural network that conditions the Stable Diffusion UNet on edge maps extracted from the user's sketch.
- **Why Selected:** Standard diffusion models rely solely on text prompts, which lack coordinate awareness. ControlNet duplicates the 12 encoding blocks and middle block of the SD UNet, injecting spatial edge priors into the locked base model via zero-convolution layers. Canny conditioning guarantees that floor lines, window boundaries, and rooflines are maintained.

#### 5. Hugging Face `diffusers`
- **Role:** Orchestration library for the generative diffusion pipeline (`StableDiffusionControlNetPipeline`), handling the denoising loop, model offloading, memory slicing, and scheduler integration.
- **Why Selected:** Provides standardized, battle-tested abstractions for multi-model diffusion pipelines, attention slicing (`pipe.enable_attention_slicing()`), and VAE slicing (`pipe.enable_vae_slicing()`).

#### 6. DPMSolverMultistepScheduler (`DPMSolverMultistepScheduler`)
- **Role:** Numerical ODE solver algorithm managing the reverse diffusion denoising trajectory.
- **Why Selected:** Replaces the legacy 50-step PNDM or DDIM schedulers. DPMSolver++ is a dedicated high-order solver for diffusion ordinary differential equations that converges to photorealistic fidelity in only **20 inference steps**, reducing generation time on a Tesla T4 GPU from ~45 seconds down to ~8–12 seconds.

#### 7. OpenCV (`opencv-python` / `cv2`)
- **Role:** Real-time computer vision processing for Canny edge detection (`cv2.Canny`) and color-space transformations (`cv2.cvtColor`).
- **Why Selected:** Highly optimized C++ computer vision primitives executing in milliseconds. Canny edge detection with hysteresis thresholds $T_{low} = 50$ and $T_{high} = 150$ produces clean, single-pixel architectural structural contours free of pencil grain and scanner noise.

#### 8. Pillow (PIL)
- **Role:** Image validation, canvas containment (`ImageOps.contain`), aspect-ratio-preserving padding, and format transcoding.
- **Why Selected:** Robust image handling with built-in defenses against corrupted files, decompression bomb attacks (DoS attacks via massive pixel arrays), and seamless byte-stream serialization (`io.BytesIO`).

### 3.2 Web Gateway & Infrastructure Stack

#### 1. Flask 3.0.x
- **Role:** Lightweight WSGI web framework serving both the local user-facing gateway (`backend/app.py` on port 5000) and the remote Colab API server (`colab_server.py` on port 7860).
- **Why Selected:** Minimal overhead, transparent thread lifecycle, zero boilerplate. Flask allows FacadeForge AI to run an asynchronous job queue via Python's standard `threading` library without requiring heavy distributed message brokers like Celery or Redis.

#### 2. Cloudflare Quick Tunnel (`cloudflared`)
- **Role:** Ephemeral, secure reverse-proxy tunneling exposing the private Colab inference server (port 7860) to the public internet via HTTPS (`*.trycloudflare.com`).
- **Why Selected:** 
  - Zero cost, requires no account registration, and requires no API keys.
  - Establishes an outbound TLS session from Colab to Cloudflare edge nodes, bypassing Colab's strict inbound firewall and avoiding the need for public IP addresses or router port forwarding.
  - Outperforms legacy alternatives (e.g., ngrok session limits or Gradio sharing tunnel timeouts).

#### 3. Custom Vanilla Frontend (HTML5, Modern CSS3, ES6 JavaScript)
- **Role:** Client-side presentation layer offering an architectural editorial aesthetic, real-time image preview, drag-and-drop file ingestion, asynchronous job polling, and side-by-side comparison rendering.
- **Why Selected:** Zero framework bloat (no React, Angular, or Vue node_modules dependencies). Fast page loads, sub-millisecond DOM updates, and complete design customization matching high-end architectural publications (DM Sans typography, Playfair Display serif headings, warm editorial palette).

---

## 4. System Architecture & Data Flow

FacadeForge AI utilizes a **Distributed Three-Tier Hybrid Architecture**. This architecture cleanly separates the user presentation, request orchestration, and heavy tensor computation into distinct execution environments.

```mermaid
flowchart TD
    subgraph Client["Tier 1: Client Presentation (Browser)"]
        UI["Modern Web Interface\n(HTML5 / CSS3 / ES6)"]
        DZ["Drop Zone & Sketch Preview"]
        POLL["Async Polling Engine\n(interval: 1200ms)"]
    end

    subgraph Gateway["Tier 2: Local Application Gateway (Flask)"]
        APP["Flask Web Server\n(127.0.0.1:5000)"]
        VAL["Image Validator\n(Pillow: <8MB, <40M px)"]
        JQ["In-Memory Thread-Safe Job Queue\n(_jobs dictionary + Lock)"]
        WT["Daemon Worker Thread\n(_run_job)"]
        DIR["Generated Storage\n(backend/generated/{uuid}.png)"]
    end

    subgraph Cloud["Tier 3: Remote GPU Worker (Google Colab)"]
        CFT["Cloudflare Quick Tunnel\n(cloudflared daemon: trycloudflare.com)"]
        CS["Colab Flask Server\n(0.0.0.0:7860)"]
        AUTH["HMAC-SHA256 Token Auth\n(compare_digest)"]
        PREP["Canny Preprocessor\n(ImageOps.contain + cv2.Canny 50/150)"]
        PIPE["Stable Diffusion v1.5 +\nControlNet Canny Pipeline"]
        GPU["Nvidia Tesla T4 GPU\n(16GB VRAM, float16)"]
    end

    UI -->|1. Drag & Drop Sketch| DZ
    DZ -->|2. POST /api/generate| APP
    APP -->|3. Validate File & Format| VAL
    VAL -->|4. Register Job UUID| JQ
    APP -->|5. Return HTTP 202 Accepted| UI
    APP -->|6. Spawn Background Task| WT
    UI -->|7. Poll GET /api/jobs/{id}| POLL
    POLL -->|8. Query State| JQ

    WT -->|9. Authenticated HTTPS Request| CFT
    CFT -->|10. Proxy Forward to Port 7860| CS
    CS -->|11. Bearer Token Check| AUTH
    AUTH -->|12. Preprocess to 512x512 Canvas| PREP
    PREP -->|13. Feed Edge Map & Style Prompt| PIPE
    PIPE -->|14. 20 Denoising Steps (DPMSolver++)| GPU
    GPU -->|15. Return Synthesized Facade Image| PIPE
    PIPE -->|16. Stream image/png binary| CS
    CS -->|17. HTTPS Response| CFT
    CFT -->|18. Return Binary Payload| WT
    WT -->|19. Write to Disk| DIR
    WT -->|20. Update State = 'complete'| JQ

    POLL -->|21. Detect 'complete' Status| UI
    UI -->|22. Fetch GET /api/jobs/{id}/image| APP
    APP -->|23. Stream PNG from Disk| UI
    UI -->|24. Display Side-by-Side Comparison| UI
```

### 4.1 Tier Breakdown and Responsibilities

#### Tier 1: Client Presentation Layer (Browser)
- Runs inside the user's web browser (`/` for marketing showcase, `/workspace` for generation studio).
- Handles user interactions: drag-and-drop sketch upload, file format filtering (`.png`, `.jpg`, `.jpeg`), and instantaneous client-side object URL previewing (`URL.createObjectURL`).
- Houses the asynchronous polling worker: when a generation job begins, the frontend initiates a recurring `pollJob()` loop that queries `/api/jobs/<job_id>` every 1,200 milliseconds until a terminal state (`complete` or `error`) is reached.
- Renders the interactive comparison grid: displays the original architectural sketch alongside the AI-generated visualization, accompanied by download triggers and style variation controls.

#### Tier 2: Local Application Gateway (Flask on Windows)
- Runs locally at `http://127.0.0.1:5000` via `backend/app.py`.
- **Decoupling Role:** Ensures the web interface remains 100% responsive, fast, and stable even if remote AI inference takes 15–30 seconds.
- **Security & Validation:** Enforces strict upload constraints: file size capped at 8 MB (`MAX_CONTENT_LENGTH`), pixel dimension ceiling of 40,000,000 pixels (preventing decompression bombs), and image format verification via Pillow (`image.verify()`).
- **Asynchronous Concurrency:** Implements a thread-safe in-memory job registry (`_jobs: dict`) protected by a mutual exclusion lock (`_jobs_lock = threading.Lock()`). Upon receiving an upload, it dispatches the network request inside a background daemon thread (`threading.Thread(target=_run_job, ...)`), returning an immediate `HTTP 202 Accepted` response containing a unique UUID4 identifier.
- **Asset Serving & Persistence:** Stores completed renders on disk in `backend/generated/{job_id}.png` and serves them via `/api/jobs/<job_id>/image` and `/api/jobs/<job_id>/download`.

#### Tier 3: Remote Inference Layer (Google Colab GPU + Cloudflare)
- Runs inside a cloud-hosted Linux runtime equipped with an Nvidia Tesla T4 GPU.
- **Daemon Ingress:** The `cloudflared` daemon creates an outbound encrypted tunnel connecting Colab port 7860 to a public Cloudflare edge URL (`https://<random>.trycloudflare.com`).
- **Authentication Gateway:** Colab runs `colab_server.py`. Every inbound request must present a Bearer token matching `FACADEFORGE_REMOTE_TOKEN`. Validation is performed using `hmac.compare_digest` to prevent side-channel timing attacks.
- **Model Residency & Pipeline Cache:** The model pipeline (`StableDiffusionControlNetPipeline`) is loaded directly into the T4 GPU's VRAM. When invoked from the Colab notebook via `colab_server.set_pipeline(pipe)`, the server registers the active in-memory pipeline, preventing redundant weight allocations or GPU Out-Of-Memory (OOM) crashes.
- **Inference Execution:** Preprocesses the incoming sketch into a 512×512 white-padded Canny edge map, applies negative prompts and style tokens, executes 20 DPM-Solver++ denoising steps in `torch.float16`, and streams the resulting PNG binary back through the tunnel.

---

## 5. Complete Working Workflow — Step-by-Step Execution Lifecycle

The complete execution lifecycle of FacadeForge AI proceeds across twelve rigorous stages:

```mermaid
sequenceDiagram
    autonumber
    actor User as Architect / User
    participant Browser as Frontend (workspace.js)
    participant Flask as Local Gateway (app.py)
    participant Worker as Background Thread (_run_job)
    participant Tunnel as Cloudflare Quick Tunnel
    participant Colab as Colab Server (colab_server.py)
    participant GPU as PyTorch / T4 GPU Pipeline

    User->>Browser: Selects facade sketch & architectural style
    Browser->>Browser: Validates file type, generates preview URL
    User->>Browser: Clicks "Generate facade"
    Browser->>Flask: POST /api/generate (multipart/form-data: sketch, style)
    Flask->>Flask: Validates image size (<8MB) & pixel bounds (<40M)
    Flask->>Flask: Generates UUID job_id, registers state="queued"
    Flask-->>Browser: HTTP 202 Accepted {id: "uuid", state: "queued"}
    Flask->>Worker: Spawns background thread (_run_job)

    loop Polling Loop (Every 1200ms)
        Browser->>Flask: GET /api/jobs/{job_id}
        Flask-->>Browser: HTTP 200 OK {state: "running", message: "..."}
    end

    Worker->>Tunnel: Authenticated POST /generate (Bearer Token)
    Tunnel->>Colab: Forward request to 127.0.0.1:7860
    Colab->>Colab: Verify HMAC token against FACADEFORGE_REMOTE_TOKEN
    Colab->>Colab: Resize with ImageOps.contain & pad to 512x512 canvas
    Colab->>Colab: Convert to grayscale & run cv2.Canny(50, 150)
    Colab->>GPU: Execute pipe(prompt, image=canny_map, steps=20, guidance=7.5)
    GPU->>GPU: 20 Denoising Steps via DPMSolverMultistepScheduler (float16)
    GPU-->>Colab: Return synthesized PIL Image
    Colab-->>Tunnel: HTTP 200 OK (image/png binary stream)
    Tunnel-->>Worker: Stream image bytes over secure TLS
    Worker->>Worker: Verify PNG format header (\x89PNG\r\n\x1a\n)
    Worker->>Worker: Save to backend/generated/{job_id}.png
    Worker->>Worker: Update job state="complete", result_url="/api/jobs/{id}/image"

    Browser->>Flask: GET /api/jobs/{job_id}
    Flask-->>Browser: HTTP 200 OK {state: "complete", result_url: "..."}
    Browser->>Browser: Hide empty-state canvas immediately
    Browser->>Flask: GET /api/jobs/{job_id}/image
    Flask-->>Browser: Stream PNG image binary
    Browser->>Browser: Render Side-by-Side Comparison & Enable Download
    User->>Browser: Clicks "Download image" to save render
```

### 5.1 Detailed Stage Analysis

#### Stage 1: Application Navigation & Initial Health Assessment
The user accesses `http://127.0.0.1:5000/workspace`. On page initialization, `workspace.js` immediately dispatches an asynchronous `GET /api/status` request to the local Flask server. Flask invokes `check_remote_health()` in `remote_client.py`, which executes an 8-second timeout probe to `https://<tunnel-url>/health` with the shared bearer token. If Colab responds with `{"available": true, "cuda_available": true}`, the UI displays a green status indicator: *"Colab is connected and the SD 1.5 + ControlNet pipeline is loaded."*

#### Stage 2: Sketch Upload & Client Validation
The architect drops an architectural drawing into the `.drop-zone`. The frontend `fileInput` listener intercepts the file and performs client-side validation:
- File type must strictly be `image/png` or `image/jpeg`.
- File size must not exceed 8 MB ($8 \\times 1024 \\times 1024$ bytes).
If valid, the browser invokes `URL.createObjectURL(file)`, instantly rendering the sketch in `#sketch-preview` without waiting for server round-trips.

#### Stage 3: Architectural Style Selection
The user chooses one of five architectural directions from `#style-select`:
1. **Modern Minimalist:** Clean geometry, smooth white concrete, floor-to-ceiling glass, restrained landscaping.
2. **Luxury Contemporary:** Natural stone cladding, warm cedar timber battens, bronze accents, premium exterior accent lighting.
3. **Traditional Indian:** Warm Dholpur sandstone, intricate jaali shading screens, traditional verandahs, climate-responsive overhangs.
4. **Industrial Modern:** Exposed board-formed concrete, dark structural steel framing, textured brick masonry, crittall factory-style glazing.
5. **Scandinavian:** Pale Nordic timber, soft light stone, pitched rooflines, soft diffuse overcast daylight.

#### Stage 4: Job Dispatch & Asynchronous Ingestion
Clicking **Generate facade** sends a `POST /api/generate` multipart request containing the binary sketch and style key. Flask validates the payload:
- Verifies that file bytes do not exceed `MAX_CONTENT_LENGTH`.
- Opens the byte array using `PIL.Image.open()` and validates the header with `image.verify()`.
- Calculates total pixel count ($width \\times height$). If pixel area exceeds 40,000,000 pixels (~40 Megapixels), it rejects the upload with `HTTP 413` to prevent memory exhaustion.
- Generates a cryptographically random UUID4 (`uuid.uuid4().hex`).
- Inserts `_jobs[job_id] = {"id": job_id, "state": "queued"}` into the thread-safe registry.
- Spawns a daemon thread `threading.Thread(target=_run_job, args=(job_id, source, style))` and returns an instant `HTTP 202 Accepted` response.

#### Stage 5: Gateway Remote Dispatch
Inside `_run_job`, the worker thread calls `generate_remote_image(source, style)`. It wraps the binary data into a standard multipart payload (`_multipart()`) with boundaries and sets a configurable timeout (default 300 seconds, capped at 900 seconds via `FACADEFORGE_REMOTE_TIMEOUT_SECONDS`). It sends an authorized HTTP POST request to `https://<tunnel-url>/generate`.

#### Stage 6: Cloudflare Edge Routing & Authorization
The request arrives at the Cloudflare edge, traverses the encrypted Argo tunnel, and reaches `colab_server.py` on Colab localhost port 7860. The server's `_authorized()` function extracts the `Authorization: Bearer <token>` header and evaluates it using `hmac.compare_digest`. Unauthorized requests are instantly rejected with `HTTP 401`.

#### Stage 7: Deterministic Sketch Preprocessing (`_prepare_sketch`)
To ensure structural alignment with Stable Diffusion's 512×512 neural grid:
1. **Aspect Ratio Preservation:** The sketch is converted to RGB and resized using `PIL.ImageOps.contain(image, (512, 512))`. This shrinks the largest dimension to 512 pixels while preserving the exact original aspect ratio without distortion or stretching.
2. **Canvas Centering:** A blank 512×512 pure white (`#FFFFFF`) canvas is created. The resized sketch is pasted dead-center at coordinates:
   $$\text{x} = \frac{512 - \text{width}}{2}, \quad \text{y} = \frac{512 - \text{height}}{2}$$
3. **Grayscale Conversion:** The RGB canvas is converted to a single-channel 8-bit unsigned integer array using OpenCV:
   $$\text{gray} = \text{cv2.cvtColor}(\text{canvas}, \text{cv2.COLOR\_RGB2GRAY})$$
4. **Canny Edge Extraction:** Edge contours are computed via `cv2.Canny(gray, 50, 150)`.
5. **3-Channel Control Map:** The resulting binary edge array (where edges are 255/white and backgrounds are 0/black) is formatted back into a 3-channel RGB PIL Image to feed ControlNet's conditioning tensor.

#### Stage 8: Prompt Synthesis & Negative Prompt Engineering
The server constructs the complete conditioning prompt by combining the immutable architectural base prompt with the style tokens:
```python
prompt = (
    "Photorealistic architectural exterior visualization of the exact building shown in the reference sketch, "
    "realistic materials, realistic daylight, detailed textures, professional architectural photography, high quality, sharp focus. "
    "Preserve the original building geometry, floor count, roofline, window positions, door positions, balconies and proportions. "
    f"Architectural direction: {STYLES[style]}."
)
```
Simultaneously, a strict negative prompt suppresses common generative distortions:
```python
negative_prompt = (
    "cartoon, illustration, painting, distorted architecture, extra floors, missing floors, extra windows, "
    "missing windows, deformed building, unrealistic geometry, blurry, low quality, watermark, text, people"
)
```

#### Stage 9: GPU Tensor Diffusion (DPM-Solver++ on T4)
Inside an inference lock (`with _inference_lock:` preventing simultaneous GPU access), the model runs:
```python
generator = torch.Generator(device="cpu").manual_seed(secrets.randbelow(2**31))
result = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=control_image,
    width=512,
    height=512,
    num_inference_steps=20,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,
    generator=generator,
)
```
In 20 rapid denoising steps, the latent representation is steered by the Canny edge tensor and text embeddings, generating a 512×512 photorealistic architectural facade render.

#### Stage 10: In-Memory Serialization & Stream Transmission
The resulting PIL image is serialized into an in-memory byte buffer (`io.BytesIO()`) as a lossless PNG. Colab returns this binary buffer via `send_file(image_buffer, mimetype="image/png")` with `max_age=0` (preventing caching).

#### Stage 11: Gateway Ingestion & Job Completion
The local gateway thread receives the byte stream:
- Asserts that HTTP status is 200.
- Verifies that the payload begins with the 8-byte PNG magic signature: `\x89PNG\r\n\x1a\n`.
- Opens the byte stream with Pillow and runs `verify()` to confirm that no packets were dropped.
- Writes the image to `backend/generated/{job_id}.png`.
- Updates the job dictionary under lock:
  ```python
  _jobs[job_id].update(
      state="complete",
      result_url=f"/api/jobs/{job_id}/image",
      download_url=f"/api/jobs/{job_id}/download",
  )
  ```

#### Stage 12: Frontend Rendering & Result Presentation
The client's active polling loop hits `/api/jobs/<job_id>`, receives `state: "complete"`, and stops polling:
- **Instant Empty-State Removal:** The `.result-empty` container (blueprint graphic and "YOUR CANVAS IS READY" text) receives `hidden="true"`. Because of our explicit CSS rule (`[hidden], .result-empty[hidden] { display: none !important; }`), it is instantly removed from the DOM layout.
- **Direct Image Placement:** The comparison grid (`#result-content`) is unhidden directly beneath the **"Visualization"** heading.
- `#result-original` displays the input sketch; `#result-generated` displays the synthesized render fetched from `/api/jobs/<job_id>/image`.
- The **Download image** button is bound to `/api/jobs/<job_id>/download`, allowing the architect to save the full-resolution render directly to their workstation.
"""
