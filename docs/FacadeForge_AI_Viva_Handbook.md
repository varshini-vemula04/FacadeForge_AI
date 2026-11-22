# FacadeForge AI — Complete Project Documentation & Viva Preparation Handbook
**Comprehensive Technical Architecture, Deep Learning Pipeline, and Academic Examination Defense Guide**

---

### Document Overview & Metadata
- **Project Title:** FacadeForge AI: Sketch-to-Realistic Architectural Building Renderer
- **Project Domain:** Generative Artificial Intelligence, Computer Vision, Architectural Visualization
- **Primary AI Methodology:** Structure-Guided Image-to-Image Translation (Latent Diffusion + ControlNet Canny)
- **Base Pretrained Models:** Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`) & ControlNet Canny (`lllyasviel/sd-controlnet-canny`)
- **System Architecture:** Distributed Hybrid Cloud Architecture (Local Python/Flask Gateway + Remote Google Colab GPU Worker via Cloudflare Quick Tunnel)
- **Primary Programming Languages:** Python 3.10, Vanilla ES6 JavaScript, HTML5, Modern CSS3
- **Primary Frameworks & Libraries:** PyTorch 2.x, Hugging Face Diffusers, Transformers, OpenCV, Pillow, Flask 3.0.x
- **Target Audience:** Project Evaluation Panelists, External Examiners, Faculty Advisors, and Engineering Students preparing for Final Capstone / Viva Voce Examinations
- **Document Version:** 2.0.0 (Comprehensive Academic Edition)
- **Last Verified Against Codebase:** November 2026

---

## Table of Contents
1. [Executive Summary & Project Identification](#1-executive-summary--project-identification)
2. [Project Introduction, Problem Statement, & Core Objectives](#2-project-introduction-problem-statement--core-objectives)
3. [Complete Technology Stack & Selection Rationale](#3-complete-technology-stack--selection-rationale)
4. [System Architecture & Data Flow](#4-system-architecture--data-flow)
5. [Complete Working Workflow — Step-by-Step Execution Lifecycle](#5-complete-working-workflow--step-by-step-execution-lifecycle)
6. [AI & Machine Learning Foundations (The Viva Core)](#6-ai--machine-learning-foundations-the-viva-core)
   - 6.1 [Diffusion Models & Mathematical Intuition](#61-diffusion-models--mathematical-intuition)
   - 6.2 [Latent Diffusion Models (LDM) & Architectural Mechanics](#62-latent-diffusion-models-ldm--architectural-mechanics)
   - 6.3 [ControlNet: Structural Conditioning via Zero Convolutions](#63-controlnet-structural-conditioning-via-zero-convolutions)
   - 6.4 [Canny Edge Detection: Mathematical Foundations & Stages](#64-canny-edge-detection-mathematical-foundations--stages)
   - 6.5 [Integrated Pipeline: Scheduler, Guidance, and Optimizations](#65-integrated-pipeline-scheduler-guidance-and-optimizations)
7. [Detailed Codebase & Module Walkthrough](#7-detailed-codebase--module-walkthrough)
   - 7.1 [Local Backend Gateway (`backend/app.py`)](#71-local-backend-gateway-backendapppy)
   - 7.2 [Remote Colab GPU Inference Server (`backend/inference/colab_server.py`)](#72-remote-colab-gpu-inference-server-backendinferencecolab_serverpy)
   - 7.3 [Authenticated Remote HTTP Client (`backend/inference/remote_client.py`)](#73-authenticated-remote-http-client-backendinferenceremote_clientpy)
   - 7.4 [Local Inference Fallback Adapter (`backend/inference/pipeline.py`)](#74-local-inference-fallback-adapter-backendinferencepipelinepy)
   - 7.5 [Interactive Frontend Web Application (`frontend/`)](#75-interactive-frontend-web-application-frontend)
8. [Remote Inference, Cloudflare Tunnels, & Deployment Engineering](#8-remote-inference-cloudflare-tunnels--deployment-engineering)
9. [Installation, Setup, & Step-by-Step Execution Guide](#9-installation-setup--step-by-step-execution-guide)
10. [Key Technical Design Decisions & Justifications](#10-key-technical-design-decisions--justifications)
11. [Real Technical Challenges Overcome & Documented Limitations](#11-real-technical-challenges-overcome--documented-limitations)
12. [Future Roadmap & Research Extensions](#12-future-roadmap--research-extensions)
13. [Individual Module Breakdown & Viva Defense Scripts](#13-individual-module-breakdown--viva-defense-scripts)
14. [Comprehensive Viva Voce Question Bank (100+ Questions & Answers across Categories A–P)](#14-comprehensive-viva-voce-question-bank)
15. [Tough Cross-Questioning & Hostile Examiner Scenarios](#15-tough-cross-questioning--hostile-examiner-scenarios)
16. [Quick Revision Sheet & Last-Minute Cheat Sheet](#16-quick-revision-sheet--last-minute-cheat-sheet)

---


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
  - SD 1.5 operates at a native resolution of $512 \times 512$ pixels, matching our input preprocessing pipeline.
  - Its latent space compression factor ($8\times$, converting $512 \times 512 \times 3$ RGB images into $64 \times 64 \times 4$ latents) enables rapid diffusion iteration within the VRAM budget of a single Tesla T4 GPU.
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
        UI["Modern Web Interface
(HTML5 / CSS3 / ES6)"]
        DZ["Drop Zone & Sketch Preview"]
        POLL["Async Polling Engine
(interval: 1200ms)"]
    end

    subgraph Gateway["Tier 2: Local Application Gateway (Flask)"]
        APP["Flask Web Server
(127.0.0.1:5000)"]
        VAL["Image Validator
(Pillow: <8MB, <40M px)"]
        JQ["In-Memory Thread-Safe Job Queue
(_jobs dictionary + Lock)"]
        WT["Daemon Worker Thread
(_run_job)"]
        DIR["Generated Storage
(backend/generated/{uuid}.png)"]
    end

    subgraph Cloud["Tier 3: Remote GPU Worker (Google Colab)"]
        CFT["Cloudflare Quick Tunnel
(cloudflared daemon: trycloudflare.com)"]
        CS["Colab Flask Server
(0.0.0.0:7860)"]
        AUTH["HMAC-SHA256 Token Auth
(compare_digest)"]
        PREP["Canny Preprocessor
(ImageOps.contain + cv2.Canny 50/150)"]
        PIPE["Stable Diffusion v1.5 +
ControlNet Canny Pipeline"]
        GPU["Nvidia Tesla T4 GPU
(16GB VRAM, float16)"]
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
    Worker->>Worker: Verify PNG format header (PNG

)
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
- File size must not exceed 8 MB ($8 \times 1024 \times 1024$ bytes).
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
- Calculates total pixel count ($width \times height$). If pixel area exceeds 40,000,000 pixels (~40 Megapixels), it rejects the upload with `HTTP 413` to prevent memory exhaustion.
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
   $$	ext{x} = rac{512 - 	ext{width}}{2}, \quad 	ext{y} = rac{512 - 	ext{height}}{2}$$
3. **Grayscale Conversion:** The RGB canvas is converted to a single-channel 8-bit unsigned integer array using OpenCV:
   $$	ext{gray} = 	ext{cv2.cvtColor}(	ext{canvas}, 	ext{cv2.COLOR\_RGB2GRAY})$$
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
- Verifies that the payload begins with the 8-byte PNG magic signature: `PNG

`.
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


## 6. AI & Machine Learning Foundations (The Viva Core)

This section contains the fundamental theoretical and mathematical knowledge required to explain and defend FacadeForge AI in front of academic examiners.

```
+--------------------------------------------------------------------------------------------------------+
|                                  FACADEFORGE AI GENERATIVE PIPELINE                                    |
+--------------------------------------------------------------------------------------------------------+
|  INPUT SKETCH                                                                                          |
|       |                                                                                                |
|       v                                                                                                |
|  [ PREPROCESSING ] ----> Pillow contain(512,512) on white canvas -> cv2.Canny(50, 150)                 |
|                                |                                                                       |
|                                v                                                                       |
|                       [ 512x512 Canny Edge Map ]                                                       |
|                                |                                                                       |
|  TEXT PROMPT                   v                                                                       |
|  ("Photorealistic...") -> [ CLIP Text Encoder ] -> Text Embeddings (77x768)                            |
|                                |                               |                                       |
|                                v                               v                                       |
|  RANDOM LATENT NOISE ---> [ ControlNet (Locked SD + Trainable Copy + Zero Convolutions) ]              |
|  z_T ~ N(0, I) (64x64x4)       |                               |                                       |
|       |                        +---------------+---------------+                                       |
|       |                                        | (Injected Residuals)                                  |
|       v                                        v                                                       |
|  [ LATENT REVERSE DIFFUSION LOOP ] <-----------+                                                       |
|  - Model: Time-Conditioned UNet (860M params)                                                          |
|  - Scheduler: DPMSolverMultistepScheduler (20 steps)                                                  |
|  - Classifier-Free Guidance (CFG scale = 7.5)                                                          |
|  - Precision: torch.float16 on Nvidia Tesla T4 GPU                                                     |
|       |                                                                                                |
|       v (Denoised Latent z_0, shape: 1x4x64x64)                                                        |
|  [ VAE DECODER (AutoencoderKL) ] (8x Spatial Upscaling)                                                |
|       |                                                                                                |
|       v                                                                                                |
|  OUTPUT IMAGE: 512x512 Photorealistic Architectural Facade Render                                     |
+--------------------------------------------------------------------------------------------------------+
```

### 6.1 Diffusion Models & Mathematical Intuition

#### What is a Diffusion Model?
A **diffusion model** is a class of generative models inspired by non-equilibrium statistical thermodynamics. Unlike Generative Adversarial Networks (GANs), which learn a direct generator mapping against an adversarial discriminator, or Variational Autoencoders (VAEs), which optimize a lower bound on data likelihood, diffusion models define image generation as an **iterative denoising process**.

A diffusion model operates through two fundamental processes:
1. **The Forward Diffusion Process ($q$):** Systematically destroys structure by incrementally adding Gaussian noise to an image across discrete time steps $t \in [1, T]$ until the image becomes pure, isotropic Gaussian noise.
2. **The Reverse Diffusion Process ($p_	heta$):** A learned neural network (UNet) predicts and subtracts the noise at each step, recovering the clean data distribution from pure noise.

#### Forward Diffusion (The Markov Chain of Noise)
Given an initial data point $x_0 \sim q(x)$, the forward process adds Gaussian noise according to a predefined variance schedule $eta_1, eta_2, \dots, eta_T$:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - eta_t} x_{t-1}, eta_t \mathbf{I})$$

A key mathematical property of the Gaussian forward process is that **any intermediate state $x_t$ can be sampled in closed form directly from $x_0$ without evaluating all $t-1$ preceding steps**. Defining $lpha_t = 1 - eta_t$ and $ar{lpha}_t = \prod_{s=1}^t lpha_s$:
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{ar{lpha}_t} x_0, (1 - ar{lpha}_t) \mathbf{I})$$
$$x_t = \sqrt{ar{lpha}_t} x_0 + \sqrt{1 - ar{lpha}_t} \epsilon, \quad 	ext{where } \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
As $t 	o T$, $ar{lpha}_t 	o 0$, and $x_T$ becomes virtually indistinguishable from pure Gaussian noise $\mathcal{N}(0, \mathbf{I})$.

#### Reverse Diffusion (Learned Denoising)
The reverse trajectory $q(x_{t-1} | x_t)$ is intractable because it depends on the entire data distribution $q(x_0)$. Therefore, we train a neural network parameterized by weights $	heta$ to approximate the reverse distribution:
$$p_	heta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_	heta(x_t, t), \Sigma_	heta(x_t, t))$$

In practice, rather than directly predicting the mean $\mu_	heta$, the model is parameterized to predict the **exact noise vector $\epsilon_	heta(x_t, t)$** that was added to $x_0$. The simplified training objective (derived by Ho et al., 2020) is:
$$\mathcal{L}_{simple}(	heta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_	heta(x_t, t) \|^2 ight]$$

---

### 6.2 Latent Diffusion Models (LDM) & Architectural Mechanics

#### The Computational Bottleneck of Pixel-Space Diffusion
Standard Denoising Diffusion Probabilistic Models (DDPM) operate directly in RGB pixel space ($512 	imes 512 	imes 3 = 786,432$ variables per step). Running multi-step neural network convolutions across 786k continuous pixel variables is computationally prohibitive, requiring multi-GPU server clusters and several minutes per image.

#### The LDM Breakthrough: Perceptual vs. Semantic Compression
Rombach et al. (2022) observed that digital images contain two forms of information:
1. **Imperceptible high-frequency details** (subtle pixel variations, compression noise).
2. **Semantic conceptual information** (spatial geometry, objects, lighting, architectural elements).

**Latent Diffusion Models (LDM)**—the foundation of Stable Diffusion—decouple image generation into two specialized stages:
1. **Perceptual Compression (The VAE):** A pretrained Variational Autoencoder (AutoencoderKL) compresses the high-resolution RGB image $x \in \mathbb{R}^{3 	imes 512 	imes 512}$ into a lower-dimensional continuous latent space $z = \mathcal{E}(x) \in \mathbb{R}^{4 	imes 64 	imes 64}$.
   $$	ext{Compression Factor: } f = rac{512}{64} = 8$$
   $$	ext{Dimensionality Reduction: } rac{3 	imes 512 	imes 512}{4 	imes 64 	imes 64} = rac{786,432}{16,384} = 48	imes 	ext{ fewer tensor elements}$$
2. **Generative Modeling (The Latent UNet):** The diffusion forward and reverse processes operate exclusively on the compact latent tensor $z$, speeding up inference by over $40	imes$ while allowing the UNet to focus on semantic composition rather than imperceptible pixel noise.
3. **Synthesis (The VAE Decoder):** Once the latent denoising process completes, yielding clean latent $z_0$, the VAE decoder reconstructs the full-resolution RGB image: $x_{rec} = \mathcal{D}(z_0)$.

#### Core Sub-Networks in Stable Diffusion v1.5
Stable Diffusion v1.5 consists of three distinct deep neural networks:

| Component | Sub-Network | Parameters | Function in FacadeForge AI |
|---|---|---|---|
| **Text Encoder** | CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) | ~123 Million | Tokenizes text prompts into a sequence of 77 tokens and maps them into a 768-dimensional semantic embedding space. |
| **Diffusion Backbone** | Time-Conditioned Cross-Attention UNet | ~860 Million | Predicts the latent noise $\epsilon_	heta(z_t, t, c)$ at each timestep conditioned on text and spatial ControlNet features. |
| **Autoencoder** | AutoencoderKL (VAE Encoder & Decoder) | ~84 Million | Compresses $512 	imes 512 	imes 3$ RGB inputs into $64 	imes 64 	imes 4$ latents and decodes synthesized latents back to photorealistic RGB pixels. |

---

### 6.3 ControlNet: Structural Conditioning via Zero Convolutions

#### The Limitation of Text Prompts in Architecture
In architectural rendering, text prompts alone cannot guarantee geometric fidelity. A prompt such as *"A two-storey modern residence with three second-floor windows"* does not specify coordinates, perspective, cantilever angles, or facade boundaries. The base UNet has no mechanism to anchor pixels to exact physical lines.

#### ControlNet Architecture (Zhang & Agrawala, 2023)
ControlNet solves this by creating a structure-aware neural sidecar that locks the base model and attaches a trainable spatial adapter:

1. **Locking the Base Model:** The original Stable Diffusion UNet weights are frozen ($\Theta_{frozen}$). This guarantees that the massive visual and material knowledge base learned during pretraining (millions of natural images, textures, daylight distributions) is never degraded or catastrophically forgotten.
2. **The Trainable Clone:** ControlNet creates an exact architectural replica of the 12 encoding blocks and 1 middle block of the UNet ($\Theta_{trainable}$).
3. **Zero Convolutions ($\mathcal{Z}$):** The trainable copy is connected to the frozen model using specialized $1 	imes 1$ convolutional layers whose weight matrices and bias vectors are initialized to **identically zero**:
   $$\mathcal{W} = 0, \quad b = 0$$

```
   Locked Base UNet Block x                   Trainable Clone Block F
          |                                               |
          |                                       [ Condition: Canny Map ]
          |                                               |
          |                                               v
          |                                      [ Zero Conv Z_in ] (init W=0, b=0)
          |                                               |
          v                                               v
    [ UNet Block ] --------------------------> [ Trainable Copy Block ]
          |                                               |
          |                                               v
          |                                      [ Zero Conv Z_out ] (init W=0, b=0)
          |                                               |
          +-------------------- (+) <--------------------+
                                 |
                                 v
                            Output y
```

#### Why Zero Convolutions are Mathematically Elegant
Let $x$ be an intermediate feature map and $\mathcal{F}(x; \Theta)$ represent a neural network block. Passing feature $x$ through ControlNet with zero convolutions $\mathcal{Z}_{in}$ and $\mathcal{Z}_{out}$:
$$y = \mathcal{F}(x; \Theta_{frozen}) + \mathcal{Z}_{out}(\mathcal{F}(x + \mathcal{Z}_{in}(c; \Theta_{z1}); \Theta_{trainable}); \Theta_{z2})$$

Because the weights of $\mathcal{Z}_{in}$ and $\mathcal{Z}_{out}$ are initialized to zero:
$$\mathcal{Z}_{in}(c) = 0 \cdot c + 0 = 0$$
$$\mathcal{Z}_{out}(\dots) = 0 \cdot (\dots) + 0 = 0$$
Therefore, at step zero of training or before fine-tuning:
$$y = \mathcal{F}(x; \Theta_{frozen}) + 0 = \mathcal{F}(x; \Theta_{frozen})$$

**Viva Takeaway:** Zero convolutions guarantee that **no harmful noise, distortion, or destructive gradient is injected into the base model at initialization**. As gradients backpropagate during training on edge-image pairs, the weights smoothly diverge from zero, gradually introducing spatial control while keeping the base model's photorealistic rendering engine completely intact.

---

### 6.4 Canny Edge Detection: Mathematical Foundations & Stages

Canny Edge Detection (developed by John F. Canny in 1986) is an optimal multi-stage edge detection algorithm governed by three mathematical criteria:
1. **Low Error Rate:** No genuine edge should be missed, and no spurious noise should be marked as an edge.
2. **Localization:** The distance between detected edge pixels and actual physical boundaries must be minimized.
3. **Single Response:** One edge must produce only one single-pixel response, avoiding thick, blurry lines.

The Canny algorithm implemented in FacadeForge AI (`backend/inference/colab_server.py`) operates in four distinct stages:

#### Stage 1: Gaussian Smoothing (Noise Reduction)
Because numerical differentiation is acutely sensitive to high-frequency image noise, the input grayscale image $I$ is first convolved with a symmetric 2D Gaussian kernel $G_\sigma(x, y)$:
$$G_\sigma(x, y) = rac{1}{2\pi \sigma^2} \exp\left( -rac{x^2 + y^2}{2\sigma^2} ight)$$
$$I_\sigma(x, y) = I(x, y) * G_\sigma(x, y)$$
In OpenCV's `cv2.Canny`, an optimized $5 	imes 5$ Gaussian kernel is applied internally.

#### Stage 2: Gradient Magnitude and Orientation Calculation
The smoothed image is filtered with horizontal ($K_x$) and vertical ($K_y$) Sobel operator kernels:
$$K_x = egin{bmatrix} -1 & 0 & +1 \ -2 & 0 & +2 \ -1 & 0 & +1 \end{bmatrix}, \quad K_y = egin{bmatrix} +1 & +2 & +1 \ 0 & 0 & 0 \ -1 & -2 & -1 \end{bmatrix}$$
The directional spatial derivatives are computed:
$$G_x = K_x * I_\sigma, \quad G_y = K_y * I_\sigma$$
From these, the edge gradient magnitude $G(x, y)$ and edge direction angle $	heta(x, y)$ are derived:
$$G(x, y) = \sqrt{G_x(x, y)^2 + G_y(x, y)^2}$$
$$	heta(x, y) = rctan\left( rac{G_y(x, y)}{G_x(x, y)} ight)$$
The gradient direction angle is rounded to one of four fundamental directions: $0^\circ$ (horizontal), $45^\circ$ (positive diagonal), $90^\circ$ (vertical), or $135^\circ$ (negative diagonal).

#### Stage 3: Non-Maximum Suppression (Edge Thinning)
Gradient calculation produces thick ridges around lines. Non-Maximum Suppression (NMS) thins these ridges into sharp, 1-pixel-wide lines:
1. For each pixel $(x, y)$, compare its gradient magnitude $G(x, y)$ against its two immediate neighbors along the gradient vector direction $	heta(x, y)$.
2. If $G(x, y)$ is smaller than either of its directional neighbors, set its value to zero (suppressed).
3. If $G(x, y)$ is strictly greater than both neighbors, preserve it as a local maximum candidate.

#### Stage 4: Hysteresis Thresholding (Dual-Threshold Filtering)
Single-threshold edge detectors suffer from two failures: setting the threshold too high fragments continuous lines, while setting it too low captures unwanted paper texture and scan noise. Hysteresis solves this using **two discrete thresholds**:
- **Low Threshold ($T_{low} = 50$)**
- **High Threshold ($T_{high} = 150$)**

Pixels are classified into three categories:
1. **Strong Edge ($G \ge T_{high} = 150$):** Instantly confirmed and retained as a definitive architectural edge.
2. **Suppressed Non-Edge ($G < T_{low} = 50$):** Instantly discarded as background noise.
3. **Weak Edge ($50 \le G < 150$):** Evaluated based on topological connectivity. A weak edge pixel is retained **if and only if it is directly connected (8-connectivity) to a confirmed strong edge pixel**.

```
Gradient Value G:
      ^
      |   STRONG EDGES (G >= 150)  --> Always Kept (Definitive wall/roof line)
T_high|---------------------------------------------------------------
      |   WEAK EDGES (50 <= G < 150) --> Kept ONLY if connected to Strong Edge
 T_low|---------------------------------------------------------------
      |   NOISE / SUPPRESSED (G < 50) --> Always Discarded (Paper grain, smudge)
      +---------------------------------------------------------------> Pixels
```

**Why 50 and 150 in FacadeForge AI?**  
Architectural sketches feature high-contrast structural contours alongside lighter shading lines. A ratio of $1:3$ ($T_{low}=50, T_{high}=150$) provides the optimal empirical balance: it traces rooflines, window mullions, and floor slabs without picking up pencil smudge, paper wrinkles, or faint hatching.

---

### 6.5 Integrated Pipeline: Scheduler, Guidance, and Optimizations

#### 1. DPMSolverMultistepScheduler
The diffusion denoising trajectory is mathematically equivalent to solving a reverse-time Stochastic Differential Equation (SDE) or an associated Probability Flow Ordinary Differential Equation (ODE). Traditional solvers like Euler-Maruyama or DDIM take small, uniform linear steps, requiring 50 to 100 steps to prevent compounding approximation errors.

FacadeForge AI uses **DPM-Solver++** (Lu et al., 2022). It employs an exact analytical formulation for the linear component of diffusion ODEs combined with high-order Adams-Bashforth multi-step polynomial interpolation for the non-linear score function. This achieves photorealistic image synthesis in just **20 steps**, slashing inference latency by over 60% compared to legacy schedulers.

#### 2. Classifier-Free Guidance (CFG Scale = 7.5)
Classifier-Free Guidance (Ho & Salimans, 2021) controls how aggressively the model adheres to the text prompt versus generating generic visuals. At each denoising step $t$, the UNet is evaluated twice:
1. **Unconditioned Evaluation:** $\epsilon_	heta(z_t, t, \emptyset)$ (empty text prompt $\emptyset$).
2. **Conditioned Evaluation:** $\epsilon_	heta(z_t, t, c)$ (conditioned on architectural prompt $c$).

The final modified noise vector is computed by extrapolating in the direction of the condition:
$$	ilde{\epsilon}_	heta(z_t, t, c) = \epsilon_	heta(z_t, t, \emptyset) + s \cdot \left[ \epsilon_	heta(z_t, t, c) - \epsilon_	heta(z_t, t, \emptyset) ight]$$
Where $s$ is the **Guidance Scale** ($s = 7.5$ in FacadeForge AI). A scale of 7.5 provides the ideal balance: high enough to enforce distinct materials (concrete, stone, glass) while avoiding the visual saturation and contrast burning that occurs at $s > 10$.

#### 3. ControlNet Conditioning Scale ($1.0$)
The parameter `controlnet_conditioning_scale=1.0` dictates the linear weight of ControlNet residual features added to the frozen UNet decoder. At 1.0, the model enforces strict structural compliance with the Canny edges. Lowering this value (e.g., 0.5) gives the model artistic freedom to alter the geometry, whereas raising it too high (>1.2) can cause harsh edge artifacts.

#### 4. Hardware Precision & VRAM Optimizations
Running Stable Diffusion + ControlNet concurrently requires careful GPU memory management:
- **`torch.float16` (Half Precision):** Model weights and activations are stored in 16-bit floating point format rather than standard 32-bit (`float32`). This cuts memory footprint exactly in half and leverages the Tensor Cores of the Nvidia Tesla T4 GPU for high-speed matrix multiplications.
- **Attention Slicing (`pipe.enable_attention_slicing()`):** Slices the memory-intensive multi-head cross-attention computation into sequential matrix chunks, preventing Out-Of-Memory (OOM) spikes during attention map calculation.
- **Direct GPU Residency (`pipe = pipe.to("cuda")`):** Instead of using CPU offloading (which repeatedly shuttles multi-gigabyte weight tensors across the slow PCIe bus to system RAM, crashing Colab), all model layers reside continuously in the 16 GB GDDR6 VRAM.

---

## 7. Detailed Codebase & Module Walkthrough

The FacadeForge AI codebase is organized modularly to enforce clear separation of concerns:

```
FacadeForge_AI/
|-- backend/
|   |-- app.py                     # Local Flask application gateway and job queue
|   |-- generated/                 # Directory holding synthesized PNG renders
|   |   +-- .gitkeep
|   +-- inference/
|       |-- __init__.py
|       |-- colab_server.py        # Remote inference server running on Colab GPU
|       |-- pipeline.py            # Local inference adapter & prerequisite checker
|       +-- remote_client.py       # Authenticated HTTP client communicating with Colab
|-- frontend/
|   |-- static/
|   |   |-- css/site.css           # Architectural editorial styling system & responsive rules
|   |   +-- js/
|   |       |-- site.js            # Landing page navigation drawer logic
|   |       +-- workspace.js       # Asynchronous studio state, upload & polling loop
|   +-- templates/
|       |-- index.html             # Landing page template (hero, features, portfolio)
|       +-- workspace.html         # Workspace studio template (dual-panel upload/render)
|-- assets/demo/                   # Demonstration portfolio images for landing page
|-- docs/                          # Project documentation and viva handbooks
|-- facade_forge.ipynb             # Google Colab experimentation and pipeline notebook
|-- requirements.txt               # Local gateway minimal dependencies (Flask, Pillow, NumPy)
|-- requirements-colab.txt         # Remote Colab server dependency flag (Flask)
|-- requirements-ai.txt            # Legacy optional local inference dependencies
+-- .env.example                   # Template for remote URL, token, and timeout settings
```

### 7.1 Local Backend Gateway (`backend/app.py`)

`backend/app.py` serves as the primary local application controller. It manages client sessions, serves static assets and templates, enforces upload safety policies, and coordinates asynchronous generation requests.

#### Key Functions and Endpoints:
- `_load_environment_file(path: Path)`: Reads `.env` key-value pairs without requiring third-party libraries like `python-dotenv`.
- `@app.before_request def _refresh_env()`: Automatically re-reads `.env` before every incoming HTTP request. This enables dynamic hot-reloading: if the Cloudflare tunnel restarts and issues a new URL, the user simply updates `.env` without needing to restart the Flask server.
- `GET /`: Renders `frontend/templates/index.html`.
- `GET /workspace`: Renders `frontend/templates/workspace.html`.
- `GET /api/status`: Calls `remote_client.check_remote_health()` and returns JSON status regarding Colab GPU readiness.
- `POST /api/generate`: Receives the sketch upload. Validates:
  1. Presence of file (`upload is None`).
  2. Max file size: capped at 8 MB (`MAX_CONTENT_LENGTH`).
  3. Image integrity: opens via `PIL.Image.open()`, checks `image.format in {"PNG", "JPEG"}`.
  4. Decompression bomb prevention: checks `image.width * image.height > 40_000_000` (rejects massive image bombs with HTTP 413).
  5. Architectural style key: validates against `{"modern", "luxury", "indian", "industrial", "scandinavian"}`.
  6. Dispatches `threading.Thread(target=_run_job, args=(job_id, source, style), daemon=True)`.
  7. Returns `HTTP 202 Accepted` with payload `{"id": job_id, "state": "queued"}`.
- `_run_job(job_id, source, style)`: Background worker function. Invokes `generate_remote_image(source, style)` from `remote_client.py`. On success, writes binary PNG bytes to `backend/generated/{job_id}.png` and updates in-memory job state to `complete`. On failure, catches `RemoteInferenceError` and records the exact user-safe error message.
- `GET /api/jobs/<job_id>`: Returns the active job status JSON (`queued`, `running`, `complete`, or `error`).
- `GET /api/jobs/<job_id>/image`: Streams the synthesized PNG from `backend/generated/{job_id}.png` with MIME type `image/png`.
- `GET /api/jobs/<job_id>/download`: Serves the generated PNG as a forced attachment download (`as_attachment=True, download_name="facadeforge-render.png"`).

---

### 7.2 Remote Colab GPU Inference Server (`backend/inference/colab_server.py`)

`colab_server.py` is the authenticated inference microservice executed inside the Google Colab T4 GPU environment.

#### Key Components:
- `set_pipeline(pipeline)`: Crucial memory optimization function. Accepts the already initialized in-memory pipeline (`pipe`) created by the notebook and assigns it to `_pipeline`. This allows the server thread to share the GPU pipeline directly, completely eliminating redundant model loading.
- `_get_pipeline()`: Lazy-loading fallback. If `set_pipeline` was not invoked, it verifies CUDA availability and loads SD 1.5 + ControlNet Canny once from Colab's local Hugging Face cache using `local_files_only=True`.
- `_authorized() -> bool`: Compares the incoming `Authorization: Bearer <token>` header against the environment variable `FACADEFORGE_REMOTE_TOKEN` using `hmac.compare_digest` to prevent timing attacks.
- `_prepare_sketch(image: Image.Image) -> Image.Image`: Implements the standard preprocessing pipeline:
  1. `ImageOps.contain(image.convert("RGB"), (512, 512))`
  2. Canvas paste on white background at center coordinates.
  3. `cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2GRAY)`
  4. `cv2.Canny(gray, 50, 150)`
  5. Converts binary edge map to 3-channel RGB PIL Image.
- `GET /health`: Authenticated health check. Verifies token and returns GPU status, model readiness, and base model IDs without exposing secrets.
- `POST /generate`: Authenticated inference endpoint:
  1. Validates bearer token.
  2. Ingests multipart form containing `sketch` and `style`.
  3. Executes `_prepare_sketch(original)`.
  4. Acquires thread lock (`with _inference_lock:`) to prevent simultaneous multi-thread GPU access.
  5. Runs the diffusion pipeline with DPMSolver++ across 20 steps at guidance scale 7.5.
  6. Encodes result into an in-memory `io.BytesIO` buffer as PNG.
  7. Returns `send_file(image_buffer, mimetype="image/png", max_age=0)`.
- `start_server(host="0.0.0.0", port=7860)`: Starts the Flask app inside a background daemon thread (`threading.Thread`), checking `is_port_in_use(port)` first to prevent duplicate port binding errors when notebook cells are re-run.

---

### 7.3 Authenticated Remote HTTP Client (`backend/inference/remote_client.py`)

`remote_client.py` provides the local gateway with a robust, authenticated transport layer to communicate with the Colab inference worker across the Cloudflare tunnel.

#### Key Functions:
- `_configuration() -> tuple[str, str]`: Reads `FACADEFORGE_REMOTE_URL` and `FACADEFORGE_REMOTE_TOKEN`. Validates that the URL uses HTTPS (or localhost for dev) and strips trailing slashes, fragments, and queries.
- `_request(path, data, content_type, timeout)`: Low-level network wrapper using Python's standard `urllib.request`. Attaches the `Authorization: Bearer <token>` header and maps standard network errors (`HTTPError`, `URLError`, `socket.timeout`) into human-readable `RemoteInferenceError` exceptions.
- `check_remote_health() -> dict`: Executes a non-blocking 8-second GET request to `/health`, translating remote status into UI-ready connection messages.
- `_multipart(sketch: bytes, style: str) -> tuple[bytes, str]`: Custom multipart form-data serializer that packs raw sketch bytes and the style string into standard boundary chunks without requiring third-party libraries like `requests-toolbelt`.
- `generate_remote_image(sketch: bytes, style: str) -> bytes`: Executes the remote generation call with a configurable timeout (default 300 seconds, max 900 seconds). Enforces strict post-generation security:
  - Validates that the response header specifies `image/png`.
  - Verifies that image size does not exceed 24 MB (`MAX_GENERATED_BYTES`).
  - Confirms the binary begins with the PNG magic byte signature (`PNG

`).
  - Uses `PIL.Image.open().verify()` to ensure no packets were corrupted during transit.

---

### 7.4 Local Fallback Adapter (`backend/inference/pipeline.py`)

`pipeline.py` is a standalone, local-only adapter designed to support environments where an Nvidia CUDA GPU is physically present on the local machine.
- Probes system dependencies (`torch`, `diffusers`, `cv2`) and CUDA availability.
- Prevents accidental multi-gigabyte downloads by enforcing `local_files_only=True`.
- In the standard project setup, this module remains idle because inference is offloaded to the Colab GPU.

---

### 7.5 Interactive Frontend (`frontend/`)

#### 1. Landing Page (`frontend/templates/index.html`)
- Serves as the public presentation showcase.
- Features high-impact hero typography, SVG blueprint animations, an interactive navigation drawer, three curated project demonstrations (Urban Apartment, Two-Storey Residence, Modern Villa), and direct CTA links to `/workspace`.

#### 2. Workspace Studio (`frontend/templates/workspace.html`)
- Structured as a balanced two-column desktop workspace (`.workspace-grid`):
  - **Left Panel (`.input-panel`):** Houses the sketch drag-and-drop zone (`#drop-zone`), real-time file details (`#preview-label`), architectural style selector (`#style-select`), dynamic style descriptions (`#style-description`), and the primary **Generate facade** button.
  - **Right Panel (`.result-panel`):** Houses the empty-state canvas placeholder (`#result-empty`) and the generation comparison grid (`#result-content`).
- Includes an active generation status badge (`#generation-status`) with real-time state styling (`data-state="ready|running|error"`).

#### 3. Frontend Controller (`frontend/static/js/workspace.js`)
- Implements state management for the entire studio workflow.
- `setFile(file)`: Validates file type and size (<8MB), creates an object URL, updates preview image sources, unhides the preview canvas, and enables the generate button.
- `clearFile()`: Cleans up object URLs (`URL.revokeObjectURL`) to prevent browser memory leaks, resets inputs, and restores empty states.
- `pollJob(jobId)`: Asynchronous loop polling `/api/jobs/<jobId>` every 1,200 ms. Dynamically updates status messages (*"Request queued...", "The model is generating your facade..."*).
- `generate()`: Dispatches the upload via `fetch('/api/generate')`, initiates polling, and upon completion sets `#result-original` and `#result-generated`, unhiding the comparison view.

#### 4. Styling System (`frontend/static/css/site.css`)
- Custom architectural editorial aesthetic utilizing Google Fonts (`DM Sans` for body copy and `Playfair Display` for refined serif headings).
- Fully responsive across desktop, tablet (`@media(max-width:1000px)`), and mobile (`@media(max-width:700px)`).
- **Critical Display Rule:** Enforces `[hidden], .result-empty[hidden], .result-content[hidden] { display: none !important; }`. This guarantees that the 340px empty-state blueprint container is completely excised from the document flow as soon as generation completes, allowing the generated image and controls to display directly beneath the **"Visualization"** heading without requiring user scrolling.

---

## 8. Remote Inference, Cloudflare Tunnels, & Deployment Engineering

```
+----------------------------------------------------------------------------------------------------+
|                                    REMOTE INFERENCE TOPOLOGY                                       |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  LOCAL MACHINE (Client & Flask Gateway)                                                            |
|  [ User Browser ] <---> [ Flask Gateway (127.0.0.1:5000) ]                                         |
|                                |                                                                   |
|                                | HTTPS POST /generate (Bearer Token Auth)                          |
|                                v                                                                   |
|  PUBLIC INTERNET / EDGE NETWORK                                                                    |
|  [ Cloudflare Global Edge Network ] (Edge SSL Termination, DDoS Protection)                        |
|                                |                                                                   |
|                                | Outbound Encrypted Tunnel (Argo Tunnel / QUIC Protocol)           |
|                                v                                                                   |
|  GOOGLE COLAB CLOUD RUNTIME                                                                        |
|  [ cloudflared Daemon Process ] (Runs in background, immune to Jupyter signals)                    |
|                                |                                                                   |
|                                | Local Forwarding to localhost:7860                                |
|                                v                                                                   |
|  [ Colab Flask Server (colab_server.py) ] <---> [ Pretrained Model Cache (T4 GPU VRAM) ]            |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 8.1 Why Remote Inference? (The Hardware Problem)
Generative diffusion models with ControlNet adapters are computationally demanding:
- **VRAM Requirements:** Running SD 1.5 + ControlNet Canny requires a bare minimum of 6 GB to 8 GB of dedicated VRAM in `torch.float16`.
- **Consumer Hardware Limitations:** Most standard student or enterprise laptops feature integrated graphics (Intel Iris Xe, Apple M-series shared memory) or low-tier discrete GPUs (2–4 GB VRAM), which immediately encounter PyTorch CUDA Out-Of-Memory (OOM) fatal crashes.
- **CPU Inference Unfeasibility:** Running diffusion models on a multi-core CPU requires 5 to 15 minutes per image, completely destroying user experience.
- **The Colab Solution:** Google Colab provides free access to enterprise-grade **Nvidia Tesla T4 GPUs** equipped with 16 GB of high-speed GDDR6 VRAM and hardware Tensor Cores. Offloading inference to Colab reduces execution time to ~8–12 seconds.

### 8.2 Cloudflare Quick Tunnels Deep Dive
Exposing a server running inside Google Colab presents major networking hurdles:
- Colab instances run inside private Google Compute Engine subnets behind strict NAT (Network Address Translation) and enterprise firewalls. Inbound traffic cannot reach Colab directly.
- Standard port-forwarding is impossible because users do not have administrative access to Google's border routers.

FacadeForge AI overcomes this using **Cloudflare Quick Tunnels (`cloudflared`)**:
1. **Outbound Tunnel Initiation:** The `cloudflared` client binary is downloaded and executed directly inside Colab:
   ```bash
   cloudflared tunnel --url http://127.0.0.1:7860
   ```
2. **Reverse WebSocket / QUIC Connection:** `cloudflared` initiates an **outbound** connection to the nearest Cloudflare edge data center. Because firewalls permit outbound traffic, the connection establishes instantly without firewall exceptions.
3. **Public Edge Routing:** Cloudflare assigns an ephemeral domain (e.g., `https://random-words.trycloudflare.com`) and manages automatic SSL/TLS certificate termination at its edge.
4. **Traffic Proxying:** Inbound HTTPS requests from our local Flask gateway hit the Cloudflare edge URL and are tunneled through the active connection directly to `127.0.0.1:7860` in Colab.

### 8.3 Security Architecture: HMAC Bearer Token Authentication
Because Cloudflare Quick Tunnel URLs are public and can theoretically be discovered by port scanners, FacadeForge AI enforces strict cryptographic authentication:
1. **Token Generation:** The developer generates a 256-bit cryptographically secure token using Python's standard `secrets` library:
   ```python
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
2. **Distribution:** The exact same token is stored in the local `.env` (`FACADEFORGE_REMOTE_TOKEN`) and entered into Colab (`os.environ["FACADEFORGE_REMOTE_TOKEN"]`).
3. **Timing-Attack Resistance:** In `colab_server.py`, incoming bearer tokens are validated using Python's `hmac.compare_digest()`:
   ```python
   def _authorized() -> bool:
       expected = os.getenv("FACADEFORGE_REMOTE_TOKEN", "")
       supplied = request.headers.get("Authorization", "")
       if not expected:
           return False
       return hmac.compare_digest(supplied, f"Bearer {expected}")
   ```
   **Viva Note:** Standard string comparison (`==`) terminates execution at the first non-matching character, allowing attackers to measure microscopic response time differences to incrementally guess secret tokens (timing side-channel attacks). `hmac.compare_digest` executes in constant time regardless of where the mismatch occurs, completely neutralizing timing attacks.


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
   python backendpp.py
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
The native resolution of SD 1.5 is $512 	imes 512$ pixels. Integrating a secondary **Real-ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)** or Latent Upscaler stage will upscale generated facade renders to $2048 	imes 2048$ (2K) or $4096 	imes 4096$ (4K) print-ready resolution, sharpening glass reflections, brick mortar lines, and wood grain for client presentations.

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


## 14. Comprehensive Viva Voce Question Bank

This section contains over 100 high-probability viva examination questions categorized systematically across sixteen academic domains. Each question is structured with:
- **Spoken Answer:** A concise, punchy 15–20 second response suitable for immediate verbal delivery.
- **Deep Technical Explanation:** The comprehensive theoretical and architectural rationale to defend against aggressive cross-questioning.
- **FacadeForge AI Context:** Specific code, configuration, or architectural evidence tying the answer directly to our repository.

---

### Category A: Basic Project Questions

#### Q1. What is FacadeForge AI in one simple sentence?
- **Spoken Answer:** FacadeForge AI is a deep-learning web platform that translates 2D architectural building facade sketches into photorealistic exterior visualizations while strictly preserving the building's geometry and fenestration using Stable Diffusion 1.5 and ControlNet Canny.
- **Deep Explanation:** It is a structure-conditioned generative image synthesis system. It takes an untextured black-and-white structural drawing, extracts its spatial boundaries via Canny edge detection, and uses a latent diffusion model conditioned on architectural style prompts to synthesize realistic building materials (concrete, glass, timber, stone) under realistic natural daylight.
- **Project Context:** Implemented across a local Flask web gateway (`backend/app.py`) and a remote Google Colab GPU server (`backend/inference/colab_server.py`).

#### Q2. What problem does FacadeForge AI solve in the real world?
- **Spoken Answer:** It eliminates the time-consuming and expensive 3D CAD modeling and rendering bottleneck required during the early concept presentation stage of architectural design.
- **Deep Explanation:** Architects can conceptualize ideas rapidly in sketches, but non-technical clients cannot visualize finished buildings from wireframes or line art. Traditional 3D modeling (Revit, 3ds Max) and ray-tracing (V-Ray, Lumion) takes 8 to 48 hours per iteration. FacadeForge AI delivers client-ready photorealistic visual studies in ~10 seconds directly from raw sketches.
- **Project Context:** Discussed in `docs/project_overview.txt` and highlighted on the landing page (`frontend/templates/index.html`).

#### Q3. Is FacadeForge AI an automated CAD generator or a visualization tool?
- **Spoken Answer:** It is strictly an architectural concept visualization tool, not an engineering or CAD drafting tool.
- **Deep Explanation:** The application generates conceptual visual renders for design exploration, client communication, and aesthetic review. It does not output BIM files, structural stress calculations, or construction-ready blueprints.
- **Project Context:** Explicitly declared in the footer of `frontend/templates/workspace.html`: *"ControlNet guides composition; structural details may vary. Preliminary concept visualization only."*

#### Q4. What are the key features of the application?
- **Spoken Answer:** Drag-and-drop sketch upload, client-side preview, five curated architectural style selections, real-time GPU health checks, asynchronous image generation, side-by-side comparative visualization, and full-resolution PNG download.
- **Deep Explanation:** Supported by an asynchronous job queue that decouples user interaction from remote GPU processing, paired with deterministic image containment on a 512×512 canvas.
- **Project Context:** Implemented in `frontend/static/js/workspace.js` and `backend/app.py`.

---

### Category B: Problem Statement and Objectives

#### Q5. Why can't architects just use traditional 3D software for early concepts?
- **Spoken Answer:** Traditional 3D software requires modeling every wall, window frame, and material texture from scratch, which is too slow and costly for early brainstorming sessions where designs change daily.
- **Deep Explanation:** In early ideation, an architect might produce ten different facade sketches in an afternoon. Creating 3D polygon meshes, unwrapping UV coordinates, applying PBR shaders, and configuring global illumination for ten concepts in software like 3ds Max would require an entire week of billable production time.
- **Project Context:** Section 2 of `docs/project_overview.txt`.

#### Q6. What are the primary academic objectives of this project?
- **Spoken Answer:** 1) Develop a structure-preserving sketch-to-image translation system; 2) Evaluate pretrained diffusion models and ControlNet conditioning; 3) Build a decoupled hybrid web architecture for accessible GPU computing; and 4) Provide an intuitive architectural comparison studio.
- **Deep Explanation:** The project aims to demonstrate that spatial conditioning algorithms can overcome the stochastic, unconstrained hallucination inherent in standard text-to-image foundation models.
- **Project Context:** Codebase `facade_forge.ipynb` cells 4–6 and `backend/inference/pipeline.py`.

---

### Category C: Technology Stack Questions

#### Q7. List the complete technology stack used in FacadeForge AI.
- **Spoken Answer:** Python 3.10, Flask 3.0, PyTorch 2.x with CUDA 12, Stable Diffusion v1.5, ControlNet Canny, Hugging Face Diffusers, OpenCV, Pillow, Google Colab Nvidia Tesla T4 GPU, Cloudflare Quick Tunnel, and Vanilla HTML5/CSS3/JavaScript.
- **Deep Explanation:** Each tool was chosen for a specific role: Flask for lightweight web orchestration, PyTorch for tensor math in `float16`, Diffusers for pipeline abstractions, OpenCV for Canny edge extraction, Pillow for canvas aspect containment, Colab for 16 GB VRAM access, and Cloudflare for secure reverse-proxy tunneling.
- **Project Context:** Verified across `requirements.txt`, `requirements-colab.txt`, and source files.

#### Q8. Why did you use Python instead of JavaScript/Node.js for the backend?
- **Spoken Answer:** Python is the native language of the deep learning ecosystem, providing direct integration with PyTorch, Hugging Face Diffusers, OpenCV, and PIL without foreign function interfaces.
- **Deep Explanation:** While Node.js excels at I/O concurrency, deep learning model weights, tensor manipulation, and GPU CUDA drivers are native to Python. Using Python across both the local gateway and remote server ensures unified data serialization and shared utility modules.
- **Project Context:** `backend/app.py` and `backend/inference/colab_server.py`.

---

### Category D: Python and Flask Questions

#### Q9. How does Flask handle concurrent generation requests in `app.py`?
- **Spoken Answer:** Flask runs with `threaded=True`. When a generation request arrives, it assigns the heavy processing to an independent background daemon thread (`threading.Thread`), instantly returning HTTP 202 Accepted so the main server thread never blocks.
- **Deep Explanation:** In `backend/app.py`, the `@app.post("/api/generate")` route generates a UUID4 job ID, registers a job state in a thread-safe `_jobs` dictionary protected by `_jobs_lock = threading.Lock()`, and spawns `_run_job` in a daemon thread. Subsequent client requests for static files or polling endpoints are handled immediately by other WSGI worker threads.
- **Project Context:** Lines 55–134 of `backend/app.py`.

#### Q10. What is the role of `threading.Lock()` in `app.py` and `colab_server.py`?
- **Spoken Answer:** It is a mutual exclusion primitive that prevents race conditions when multiple threads read or write to shared dictionaries or send simultaneous requests to the GPU.
- **Deep Explanation:** In `app.py`, `_jobs_lock` prevents corrupted state if two threads update job metadata simultaneously. In `colab_server.py`, `_inference_lock` guarantees that only one thread can invoke the PyTorch pipeline at any given moment, preventing GPU CUDA core contention and VRAM memory corruptions.
- **Project Context:** `backend/app.py` line 56 and `backend/inference/colab_server.py` line 57.

#### Q11. How does the application prevent decompression bomb attacks?
- **Spoken Answer:** Pillow checks whether the total pixel count ($width 	imes height$) exceeds 40,000,000 pixels. If it does, Flask rejects the upload with an HTTP 413 error before allocating memory.
- **Deep Explanation:** Attackers can craft tiny 50 KB PNG files that expand into 10 gigabytes of uncompressed bitmap data in memory (a decompression bomb or "zip bomb" for images). Pillow's `Image.DecompressionBombError` and our explicit pixel ceiling ($40 	ext{ MP}$) neutralize this denial-of-service vector.
- **Project Context:** `backend/app.py` lines 116–120.

---

### Category E: Frontend and Backend Integration

#### Q12. How does the frontend communicate with the backend?
- **Spoken Answer:** Through asynchronous REST API calls using the browser's native `fetch()` API, accompanied by a 1,200 ms polling interval to monitor generation job progress.
- **Deep Explanation:** When a user clicks generate, `workspace.js` sends a `POST /api/generate` multipart request. The server replies with an HTTP 202 JSON payload containing the job ID. The frontend then runs `pollJob()`, executing `GET /api/jobs/<jobId>` every 1.2 seconds until `state === 'complete'`.
- **Project Context:** `frontend/static/js/workspace.js` lines 91–140.

#### Q13. Why did you use polling instead of WebSockets?
- **Spoken Answer:** Polling is simpler, stateless, firewall-friendly, and perfectly suited for a low-frequency workflow where generation takes ~10 seconds and only a few status checks occur.
- **Deep Explanation:** WebSockets require persistent, stateful TCP connections, bidirectional heartbeat pings, and dedicated socket servers that complicate deployment across reverse-proxy tunnels like Cloudflare. Polling over standard HTTP/1.1 or HTTP/2 endpoints is highly reliable, stateless, and trivial to scale.
- **Project Context:** `workspace.js` function `pollJob()`.

---

### Category F: Stable Diffusion Questions

#### Q14. What is the difference between pixel-space diffusion and latent-space diffusion?
- **Spoken Answer:** Pixel-space diffusion operates directly on full-resolution RGB pixels ($512 	imes 512 	imes 3$), making it computationally sluggish; latent diffusion compresses images $8	imes$ into a compact latent space ($64 	imes 64 	imes 4$) using a VAE, accelerating diffusion math by over $40	imes$.
- **Deep Explanation:** High-resolution digital images contain redundant perceptual high-frequency details. Standard DDPM spends immense compute denoising imperceptible pixel noise. Latent Diffusion Models (LDM) compress the spatial resolution to a $64 	imes 64$ continuous manifold using an AutoencoderKL, allowing the UNet to focus on semantic composition and spatial structure with drastic VRAM and compute savings.
- **Project Context:** Chapter 6.2 of this handbook.

#### Q15. What are the three core sub-networks inside Stable Diffusion v1.5?
- **Spoken Answer:** 1) The CLIP ViT-L/14 text encoder; 2) The time-conditioned cross-attention UNet; and 3) The AutoencoderKL (VAE) encoder/decoder.
- **Deep Explanation:** The CLIP text encoder converts text prompts into 768-dimensional token vectors. The UNet predicts the noise vector $\epsilon_	heta$ at each timestep conditioned on text embeddings and time step embeddings. The VAE decodes the final denoised latent tensor $z_0$ into the final $512 	imes 512 	imes 3$ RGB image.
- **Project Context:** Cell 5 of `facade_forge.ipynb`.

#### Q16. What is the role of the VAE in Stable Diffusion?
- **Spoken Answer:** The VAE acts as a perceptual compressor and decompressor, translating RGB pixel images into latent space for the diffusion UNet, and translating the final denoised latent tensor back into viewable RGB pixels.
- **Deep Explanation:** Specifically, AutoencoderKL employs a regularized latent space governed by a Kullback-Leibler (KL) divergence loss against standard normal distributions, preventing arbitrary variance explosion and ensuring smooth, interpolatable latent spaces.
- **Project Context:** AutoencoderKL in `diffusers` library.

---

### Category G: ControlNet Questions

#### Q17. How does ControlNet differ from standard image-to-image (img2img)?
- **Spoken Answer:** Standard img2img simply adds noise to an initial image and denoises it, losing fine structural lines and warping geometry; ControlNet conditions the diffusion UNet on an explicit spatial edge map, guaranteeing structural preservation.
- **Deep Explanation:** In standard img2img, the entire image is mapped to latent space and partially corrupted with noise. High noise levels destroy the building's lines, while low noise levels prevent realistic material changes. ControlNet decouples structure from material: the diffusion process starts from pure noise (allowing full creative freedom for materials), but ControlNet constrains the UNet's attention layers to follow the structural Canny edges.
- **Project Context:** Section 6.3 of this handbook.

#### Q18. What are "Zero Convolutions" in ControlNet and why are they important?
- **Spoken Answer:** They are $1 	imes 1$ convolutional layers whose weights and biases are initialized to zero, ensuring that the model outputs zero residual features at step zero, protecting the base model from destructive noise during training.
- **Deep Explanation:** In ControlNet, feature outputs from the trainable clone are passed through zero convolutions before being added to the frozen UNet blocks: $y = \mathcal{F}(x) + \mathcal{Z}(F_{clone})$. Because $\mathcal{Z}$ starts at zero, $y = \mathcal{F}(x)$, meaning the model begins with the exact baseline performance of pretrained Stable Diffusion without degradation.
- **Project Context:** Zhang & Agrawala (2023) research paper; referenced in `colab_server.py`.

#### Q19. What happens if the `controlnet_conditioning_scale` is set too low or too high?
- **Spoken Answer:** If set too low (e.g., 0.2), the model ignores the sketch lines and hallucinates new geometry; if set too high (e.g., 1.8), it produces harsh, dark edge artifacts and over-saturated outlines.
- **Deep Explanation:** At 1.0 (our project setting), the spatial edge guidance is given equal weight to the base model's internal generative priors, producing photorealistic materials that snap perfectly to the sketch lines without edge distortion.
- **Project Context:** `colab_server.py` line 182: `controlnet_conditioning_scale=1.0`.

---

### Category H: Canny Edge Detection Questions

#### Q20. Walk through the four stages of the Canny edge detection algorithm.
- **Spoken Answer:** 1) Gaussian blur for noise reduction; 2) Sobel gradient calculation for intensity changes; 3) Non-maximum suppression to thin edges to single pixels; and 4) Hysteresis thresholding to connect genuine edges.
- **Deep Explanation:** Gaussian filtering eliminates pixel noise. Sobel kernels compute spatial derivatives $G_x$ and $G_y$ to determine gradient magnitude $G = \sqrt{G_x^2 + G_y^2}$ and direction $	heta$. Non-maximum suppression suppresses pixels that are not local maxima along the gradient direction. Hysteresis classifies pixels using two thresholds ($T_{low}=50, T_{high}=150$), retaining weak edges only if they link to strong edges.
- **Project Context:** `backend/inference/colab_server.py` lines 122–124: `cv2.Canny(gray, 50, 150)`.

#### Q21. Why did you choose hysteresis thresholds of 50 and 150?
- **Spoken Answer:** A 1:3 ratio with $T_{low}=50$ and $T_{high}=150$ isolates bold architectural structural outlines and window frames while ignoring pencil texture, paper grain, and faint sketch shading.
- **Deep Explanation:** Setting $T_{low} < 30$ causes the detector to pick up paper texture and scanner smudges as structural edges. Setting $T_{high} > 200$ causes faint architectural lines (like balcony railings or thin mullions) to break into fragmented dashes. The $50/150$ pair provides robust empirical performance across both scanned ink drawings and digital sketches.
- **Project Context:** `colab_server.py` line 123.

---

### Category I: Deep Learning and Generative AI Fundamentals

#### Q22. Compare Stable Diffusion with Generative Adversarial Networks (GANs).
| Feature | Stable Diffusion (Diffusion Models) | GANs (e.g., Pix2Pix, CycleGAN) |
|---|---|---|
| **Training Stability** | Highly stable; optimizes convex L2 noise reconstruction loss. | Highly unstable; suffers from minimax oscillation and mode collapse. |
| **Output Diversity** | Massive visual diversity governed by stochastic latent sampling. | Tendency toward mode collapse; often produces repetitive textures. |
| **Architectural Adaptability** | Decoupled; accepts plug-and-play spatial adapters (ControlNet, LoRA). | Monolithic; requires full retraining for new conditioning domains. |
| **Inference Speed** | Multi-step iterative denoising (~8–12 seconds on T4 GPU). | Single forward generator pass (sub-second). |

#### Q23. What is Mode Collapse and why does Stable Diffusion avoid it?
- **Spoken Answer:** Mode collapse is a failure mode in GANs where the generator outputs only a few repetitive images that trick the discriminator; diffusion models avoid it because they optimize a stationary maximum likelihood objective across all data points.
- **Deep Explanation:** GANs train a generator against a dynamic discriminator in a non-cooperative game. If the generator discovers one specific building texture that consistently fools the discriminator, it collapses into generating only that texture. Diffusion models are trained to reverse a thermodynamic noise process across thousands of timesteps using mean squared error ($\|\epsilon - \epsilon_	heta\|^2$), which forces the network to model the full distribution density.
- **Project Context:** Chapter 6.1 of this handbook.

---

### Category J: System Architecture and Workflow Questions

#### Q24. Explain the lifecycle of an image upload in your system.
- **Spoken Answer:** User uploads sketch $	o$ browser creates instant object URL preview $	o$ POST request to Flask gateway $	o$ validation checks $	o$ async thread dispatches to Cloudflare tunnel $	o$ Colab server receives and authenticates $	o$ PIL pads on 512×512 white canvas $	o$ OpenCV Canny edge extraction $	o$ SD 1.5 + ControlNet 20-step inference on T4 GPU $	o$ lossless PNG streamed back $	o$ saved to disk $	o$ frontend polling receives complete state $	o$ DOM updates with side-by-side comparison.
- **Deep Explanation:** Documented in detail in Chapter 5.1 of this handbook.
- **Project Context:** Sequence diagram in Chapter 5.

#### Q25. Why do you paste the sketch onto a 512×512 pure white canvas?
- **Spoken Answer:** Because architectural sketches have variable aspect ratios, and Stable Diffusion requires an exact $512 	imes 512$ square tensor; pasting onto a centered white canvas preserves the building's natural proportions without squishing or stretching.
- **Deep Explanation:** Direct resizing (e.g., forcing a 16:9 widescreen sketch into $512 	imes 512$) distorts geometric aspect ratios, squashing two-storey buildings into squat boxes. `ImageOps.contain` scales the sketch proportionally so the largest side is 512, and placing it on a white `#FFFFFF` canvas ensures empty border regions produce zero Canny edges (since uniform white has zero gradient).
- **Project Context:** `colab_server.py` lines 119–121.

---

### Category K: APIs, HTTP, Authentication, and Remote Inference

#### Q26. Why do you use `hmac.compare_digest` instead of standard `==` for token checks?
- **Spoken Answer:** To prevent timing side-channel attacks where an attacker measures response latency to guess authentication tokens character by character.
- **Deep Explanation:** In standard Python `str1 == str2`, string evaluation exits immediately upon encountering the first mismatched character. An attacker with a precision timer can measure round-trip latency variations to determine if character 1, 2, or 3 was correct. `hmac.compare_digest` executes in strictly constant time regardless of where mismatches occur.
- **Project Context:** `backend/inference/colab_server.py` line 114.

#### Q27. What HTTP status codes are used across your REST API and what do they mean?
- **Spoken Answer:**
  - `HTTP 200 OK`: Successful health check, status query, or image delivery.
  - `HTTP 202 Accepted`: Generation request accepted and queued asynchronously.
  - `HTTP 400 Bad Request`: Missing file, unreadable image, or invalid style parameter.
  - `HTTP 401 Unauthorized`: Missing or incorrect Bearer token.
  - `HTTP 404 Not Found`: Expired or unknown job ID.
  - `HTTP 413 Payload Too Large`: Image file exceeds 8 MB or pixel count exceeds 40 MP.
  - `HTTP 415 Unsupported Media Type`: File uploaded is not a PNG or JPEG.
  - `HTTP 503 Service Unavailable`: Colab GPU worker model pipeline not loaded.
- **Project Context:** Implemented across `backend/app.py` and `backend/inference/colab_server.py`.

---

### Category L: Implementation-Specific Code Questions

#### Q28. What does `colab_server.set_pipeline(pipe)` do and why is it essential?
- **Spoken Answer:** It passes the already initialized in-memory diffusion pipeline from the Colab notebook directly to the Flask server thread, preventing the server from loading model weights a second time and crashing the GPU.
- **Deep Explanation:** When running a Jupyter notebook on Colab, cells 0–6 load SD 1.5 and ControlNet into GPU VRAM (occupying ~7 GB). If `colab_server.py` attempted to load the models again from disk, PyTorch would try to allocate an additional 7 GB of VRAM, exceeding the 15 GB ceiling of the Tesla T4 GPU and triggering a fatal `CUDA Out of Memory` runtime crash. `set_pipeline` reuses the active GPU memory pointer.
- **Project Context:** `backend/inference/colab_server.py` lines 64–70 and notebook Cell 12.

#### Q29. How does dynamic `.env` hot-reloading work in `app.py`?
- **Spoken Answer:** By attaching `_load_environment_file(ROOT / ".env")` to Flask's `@app.before_request` hook, the server re-reads the configuration before every request, allowing tunnel URL updates without restarting Flask.
- **Deep Explanation:** In `backend/app.py` lines 50–53, the `_refresh_env` function runs prior to dispatching any route handler. When a developer updates `FACADEFORGE_REMOTE_URL` in `.env` after a Colab tunnel restart, the change takes effect on the very next browser click without restarting the Python process.
- **Project Context:** `backend/app.py` lines 50–53.

---

### Category M: Testing and Troubleshooting Questions

#### Q30. How did you verify that the application does not substitute fake demonstration images?
- **Spoken Answer:** By executing negative-authentication tests and inspecting image hashes. When the bearer token is intentionally modified by one character, the application returns an explicit authentication error and produces zero image output.
- **Deep Explanation:** Preloaded portfolio images in `assets/demo/` have fixed MD5 checksums. Renders generated by the pipeline are written to `backend/generated/{uuid}.png` and feature unique stochastic textures, confirmable via timestamps and file size analysis. Furthermore, disconnecting Colab causes immediate `HTTP 502/504` errors rather than fallback visuals.
- **Project Context:** Tested via `test_remote.py` and documented in `README.md` Section 4.

---

### Category N: Limitations and Future Scope Questions

#### Q31. What is the single biggest limitation of the current implementation?
- **Spoken Answer:** The dependence on ephemeral Google Colab runtimes, which automatically disconnect after idle periods and issue a new Cloudflare tunnel URL on each restart.
- **Deep Explanation:** Because Colab is intended for interactive research rather than 24/7 production web hosting, its free GPU instances time out. A commercial production deployment requires migrating the inference microservice to dedicated serverless GPU infrastructure (such as RunPod, Modal, or AWS EC2 g4dn) with static DNS endpoints.
- **Project Context:** Chapter 11.2 and Chapter 12 of this handbook.

---

### Category O: Technical Decision-Making and Comparisons

#### Q32. Why did you choose DPMSolverMultistepScheduler over DDIM or Euler?
- **Spoken Answer:** DPMSolver++ is a specialized 2nd-order ODE solver that converges to photorealistic visual fidelity in only 20 steps, compared to 50 steps for DDIM or Euler, cutting inference latency by over 60%.
- **Deep Explanation:** Standard Euler-Maruyama discretization assumes locally constant gradients, leading to step accumulation errors unless step size $\Delta t$ is very small ($N \ge 50$). DPM-Solver analytically integrates the semi-linear component of diffusion ODEs and applies higher-order polynomial approximations to the score function, achieving higher structural fidelity at $N = 20$.
- **Project Context:** Cell 5 of `facade_forge.ipynb` and `colab_server.py` line 95.

---

### Category P: Challenging External Examiner Questions

#### Q33. If ControlNet conditions on edges, why doesn't the model just color the sketch like a flat painting? Where does the 3D realism come from?
- **Spoken Answer:** ControlNet injects spatial edge constraints into the intermediate layers of the UNet, but the UNet's self-attention and cross-attention layers retain the massive photorealistic prior of the base Stable Diffusion model, generating realistic volumetric light, shadows, and depth around those edges.
- **Deep Explanation:** ControlNet does not perform pixel replacement; it conditions the latent reverse diffusion process. The base UNet has been trained on hundreds of millions of natural photographs and understands how sunlight strikes building surfaces, how cast shadows fall from roof cantilevers, and how specular highlights reflect off glass windows. The Canny edge map merely anchors *where* the boundaries are; the base model supplies the *physics of light, material, and perspective*.
- **Project Context:** Section 6.3 of this handbook.

#### Q34. What happens if a user uploads a sketch with contradictory perspective lines?
- **Spoken Answer:** ControlNet strictly enforces the edge positions, while the base diffusion model attempts to synthesize plausible materials onto the warped geometry, often resulting in surreal or M.C. Escher-like architectural features.
- **Deep Explanation:** ControlNet treats edge maps as hard spatial conditioning signals. If an architect draws an impossible optical illusion or invalid two-point perspective, ControlNet still forces the UNet to place boundaries along those lines. The model cannot alter the underlying drawing geometry; it can only render materials conforming to the edges provided.
- **Project Context:** Section 11.2 of this handbook.

---

## 15. Tough Cross-Questioning & Hostile Examiner Scenarios

Examiners often employ cross-questioning to test whether students truly built the application or simply followed a generic online tutorial. Prepare these precise defenses:

### Scenario 1: "Why should anyone use your system when they can just type a prompt into Midjourney or DALL-E 3?"
- **The Trap:** The examiner suggests your project is redundant because commercial image generators already exist.
- **The Defense:**  
  *"Midjourney and DALL-E 3 are unconstrained text-to-image foundation models. They lack coordinate-level spatial conditioning. If an architect uploads a sketch of a building with three second-floor windows, a cantilevered entrance, and a 45-degree butterfly roof, Midjourney will generate an entirely different building that ignores the exact window spacing and structural boundaries.*  
  *Architects cannot present hallucinated geometry to clients because that geometry must eventually be built. FacadeForge AI uses ControlNet Canny edge guidance, which mathematically locks the generative process to the architect's exact blueprint lines, preserving floor counts, window positions, and structural proportions while synthesizing realistic materials."*

### Scenario 2: "Does your system guarantee 100% exact architectural preservation?"
- **The Trap:** If you claim "Yes, 100% exact", the examiner will find a tiny curved line that changed slightly. If you say "No", they may claim the project failed.
- **The Defense:**  
  *"Our system enforces strict **spatial geometric compliance**, not pixel-identical recreation. ControlNet Canny guidance ensures that floor lines, window boundaries, roof angles, and structural openings match the sketch with high precision.*  
  *However, diffusion models are stochastic generative models, not deterministic CAD plotters. Minor micro-details—such as the exact density of window mullion subdivisions or the subtle organic texture of masonry—are synthesized probabilistically based on the chosen architectural style prompt. This is why our workspace explicitly displays: 'ControlNet guides composition; structural details may vary. Preliminary concept visualization only.' This strikes the exact balance required for conceptual design."*

### Scenario 3: "What if the user uploads a blurry photo of a cat or an unreadable doodle?"
- **The Trap:** The examiner tests your edge-case handling and error boundaries.
- **The Defense:**  
  *"The system operates on multi-stage validation. First, the Flask gateway validates file format and decompression limits. If a user uploads a photo of a cat, the Canny edge detector will extract the contours of the cat, and the diffusion model—conditioned by our strong architectural base prompts ('Photorealistic architectural exterior visualization...')—will attempt to interpret the cat's contour as an avant-garde organic architectural pavilion.*  
  *However, if the upload is a faint pencil smudge below the Canny low threshold ($T_{low}=50$), zero edges are detected, resulting in an empty control map where the model relies purely on text prompts. In future iterations, we plan to implement a lightweight CLIP-based architectural classification gate that rejects non-architectural inputs before passing them to the GPU."*

### Scenario 4: "What happens if Colab disconnects midway through generation?"
- **The Trap:** Testing system resilience against distributed network failures.
- **The Defense:**  
  *"The local gateway is engineered defensively against remote dropouts. The network request in `remote_client.py` is wrapped in socket and HTTP timeout handlers with a strict timeout ceiling. If Colab disconnects, the socket closes, or the Cloudflare tunnel returns an HTTP 530 error, `remote_client.py` intercepts the exception and raises a user-safe `RemoteInferenceError`.*  
  *The background worker thread updates the job state to `state: 'error'` with a clear message: 'Could not reach Colab. The runtime may have disconnected or the Quick Tunnel URL may have changed.' The frontend polling engine catches this error state, turns the status dot red, and displays the exact remediation steps to the user without crashing the Flask web server."*

### Scenario 5: "What did your team actually build versus what did open-source libraries provide?"
- **The Trap:** The examiner implies you only wrote a 5-line wrapper script around Hugging Face.
- **The Defense:**  
  *"Open-source libraries provided the raw neural network weights and mathematical primitives (PyTorch, Diffusers, OpenCV). Our team engineered the complete distributed software system that makes these models functional for end users:*  
  *1) We designed the distributed hybrid architecture that bridges local consumer hardware to cloud GPUs across secure Cloudflare tunnels.*  
  *2) We developed the authenticated remote microservice (`colab_server.py`) featuring memory-efficient pipeline reuse (`set_pipeline`) that prevents GPU Out-Of-Memory crashes.*  
  *3) We built the multi-stage image preprocessing pipeline (`ImageOps.contain` + canvas aspect centering + tuned Canny thresholds).*  
  *4) We developed the asynchronous local web gateway (`backend/app.py`) with thread-safe job management, preventing browser timeouts.*  
  *5) We built the complete custom frontend studio from scratch in semantic HTML5 and CSS3, implementing real-time client validation, polling engines, and side-by-side comparative visualization.*  
  *6) We engineered the curated architectural prompt engineering and negative prompt matrices that enable distinct material disentanglement across five styles."*

---

## 16. Quick Revision Sheet & Last-Minute Cheat Sheet

Keep this section open 10 minutes before your viva examination for rapid mental alignment.

### 16.1 The 60-Second Elevator Pitch
> *"FacadeForge AI is a distributed generative AI platform that transforms 2D architectural building facade sketches into photorealistic exterior visualizations in approximately 10 seconds. Traditional architectural rendering requires days of manual 3D modeling and ray-tracing. We solve this by combining Stable Diffusion v1.5 with ControlNet Canny edge conditioning. Our system takes a sketch, extracts structural contours using tuned Canny edge detection (thresholds 50/150), and denoises latent representations in 20 steps using DPMSolver++ on a cloud-hosted Nvidia Tesla T4 GPU. A local Python/Flask gateway orchestrates asynchronous jobs across an encrypted Cloudflare Quick Tunnel, presenting the user with an architectural comparison studio where floor counts, window placements, and structural geometry are rigorously preserved across five curated design styles."*

### 16.2 The Complete Workflow in 10 Bullets
1. User uploads a PNG/JPG sketch into the workspace drop zone.
2. Client-side JS validates file size (<8MB) and instantly renders a local object preview.
3. User selects one of five architectural styles and clicks "Generate facade".
4. Flask gateway checks image safety (<40 MP), assigns a UUID4, and queues the job asynchronously.
5. Background worker thread dispatches an authorized HTTPS request across the Cloudflare tunnel to Google Colab.
6. Colab server validates the HMAC-SHA256 bearer token in constant time.
7. Sketch is scaled with aspect preservation and pasted onto a 512×512 white canvas; Canny edge detection extracts contours.
8. SD 1.5 + ControlNet Canny executes 20 denoising steps using DPMSolver++ in `torch.float16` on the Tesla T4 GPU.
9. Colab streams the synthesized PNG back through the tunnel; Flask saves it to `backend/generated/{uuid}.png`.
10. Frontend polling detects `complete` status, excises the empty placeholder, and renders the side-by-side comparison with a download link.

### 16.3 Technology Matrix Cheat Sheet
| Category | Technology | Version / Model ID | Exact Project Role |
|---|---|---|---|
| **Language** | Python | 3.10+ | Local gateway and remote inference microservice. |
| **Web Gateway** | Flask | 3.0.x | WSGI routing, asynchronous thread dispatch, asset delivery. |
| **Generative Model** | Stable Diffusion | `runwayml/stable-diffusion-v1-5` | Latent diffusion backbone synthesizing textures, light, materials. |
| **Conditioning** | ControlNet Canny | `lllyasviel/sd-controlnet-canny` | Injects zero-convolution spatial edge priors into the UNet. |
| **Vision Library** | OpenCV | 4.10+ (`cv2`) | Grayscale conversion and Canny edge extraction ($T_{low}=50, T_{high}=150$). |
| **Image Toolkit** | Pillow | 10.x+ (`PIL`) | Canvas padding (`ImageOps.contain`) and decompression bomb defense. |
| **Orchestration** | Diffusers | 0.30+ | `StableDiffusionControlNetPipeline`, attention slicing. |
| **Scheduler** | DPM-Solver++ | `DPMSolverMultistepScheduler` | 20-step high-order numerical ODE solver. |
| **Cloud GPU** | Nvidia Tesla T4 | 16 GB GDDR6 VRAM | Remote CUDA hardware acceleration in `torch.float16`. |
| **Tunneling** | Cloudflare | `cloudflared` Quick Tunnel | Outbound encrypted TLS reverse proxy to `trycloudflare.com`. |
| **Security** | Python `hmac` | SHA-256 | Timing-safe bearer token verification (`compare_digest`). |
| **Frontend** | Vanilla Web | HTML5 / CSS3 / ES6 JS | Custom architectural editorial UI, polling loop, comparison view. |

### 16.4 Core Hyperparameters Summary
- **Input / Output Dimensions:** $512 	imes 512$ pixels.
- **Latent Tensor Dimensions:** $64 	imes 64 	imes 4$ channels.
- **Inference Steps:** 20 steps (DPMSolver++).
- **Classifier-Free Guidance (CFG):** 7.5.
- **ControlNet Conditioning Scale:** 1.0 (strict edge enforcement).
- **Canny Edge Thresholds:** $T_{low} = 50$, $T_{high} = 150$ (1:3 optimal ratio).
- **Execution Precision:** `torch.float16` (Half Precision).
- **Max Upload Limit:** 8 MB / 40,000,000 pixels.
- **Polling Frequency:** 1,200 ms interval.

### 16.5 Five Major Project Strengths
1. **Authentic Structural Preservation:** ControlNet Canny prevents the random geometry warping typical of pure text-to-image models.
2. **Zero-Cost High-Performance Architecture:** Leverages free Colab T4 GPUs via Cloudflare tunnels, making enterprise AI accessible without server fees.
3. **Responsive Non-Blocking UI:** Asynchronous job queues and polling ensure the web interface never hangs during multi-second AI inference.
4. **Resilient Security Design:** HMAC timing-safe bearer authentication and dynamic `.env` hot-reloading protect public tunnel endpoints.
5. **No Synthetic Demonstration Fallbacks:** The platform strictly displays genuine GPU model inference, upholding academic and production integrity.

### 16.6 Five Major Limitations & Future Fixes
1. **Colab Session Timeouts:** Addressed in future work by containerized serverless GPU hosting (RunPod / Modal).
2. **Tunnel URL Changes on Restart:** Addressed in future work by custom domain routing via persistent Cloudflare named tunnels.
3. **Single Concurrency Bottleneck:** Addressed in future work by multi-GPU worker pools behind a Celery/Redis queue.
4. **2D Flat Depth Ambiguity:** Addressed in future work by dual-conditioning Multi-ControlNet (Canny + MiDaS Depth).
5. **Native 512px Resolution:** Addressed in future work by Real-ESRGAN super-resolution upscaling to 4K.

### 16.7 Last-Minute Pre-Viva Checklist
- [ ] Confirm Colab notebook `facade_forge.ipynb` has run through Cell 12.
- [ ] Confirm Cloudflare tunnel printed active `https://*.trycloudflare.com` URL.
- [ ] Confirm local `.env` has matching URL and matching `FACADEFORGE_REMOTE_TOKEN`.
- [ ] Confirm local Flask server is running at `http://127.0.0.1:5000`.
- [ ] Open `/workspace` and confirm green status: *"Colab is connected..."*
- [ ] Execute one test generation with a sample sketch to verify live end-to-end functionality.
- [ ] Remember your role's specific defense script from Chapter 13.
- [ ] Keep calm, be technically specific, and speak with confidence!
