# -*- coding: utf-8 -*-
"""Chapters 6 to 8 for FacadeForge AI Viva Preparation Handbook."""

SECTION_6_TO_8 = """
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
2. **The Reverse Diffusion Process ($p_\theta$):** A learned neural network (UNet) predicts and subtracts the noise at each step, recovering the clean data distribution from pure noise.

#### Forward Diffusion (The Markov Chain of Noise)
Given an initial data point $x_0 \sim q(x)$, the forward process adds Gaussian noise according to a predefined variance schedule $\beta_1, \beta_2, \dots, \beta_T$:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$$

A key mathematical property of the Gaussian forward process is that **any intermediate state $x_t$ can be sampled in closed form directly from $x_0$ without evaluating all $t-1$ preceding steps**. Defining $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$:
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
As $t \to T$, $\bar{\alpha}_t \to 0$, and $x_T$ becomes virtually indistinguishable from pure Gaussian noise $\mathcal{N}(0, \mathbf{I})$.

#### Reverse Diffusion (Learned Denoising)
The reverse trajectory $q(x_{t-1} | x_t)$ is intractable because it depends on the entire data distribution $q(x_0)$. Therefore, we train a neural network parameterized by weights $\theta$ to approximate the reverse distribution:
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

In practice, rather than directly predicting the mean $\mu_\theta$, the model is parameterized to predict the **exact noise vector $\epsilon_\theta(x_t, t)$** that was added to $x_0$. The simplified training objective (derived by Ho et al., 2020) is:
$$\mathcal{L}_{simple}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

---

### 6.2 Latent Diffusion Models (LDM) & Architectural Mechanics

#### The Computational Bottleneck of Pixel-Space Diffusion
Standard Denoising Diffusion Probabilistic Models (DDPM) operate directly in RGB pixel space ($512 \times 512 \times 3 = 786,432$ variables per step). Running multi-step neural network convolutions across 786k continuous pixel variables is computationally prohibitive, requiring multi-GPU server clusters and several minutes per image.

#### The LDM Breakthrough: Perceptual vs. Semantic Compression
Rombach et al. (2022) observed that digital images contain two forms of information:
1. **Imperceptible high-frequency details** (subtle pixel variations, compression noise).
2. **Semantic conceptual information** (spatial geometry, objects, lighting, architectural elements).

**Latent Diffusion Models (LDM)**—the foundation of Stable Diffusion—decouple image generation into two specialized stages:
1. **Perceptual Compression (The VAE):** A pretrained Variational Autoencoder (AutoencoderKL) compresses the high-resolution RGB image $x \in \mathbb{R}^{3 \times 512 \times 512}$ into a lower-dimensional continuous latent space $z = \mathcal{E}(x) \in \mathbb{R}^{4 \times 64 \times 64}$.
   $$\text{Compression Factor: } f = \frac{512}{64} = 8$$
   $$\text{Dimensionality Reduction: } \frac{3 \times 512 \times 512}{4 \times 64 \times 64} = \frac{786,432}{16,384} = 48\times \text{ fewer tensor elements}$$
2. **Generative Modeling (The Latent UNet):** The diffusion forward and reverse processes operate exclusively on the compact latent tensor $z$, speeding up inference by over $40\times$ while allowing the UNet to focus on semantic composition rather than imperceptible pixel noise.
3. **Synthesis (The VAE Decoder):** Once the latent denoising process completes, yielding clean latent $z_0$, the VAE decoder reconstructs the full-resolution RGB image: $x_{rec} = \mathcal{D}(z_0)$.

#### Core Sub-Networks in Stable Diffusion v1.5
Stable Diffusion v1.5 consists of three distinct deep neural networks:

| Component | Sub-Network | Parameters | Function in FacadeForge AI |
|---|---|---|---|
| **Text Encoder** | CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) | ~123 Million | Tokenizes text prompts into a sequence of 77 tokens and maps them into a 768-dimensional semantic embedding space. |
| **Diffusion Backbone** | Time-Conditioned Cross-Attention UNet | ~860 Million | Predicts the latent noise $\epsilon_\theta(z_t, t, c)$ at each timestep conditioned on text and spatial ControlNet features. |
| **Autoencoder** | AutoencoderKL (VAE Encoder & Decoder) | ~84 Million | Compresses $512 \times 512 \times 3$ RGB inputs into $64 \times 64 \times 4$ latents and decodes synthesized latents back to photorealistic RGB pixels. |

---

### 6.3 ControlNet: Structural Conditioning via Zero Convolutions

#### The Limitation of Text Prompts in Architecture
In architectural rendering, text prompts alone cannot guarantee geometric fidelity. A prompt such as *"A two-storey modern residence with three second-floor windows"* does not specify coordinates, perspective, cantilever angles, or facade boundaries. The base UNet has no mechanism to anchor pixels to exact physical lines.

#### ControlNet Architecture (Zhang & Agrawala, 2023)
ControlNet solves this by creating a structure-aware neural sidecar that locks the base model and attaches a trainable spatial adapter:

1. **Locking the Base Model:** The original Stable Diffusion UNet weights are frozen ($\Theta_{frozen}$). This guarantees that the massive visual and material knowledge base learned during pretraining (millions of natural images, textures, daylight distributions) is never degraded or catastrophically forgotten.
2. **The Trainable Clone:** ControlNet creates an exact architectural replica of the 12 encoding blocks and 1 middle block of the UNet ($\Theta_{trainable}$).
3. **Zero Convolutions ($\mathcal{Z}$):** The trainable copy is connected to the frozen model using specialized $1 \times 1$ convolutional layers whose weight matrices and bias vectors are initialized to **identically zero**:
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
$$G_\sigma(x, y) = \frac{1}{2\pi \sigma^2} \exp\left( -\frac{x^2 + y^2}{2\sigma^2} \right)$$
$$I_\sigma(x, y) = I(x, y) * G_\sigma(x, y)$$
In OpenCV's `cv2.Canny`, an optimized $5 \times 5$ Gaussian kernel is applied internally.

#### Stage 2: Gradient Magnitude and Orientation Calculation
The smoothed image is filtered with horizontal ($K_x$) and vertical ($K_y$) Sobel operator kernels:
$$K_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \quad K_y = \begin{bmatrix} +1 & +2 & +1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{bmatrix}$$
The directional spatial derivatives are computed:
$$G_x = K_x * I_\sigma, \quad G_y = K_y * I_\sigma$$
From these, the edge gradient magnitude $G(x, y)$ and edge direction angle $\theta(x, y)$ are derived:
$$G(x, y) = \sqrt{G_x(x, y)^2 + G_y(x, y)^2}$$
$$\theta(x, y) = \arctan\left( \frac{G_y(x, y)}{G_x(x, y)} \right)$$
The gradient direction angle is rounded to one of four fundamental directions: $0^\circ$ (horizontal), $45^\circ$ (positive diagonal), $90^\circ$ (vertical), or $135^\circ$ (negative diagonal).

#### Stage 3: Non-Maximum Suppression (Edge Thinning)
Gradient calculation produces thick ridges around lines. Non-Maximum Suppression (NMS) thins these ridges into sharp, 1-pixel-wide lines:
1. For each pixel $(x, y)$, compare its gradient magnitude $G(x, y)$ against its two immediate neighbors along the gradient vector direction $\theta(x, y)$.
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
1. **Unconditioned Evaluation:** $\epsilon_\theta(z_t, t, \emptyset)$ (empty text prompt $\emptyset$).
2. **Conditioned Evaluation:** $\epsilon_\theta(z_t, t, c)$ (conditioned on architectural prompt $c$).

The final modified noise vector is computed by extrapolating in the direction of the condition:
$$\tilde{\epsilon}_\theta(z_t, t, c) = \epsilon_\theta(z_t, t, \emptyset) + s \cdot \left[ \epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset) \right]$$
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
  - Confirms the binary begins with the PNG magic byte signature (`\x89PNG\r\n\x1a\n`).
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
"""
