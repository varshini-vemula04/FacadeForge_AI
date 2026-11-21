# FacadeForge AI

FacadeForge has a local Flask website and a separate Colab inference server. Windows only handles the interface, uploads, job status, and downloads. Stable Diffusion 1.5 + Canny ControlNet run on the Colab GPU; the local app does not import or start the local diffusion adapter.

The three images in `assets/demo` are portfolio examples only. Generation results are saved only after Colab returns a valid PNG for the current upload.

## 1. Start the existing model in Google Colab

1. Open `facade_forge.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or another available CUDA GPU).
3. Run the notebook's existing setup and model-loading cells through the cell that creates `pipe`. Confirm its output says both models loaded successfully. This uses the notebook's existing model cache and in-memory `pipe`; do not start a second pipeline.
4. The server needs Flask. If `import flask` fails in Colab, run this small server-only install cell (it does not install or replace PyTorch):

   ```python
   %pip install -q "Flask>=3.0,<4.0"
   ```

5. In the Colab file sidebar, upload `backend/inference/colab_server.py` from this project to `/content/colab_server.py`.
6. Generate a shared secret in a local PowerShell window (do not put the value in source code or commit it):

   ```powershell
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

7. In Colab, enter the shared secret securely, register the pipeline, and start the API safely (this is also provided as the final cells in `facade_forge.ipynb`):

   ```python
   import getpass
   import os
   import sys

   os.environ["FACADEFORGE_REMOTE_TOKEN"] = getpass.getpass("FacadeForge shared token: ")
   sys.path.insert(0, "/content")

   import colab_server
   colab_server.set_pipeline(pipe)  # Reuse the notebook pipeline already on the T4.
   colab_server.start_server(host="0.0.0.0", port=7860)  # Prevents accidental duplicate port binding
   ```

   Keep the Colab runtime and this server thread running while you use the website. The endpoints require the shared bearer token. `/health` does not disclose it.

If you instead start `colab_server.py` without calling `set_pipeline(pipe)`, it lazily loads SD 1.5 and Canny ControlNet once from the current Colab Hugging Face cache. It uses `local_files_only=True` and reports an error if the models are not cached; it does not download weights.

## 2. Expose the Colab server with a Cloudflare Quick Tunnel

In another Colab cell, install the small `cloudflared` tunnel client and start a temporary tunnel to port 7860:

```bash
!curl --location --output cloudflared.deb "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"
!dpkg -i cloudflared.deb
!cloudflared tunnel --url http://127.0.0.1:7860
```

Cloudflare prints a URL such as `https://random-words.trycloudflare.com`. Copy that URL. A Quick Tunnel is temporary, needs no paid plan or Cloudflare account, and stops working when its process or Colab runtime stops; a later tunnel gets a different URL. Keep the tunnel cell running. [Cloudflare Quick Tunnel documentation](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/) · [Cloudflare download instructions](https://developers.cloudflare.com/tunnel/downloads/)

The tunnel URL is public, but both API routes still require the shared token. Treat the token as a password and do not share it or paste it into screenshots.

## 3. Connect the Windows website to Colab

From the project root in PowerShell, install only the small local website requirements. **Do not install `requirements-ai.txt` on Windows**; it is the legacy local inference dependency list and is not used by this remote setup.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

In `.env`, set the tunnel URL printed by Cloudflare and the exact same token entered in Colab:

```dotenv
FACADEFORGE_REMOTE_URL=https://random-words.trycloudflare.com
FACADEFORGE_REMOTE_TOKEN=paste-the-generated-shared-secret-here
FACADEFORGE_REMOTE_TIMEOUT_SECONDS=300
```

`.env` is excluded from Git. Do not commit it. Restart Flask after changing these values:

```powershell
python backend\app.py
```

Open <http://127.0.0.1:5000/workspace>. The status message checks the authenticated Colab `/health` endpoint. If Colab or the Quick Tunnel restarts, replace the URL in `.env` and restart Flask.

## 4. Verify actual image generation

1. Confirm the workspace status says Colab and its GPU are connected. If it says disconnected, check the Colab server cell, tunnel cell, URL, and matching token.
2. Upload a PNG or JPG facade sketch, choose a style, then select **Generate facade**.
3. The workspace should show that it sent the sketch to Colab, then display a side-by-side original and returned image when the request completes. The request is asynchronous on the local website; the inference call may take several minutes.
4. Use **Download image**. Confirm the file is a PNG and is not one of the preloaded files in `assets/demo`.
5. For a negative-auth check, temporarily change one character of the local token and refresh the workspace. It must show a token error and generation must not return an image. Restore the matching token afterward.

The Colab `/generate` endpoint accepts multipart `sketch` and `style` fields and returns the model output directly as `image/png`. It matches the notebook's 512×512 white canvas, OpenCV Canny thresholds 50/150, 20 denoising steps, guidance 7.5, and ControlNet scale 1.0. The selected style changes the prompt. ControlNet guides structure but does not guarantee exact architectural preservation.

## Failure behavior

- Missing local URL or token: the website reports that Colab is not configured.
- Bad token: the website reports an authentication error.
- Colab or tunnel offline / URL changed: the website reports that it cannot reach Colab.
- Slow or interrupted generation: the website reports a timeout or closed connection; increase `FACADEFORGE_REMOTE_TIMEOUT_SECONDS` if needed (maximum 900 seconds).
- Model cache/GPU not ready: Colab returns an explicit service error. The local server never substitutes a demo image.

Keep the project pages and style options in the local Flask app; the Colab server is only the authenticated inference API. The original notebook is unchanged.
