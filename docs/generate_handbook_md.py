# -*- coding: utf-8 -*-
"""Script to generate FacadeForge_AI_Viva_Handbook.md."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "FacadeForge_AI_Viva_Handbook.md"

def build_handbook_md():
    parts = []
    
    # Title and Metadata
    parts.append("""# FacadeForge AI — Complete Project Documentation & Viva Preparation Handbook
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
""")
    return "".join(parts)

if __name__ == "__main__":
    content = build_handbook_md()
    print("Base content generated length:", len(content))
