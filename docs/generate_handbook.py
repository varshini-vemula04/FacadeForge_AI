"""Generator for FacadeForge AI Complete Project Documentation & Viva Preparation Handbook.
Generates:
  1. docs/FacadeForge_AI_Viva_Handbook.md
  2. docs/FacadeForge_AI_Viva_Handbook.pdf
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
MD_PATH = DOCS_DIR / "FacadeForge_AI_Viva_Handbook.md"
PDF_PATH = DOCS_DIR / "FacadeForge_AI_Viva_Handbook.pdf"

print(f"Project root: {ROOT}")
print(f"Output Markdown path: {MD_PATH}")
print(f"Output PDF path: {PDF_PATH}")
