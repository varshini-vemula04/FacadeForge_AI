# -*- coding: utf-8 -*-
"""Master builder script for FacadeForge AI Viva Preparation Handbook.
Generates:
  1. docs/FacadeForge_AI_Viva_Handbook.md
  2. docs/FacadeForge_AI_Viva_Handbook.pdf
"""

import os
import re
import sys
from pathlib import Path

# Add docs to path so we can import section modules
DOCS_DIR = Path(__file__).resolve().parent
ROOT_DIR = DOCS_DIR.parent
sys.path.insert(0, str(DOCS_DIR))

from generate_handbook_md import build_handbook_md
from handbook_sections_1_5 import SECTION_1_TO_5
from handbook_sections_6_8 import SECTION_6_TO_8
from handbook_sections_9_13 import SECTION_9_TO_13
from handbook_sections_14_16 import SECTION_14_TO_16

import reportlab
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

MD_OUTPUT_PATH = DOCS_DIR / "FacadeForge_AI_Viva_Handbook.md"
PDF_OUTPUT_PATH = DOCS_DIR / "FacadeForge_AI_Viva_Handbook.pdf"


def assemble_markdown() -> str:
    """Concatenate all chapters into a single comprehensive Markdown handbook."""
    md_content = (
        build_handbook_md()
        + "\n"
        + SECTION_1_TO_5
        + "\n"
        + SECTION_6_TO_8
        + "\n"
        + SECTION_9_TO_13
        + "\n"
        + SECTION_14_TO_16
    )
    MD_OUTPUT_PATH.write_text(md_content, encoding="utf-8")
    print(f"Successfully generated Markdown handbook: {MD_OUTPUT_PATH}")
    print(f"Total Markdown file size: {len(md_content):,} characters / {len(md_content.splitlines()):,} lines")
    return md_content


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic total page counting and running headers/footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        # Omit headers and footers on the cover page (Page 1)
        if self._pageNumber == 1:
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#777067"))

        # Running Header
        self.drawString(54, 752, "FacadeForge AI — Complete Project Documentation & Viva Preparation Handbook")
        self.setStrokeColor(colors.HexColor("#ded9d1"))
        self.setLineWidth(0.5)
        self.line(54, 744, 558, 744)

        # Running Footer
        self.line(54, 46, 558, 46)
        self.drawString(54, 34, "FacadeForge AI | Academic Viva Voce & Technical Defense Handbook")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(558, 34, page_str)

        self.restoreState()


def clean_text_for_xml(text: str) -> str:
    """Escape XML entities for ReportLab Paragraphs."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    # Convert markdown bold and italics to HTML tags
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<font face='Courier' color='#aa775a'><b>\1</b></font>", text)
    return text


def build_pdf(md_text: str):
    """Compile the comprehensive PDF handbook from the structured content."""
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom palette definitions
    C_INK = colors.HexColor("#211a17")
    C_BRONZE = colors.HexColor("#aa775a")
    C_MUTED = colors.HexColor("#777067")
    C_LINE = colors.HexColor("#ded9d1")
    C_BG_CODE = colors.HexColor("#f8f6f2")
    C_BG_TABLE = colors.HexColor("#faf9f6")

    # Typography styles
    styles.add(ParagraphStyle(
        name="DocTitle",
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=C_INK,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="DocSubtitle",
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=C_BRONZE,
        spaceAfter=15,
    ))
    styles.add(ParagraphStyle(
        name="MetaLabel",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=C_INK,
    ))
    styles.add(ParagraphStyle(
        name="MetaValue",
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=C_MUTED,
    ))
    styles.add(ParagraphStyle(
        name="H1",
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=C_INK,
        spaceBefore=18,
        spaceAfter=8,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="H2",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=C_BRONZE,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="H3",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=C_INK,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="H4",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=C_BRONZE,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="Body",
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=C_INK,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="BulletItem",
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=C_INK,
        leftIndent=15,
        spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="CodeBlock",
        fontName="Courier",
        fontSize=7.5,
        leading=9.5,
        textColor=C_INK,
    ))
    styles.add(ParagraphStyle(
        name="TableHeader",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        name="TableCell",
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=C_INK,
    ))
    styles.add(ParagraphStyle(
        name="CalloutText",
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=C_INK,
    ))

    story = []

    # Parse lines of markdown into flowables
    lines = md_text.splitlines()
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []

    for line in lines:
        stripped = line.strip()

        # Handle Code Blocks
        if stripped.startswith("```"):
            if in_code_block:
                # End code block
                in_code_block = False
                code_text = "\n".join(code_lines)
                code_lines = []
                # Render code box as a single-cell Table
                p = Paragraph(clean_text_for_xml(code_text).replace("\n", "<br/>"), styles["CodeBlock"])
                t = Table([[p]], colWidths=[504])
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), C_BG_CODE),
                    ("BOX", (0, 0), (-1, -1), 0.5, C_LINE),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(t)
                story.append(Spacer(1, 6))
            else:
                # Start code block
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        # Handle Markdown Tables
        if "|" in line and (line.startswith("|") or line.endswith("|")):
            # Check if this is a separator line like |---|---|
            if re.match(r"^\|?(\s*:?-+:?\s*\|)+\s*$", stripped):
                continue
            cols = [c.strip() for c in line.split("|")]
            if cols[0] == "":
                cols = cols[1:]
            if cols and cols[-1] == "":
                cols = cols[:-1]
            if cols:
                in_table = True
                table_rows.append(cols)
                continue

        if in_table and (not line.strip() or "|" not in line):
            # Flush accumulated table
            in_table = False
            if table_rows:
                # Build Table Flowable
                num_cols = max(len(r) for r in table_rows)
                # Compute widths proportionally
                col_width = 504 / num_cols
                formatted_data = []
                for row_idx, row in enumerate(table_rows):
                    row_cells = []
                    # Pad if short
                    while len(row) < num_cols:
                        row.append("")
                    for c_idx, cell in enumerate(row):
                        cell_style = styles["TableHeader"] if row_idx == 0 else styles["TableCell"]
                        row_cells.append(Paragraph(clean_text_for_xml(cell), cell_style))
                    formatted_data.append(row_cells)

                t = Table(formatted_data, colWidths=[col_width] * num_cols)
                t_style = [
                    ("BACKGROUND", (0, 0), (-1, 0), C_INK),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("BOX", (0, 0), (-1, -1), 0.5, C_LINE),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, C_LINE),
                ]
                # Alternating row background
                for r_idx in range(1, len(formatted_data)):
                    if r_idx % 2 == 1:
                        t_style.append(("BACKGROUND", (0, r_idx), (-1, r_idx), C_BG_TABLE))
                t.setStyle(TableStyle(t_style))
                story.append(t)
                story.append(Spacer(1, 6))
            table_rows = []

        if not stripped:
            continue

        # Horizontal rule
        if stripped in {"---", "***", "___"}:
            story.append(HRFlowable(width="100%", thickness=0.5, color=C_LINE, spaceBefore=8, spaceAfter=8))
            continue

        # Headings
        if stripped.startswith("# "):
            title_text = stripped[2:].strip()
            story.append(Paragraph(clean_text_for_xml(title_text), styles["DocTitle"]))
            continue
        elif stripped.startswith("## "):
            h1_text = stripped[3:].strip()
            # If start of major chapter, add page break
            if any(h1_text.startswith(f"{i}.") for i in range(1, 17)):
                story.append(PageBreak())
            story.append(Paragraph(clean_text_for_xml(h1_text), styles["H1"]))
            continue
        elif stripped.startswith("### "):
            story.append(Paragraph(clean_text_for_xml(stripped[4:].strip()), styles["H2"]))
            continue
        elif stripped.startswith("#### "):
            story.append(Paragraph(clean_text_for_xml(stripped[5:].strip()), styles["H3"]))
            continue
        elif stripped.startswith("##### "):
            story.append(Paragraph(clean_text_for_xml(stripped[6:].strip()), styles["H4"]))
            continue

        # Blockquote / Callout
        if stripped.startswith("> "):
            callout = clean_text_for_xml(stripped[2:].strip())
            p = Paragraph(f"<b>Notice / Key Concept:</b> {callout}", styles["CalloutText"])
            t = Table([[p]], colWidths=[504])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbf9f5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBEFORE", (0, 0), (0, -1), 3, C_BRONZE),
                ("BOX", (0, 0), (-1, -1), 0.5, C_LINE),
            ]))
            story.append(t)
            story.append(Spacer(1, 6))
            continue

        # Bullet List Items
        if stripped.startswith("- ") or stripped.startswith("* "):
            item_text = clean_text_for_xml(stripped[2:].strip())
            story.append(Paragraph(f"&bull; &nbsp; {item_text}", styles["BulletItem"]))
            continue

        # Numbered List Items
        num_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if num_match:
            n_num, n_text = num_match.groups()
            story.append(Paragraph(f"<b>{n_num}.</b> &nbsp; {clean_text_for_xml(n_text)}", styles["BulletItem"]))
            continue

        # Standard Body Paragraph
        story.append(Paragraph(clean_text_for_xml(stripped), styles["Body"]))

    # Final table flush if document ended with a table
    if table_rows:
        num_cols = max(len(r) for r in table_rows)
        col_width = 504 / num_cols
        formatted_data = []
        for row_idx, row in enumerate(table_rows):
            row_cells = []
            while len(row) < num_cols:
                row.append("")
            for c_idx, cell in enumerate(row):
                cell_style = styles["TableHeader"] if row_idx == 0 else styles["TableCell"]
                row_cells.append(Paragraph(clean_text_for_xml(cell), cell_style))
            formatted_data.append(row_cells)
        t = Table(formatted_data, colWidths=[col_width] * num_cols)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_INK),
            ("BOX", (0, 0), (-1, -1), 0.5, C_LINE),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, C_LINE),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)

    # Build the document using the dynamic NumberedCanvas
    print("Compiling PDF with ReportLab NumberedCanvas...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully compiled PDF handbook: {PDF_OUTPUT_PATH}")
    print(f"PDF file size: {PDF_OUTPUT_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    print("=" * 60)
    print("FACADEFORGE AI: COMPILING COMPLETE VIVA & TECHNICAL HANDBOOK")
    print("=" * 60)
    md_text = assemble_markdown()
    build_pdf(md_text)
    print("=" * 60)
    print("HANDBOOK BUILD COMPLETE!")
    print(f"1. Markdown Document: {MD_OUTPUT_PATH}")
    print(f"2. Polished PDF File: {PDF_OUTPUT_PATH}")
    print("=" * 60)
