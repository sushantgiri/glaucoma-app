# backend/report_pdf.py
import os
from io import BytesIO
from typing import Optional, Tuple

from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib import colors


def _wrap_text(c: canvas.Canvas, text: str, font_name: str, font_size: int, max_width: float):
    """
    Simple word-wrap helper: returns list of wrapped lines that fit max_width.
    """
    c.setFont(font_name, font_size)
    words = (text or "").replace("\n", " ").split()
    lines = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if c.stringWidth(trial, font_name, font_size) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _pil_to_reader(pil_img: PILImage.Image) -> ImageReader:
    """
    Convert PIL image to ReportLab ImageReader without saving to disk.
    """
    bio = BytesIO()
    pil_img.save(bio, format="PNG")
    bio.seek(0)
    return ImageReader(bio)


def generate_report_pdf_like_template(
    *,
    image_id: str,
    classification: str,
    confidence: float,
    explanation: str,
    final_assessment: str,
    disclaimer: str,
    original_pil: PILImage.Image,
    gradcam_path: Optional[str],
) -> bytes:
    """
    Generates a 1-page PDF laid out like your screenshot:
    - Left: original + gradcam side-by-side
    - Right: header + explanation + final assessment + disclaimer
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    page_w, page_h = letter  # 612 x 792

    # Page margins
    margin = 36  # 0.5 inch
    top = page_h - margin
    left = margin
    right = page_w - margin
    bottom = margin

    # Outer border (like screenshot)
    c.setStrokeColor(colors.black)
    c.setLineWidth(1)
    c.rect(margin - 10, margin - 10, page_w - 2 * (margin - 10), page_h - 2 * (margin - 10), stroke=1, fill=0)

    # Layout zones
    # Left image block
    img_block_w = 260
    img_block_h = 200
    img_block_x = left + 10
    img_block_y = top - 90 - img_block_h  # place below header area

    # Right text block
    text_x = img_block_x + img_block_w + 25
    text_w = right - text_x - 10
    text_top_y = top - 30

    # Header (top-right)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(text_x, top - 10, f"Image ID: {image_id}")
    c.drawString(text_x, top - 25, f"Classification (Model): {classification}")
    c.drawString(text_x, top - 40, f"Confidence Score: {confidence:.3f}")

    # Images: original + gradcam side-by-side inside left block
    # We will fit two squares into img_block_w
    gap = 10
    single_w = (img_block_w - gap) / 2
    single_h = img_block_h

    # Convert original PIL for drawing
    orig = original_pil.convert("RGB")
    orig_reader = _pil_to_reader(orig)

    # Load gradcam image if available, else use a blank placeholder
    if gradcam_path and os.path.exists(gradcam_path):
        grad_pil = PILImage.open(gradcam_path).convert("RGB")
    else:
        grad_pil = PILImage.new("RGB", (512, 512), (240, 240, 240))
    grad_reader = _pil_to_reader(grad_pil)

    c.drawImage(orig_reader, img_block_x, img_block_y, width=single_w, height=single_h, preserveAspectRatio=True, anchor='sw')
    c.drawImage(grad_reader, img_block_x + single_w + gap, img_block_y, width=single_w, height=single_h, preserveAspectRatio=True, anchor='sw')

    # Explanation label + paragraph
    y = text_top_y - 80
    c.setFont("Helvetica-Bold", 10)
    c.drawString(text_x, y, "Explanation:")
    y -= 14

    body_font = "Helvetica"
    body_size = 9
    lines = _wrap_text(c, explanation, body_font, body_size, text_w)

    c.setFont(body_font, body_size)
    line_h = 12
    for line in lines:
        if y < bottom + 80:  # keep space for final assessment + disclaimer
            break
        c.drawString(text_x + 5, y, line)
        y -= line_h

    # Final Assessment
    y -= 10
    c.setFont("Helvetica-Bold", 10)
    c.drawString(text_x, y, "Final Assessment:")
    c.setFont("Helvetica", 10)
    c.drawString(text_x + 95, y, final_assessment)

    # Disclaimer
    y -= 22
    c.setFont("Helvetica-Oblique", 9)
    disc_lines = _wrap_text(c, disclaimer, "Helvetica-Oblique", 9, text_w)
    for line in disc_lines:
        if y < bottom:
            break
        c.drawString(text_x, y, line)
        y -= 11

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()
