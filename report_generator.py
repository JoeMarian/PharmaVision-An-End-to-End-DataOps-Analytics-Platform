from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import pandas as pd

def generate_pdf(path, title, kpis={}, text_lines=[]):
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 18)
    c.drawString(1*inch, height - 1*inch, title)
    c.setFont("Helvetica", 12)
    y = height - 1.5*inch
    for k, v in kpis.items():
        c.drawString(1*inch, y, f"{k}: {v}")
        y -= 0.3*inch
    y -= 0.1*inch
    for line in text_lines:
        if y < 1*inch:
            c.showPage()
            y = height - 1*inch
        c.drawString(1*inch, y, str(line))
        y -= 0.25*inch
    c.save()
