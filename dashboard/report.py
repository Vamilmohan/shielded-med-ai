from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def generate_pdf(prediction, confidence, image_path, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 800, "Shielded Med-AI Diagnostic Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 760, f"Prediction: {prediction}")
    c.drawString(50, 740, f"Confidence: {confidence:.2f}%")
    c.drawString(50, 720, "Method: Federated Learning + Grad-CAM")

    c.drawImage(image_path, 50, 420, width=300, height=300)

    c.drawString(50, 380, "Privacy Notice:")
    c.drawString(50, 360, "- No raw medical data was shared")
    c.drawString(50, 340, "- Federated model aggregation used")

    c.save()
