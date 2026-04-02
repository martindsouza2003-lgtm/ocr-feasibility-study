import os
import glob
import html
import pandas as pd
from jiwer import cer
import sys

# Tells Python to look inside the scripts folder for your normalizer!
sys.path.append("scripts") 
from metrics_tracker import normalize_text 

# ==========================================
# DYNAMIC MODEL CONFIGURATION
# ==========================================
# To add new models in the future, just add them to this list!
MODELS = [
    {"id": "Tess_Raw", "name": "Tesseract (Raw)", "file": "Tess_Raw.txt"},
    {"id": "Tess_Proc", "name": "Tesseract (Processed)", "file": "Tess_Proc.txt"},
    {"id": "EasyOCR", "name": "EasyOCR", "file": "EasyOCR.txt"},
    {"id": "PaddleOCR", "name": "PaddleOCR", "file": "PaddleOCR.txt"},
    {"id": "TrOCR", "name": "Microsoft TrOCR", "file": "TrOCR.txt"},
    {"id": "PLATTER", "name": "IIT PLATTER (Indic)", "file": "platter_pred.txt"},
    {"id": "Bhaasha", "name": "BhaashaHWOCR (Indic)", "file": "bhaashahwocr_pred.txt"} # Pre-loaded for tomorrow!
]

# ==========================================
# HTML TEMPLATE
# ==========================================
def create_html_template(image_name, ground_truth, predictions_dict):

    def safe_text(text):
        if pd.isna(text) or text is None or str(text).strip() == "":
            return "⚠️ No Output"
        if str(text).strip().upper() == "ERROR":
            return "❌ OCR Failed"
        return str(text)

    def get_similarity(gt, pred):
        try:
            gt_clean = normalize_text(gt)
            pred_clean = normalize_text(pred)
            return max(0.0, 1 - cer(gt_clean, pred_clean))
        except:
            return 0.0

    def get_style(score):
        if score > 0.85:
            return "border-green-300 bg-green-50 text-green-900"
        elif score > 0.5:
            return "border-yellow-300 bg-yellow-50 text-yellow-900"
        else:
            return "border-gray-200 bg-gray-50 text-gray-800"

    gt_raw = safe_text(ground_truth)
    safe_image_name = html.escape(image_name)
    safe_gt = html.escape(gt_raw)

    # Start HTML String
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>OCR Audit - {safe_image_name}</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 font-sans p-8">

        <div class="max-w-4xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden">
            
            <div class="bg-slate-800 text-white p-6">
                <h1 class="text-2xl font-bold">Visual OCR Audit</h1>
                <p class="text-slate-300 text-sm mt-1">File: {safe_image_name}</p>
            </div>

            <div class="p-6">

                <h2 class="text-xs font-bold text-gray-400 uppercase mb-2">Original Handwriting</h2>
                <div class="flex justify-center bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg p-2 mb-8">
                    <img src="../dataset/images/{image_name}" class="max-h-80 object-contain rounded" alt="Handwritten Sample">
                </div>

                <h2 class="text-xs font-bold text-green-600 uppercase mb-2">✅ Ground Truth</h2>
                <div class="bg-green-50 border-l-4 border-green-500 p-4 mb-8 whitespace-pre-wrap break-words">
                    {safe_gt}
                </div>

                <h2 class="text-xs font-bold text-gray-400 uppercase mb-4">🤖 OCR Outputs</h2>
                <div class="space-y-4">
    """

    # Dynamically generate HTML blocks for every model in our list
    for model in MODELS:
        pred_raw = safe_text(predictions_dict.get(model["id"], ""))
        
        # If it says "No Output" and it's Bhaasha, we know we just haven't run it yet, so display slightly differently
        if pred_raw == "⚠️ No Output" and model["id"] == "Bhaasha":
             score = 0.0
             display_text = "⏳ Pending Integration..."
        else:
             score = get_similarity(gt_raw, pred_raw)
             display_text = html.escape(pred_raw)

        style = get_style(score)

        html_content += f"""
                    <div class="border rounded-lg p-4 {style}">
                        <div class="flex justify-between">
                            <span class="text-xs font-bold uppercase">{model['name']}</span>
                            <span class="text-xs font-semibold">{round(score*100,1)}% Similarity</span>
                        </div>
                        <p class="whitespace-pre-wrap break-words mt-2">{display_text}</p>
                    </div>
        """

    # Close HTML string
    html_content += """
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return html_content


# ==========================================
# FILE READ HELPER
# ==========================================
def read_text_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


# ==========================================
# MAIN GENERATOR
# ==========================================
def build_audit_reports():
    print("🚀 Starting Visual Audit Generation...")

    img_dir = "dataset/images"
    gt_dir = "dataset/ground_truth"
    predictions_dir = "results/predictions"
    out_dir = "audit_reports"

    os.makedirs(out_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths = [p for p in image_paths if p.lower().endswith(valid_ext)]

    dashboard_links = []

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]

        print(f"📄 Processing {img_name}...")

        # -------- Ground Truth --------
        gt_text = read_text_file(os.path.join(gt_dir, f"{base_name}.txt"))

        # -------- Load All Predictions Dynamically --------
        pred_folder = os.path.join(predictions_dir, base_name)
        predictions_dict = {}
        
        if os.path.exists(pred_folder):
            for model in MODELS:
                file_path = os.path.join(pred_folder, model["file"])
                predictions_dict[model["id"]] = read_text_file(file_path)

        # -------- Generate HTML --------
        html_output = create_html_template(
            image_name=img_name,
            ground_truth=gt_text,
            predictions_dict=predictions_dict
        )

        report_filename = f"report_{base_name}.html"
        with open(os.path.join(out_dir, report_filename), "w", encoding="utf-8") as f:
            f.write(html_output)

        dashboard_links.append((img_name, report_filename))

    # -------- Dashboard --------
    print("🏗️ Building Dashboard...")

    dashboard_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>OCR Audit Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 p-8 font-sans">
        <div class="max-w-3xl mx-auto bg-white rounded-xl shadow p-8">
            <h1 class="text-3xl font-bold mb-6">OCR Feasibility Audit - Dashboard</h1>
            <ul class="space-y-3">
    """

    for i, (img_name, link) in enumerate(dashboard_links, 1):
        dashboard_html += f"""
        <li>
            <a href="{link}" class="text-blue-600 font-medium hover:underline p-3 border rounded block bg-gray-50 hover:bg-blue-50 transition">
                #{i} 📂 {html.escape(img_name)}
            </a>
        </li>
        """

    dashboard_html += """
            </ul>
        </div>
    </body>
    </html>
    """

    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(dashboard_html)

    print("✅ Done! Open audit_reports/index.html in your web browser!")

if __name__ == "__main__":
    build_audit_reports()