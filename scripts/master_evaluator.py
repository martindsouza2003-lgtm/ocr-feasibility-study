import os
import glob
import pandas as pd
import sys
import gc
import time

# Ensure Python can find custom scripts
sys.path.append(os.path.dirname(__file__))

# Imports
from metrics_tracker import calculate_error_rates, get_current_ram_mb, ResourceMonitor, normalize_text
from nextgen_engines import load_trocr_model, run_trocr

try:
    from ocr_engines import run_tesseract, run_easyocr, run_paddleocr
    LEGACY_ENGINES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not load legacy engines ({e}). Running TrOCR only.")
    LEGACY_ENGINES_AVAILABLE = False

# ==========================================
# PATH CONFIG
# ==========================================
DATASET_IMG_DIR = "dataset/images"
DATASET_GT_DIR = "dataset/ground_truth"
RESULTS_DIR = "results"
PREDICTIONS_DIR = "results/predictions"

# ==========================================
# FILE HELPERS
# ==========================================
def read_ground_truth(image_filename):
    base_name = os.path.splitext(image_filename)[0]
    txt_path = os.path.join(DATASET_GT_DIR, f"{base_name}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None

def save_prediction_text(image_basename, model_name, text):
    folder_path = os.path.join(PREDICTIONS_DIR, image_basename)
    os.makedirs(folder_path, exist_ok=True)
    
    file_path = os.path.join(folder_path, f"{model_name}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(text))

# ==========================================
# MODEL EVALUATION WRAPPER
# ==========================================
def evaluate_model(model_name, func, gt_text, *args):
    print(f"   -> Running {model_name}...")
    
    monitor = ResourceMonitor()
    ram_before = get_current_ram_mb()

    monitor.start()
    time.sleep(0.05)

    try:
        predicted_text, exec_time = func(*args)
    except Exception as e:
        print(f"      ❌ Error in {model_name}: {e}")
        predicted_text, exec_time = "ERROR", 0.0

    avg_cpu = monitor.stop()
    ram_after = get_current_ram_mb()
    ram_delta = max(0.0, ram_after - ram_before)

    # Normalize ONLY for metrics
    gt_clean = normalize_text(gt_text)
    pred_clean = normalize_text(predicted_text)

    cer, wer = calculate_error_rates(gt_clean, pred_clean)

    gc.collect()

    return {
        f"{model_name}_CER": cer,
        f"{model_name}_WER": wer,
        f"{model_name}_Time": round(exec_time, 2),
        f"{model_name}_RAM_MB": round(ram_delta, 2),
        f"{model_name}_CPU_%": avg_cpu
    }, predicted_text, pred_clean  # ✅ Return BOTH raw + clean

# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    print("=" * 60)
    print("🚀 STARTING OCR EVALUATION PIPELINE")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(DATASET_IMG_DIR, "*.*")))

    # ✅ Filter only valid image types
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths = [p for p in image_paths if p.lower().endswith(valid_ext)]

    if not image_paths:
        print("❌ No valid images found")
        return

    trocr_processor, trocr_model = load_trocr_model()
    all_results = []

    for idx, img_path in enumerate(image_paths):
        img_filename = os.path.basename(img_path)
        base_name = os.path.splitext(img_filename)[0]

        print(f"\n📄 Processing: {img_filename}")

        gt_text = read_ground_truth(img_filename)
        if not gt_text:
            print("   ⚠️ Skipping (no GT)")
            continue

        row_data = {"Image_Name": img_filename}

        # ---------------- TrOCR ----------------
        result, raw_pred, _ = evaluate_model(
            "TrOCR", run_trocr, gt_text,
            img_path, trocr_processor, trocr_model
        )
        row_data.update(result)
        save_prediction_text(base_name, "TrOCR", raw_pred)

        if LEGACY_ENGINES_AVAILABLE:

            # -------- Tesseract Raw --------
            result, raw_pred, _ = evaluate_model(
                "Tess_Raw", run_tesseract, gt_text,
                img_path, False
            )
            row_data.update(result)
            save_prediction_text(base_name, "Tess_Raw", raw_pred)

            # -------- Tesseract Processed --------
            result, raw_pred, _ = evaluate_model(
                "Tess_Proc", run_tesseract, gt_text,
                img_path, True
            )
            row_data.update(result)
            save_prediction_text(base_name, "Tess_Proc", raw_pred)

            # -------- EasyOCR --------
            result, raw_pred, _ = evaluate_model(
                "EasyOCR", run_easyocr, gt_text,
                img_path
            )
            row_data.update(result)
            save_prediction_text(base_name, "EasyOCR", raw_pred)

            # -------- PaddleOCR --------
            try:
                result, raw_pred, _ = evaluate_model(
                    "PaddleOCR", run_paddleocr, gt_text,
                    img_path
                )
                row_data.update(result)
                save_prediction_text(base_name, "PaddleOCR", raw_pred)
            except Exception as e:
                print(f"   ⚠️ PaddleOCR failed: {e}")
                row_data.update({
                    "PaddleOCR_CER": 100.0,
                    "PaddleOCR_WER": 100.0,
                    "PaddleOCR_Time": 0.0,
                    "PaddleOCR_RAM_MB": 0.0,
                    "PaddleOCR_CPU_%": 0.0
                })
                save_prediction_text(base_name, "PaddleOCR", "ERROR")

        all_results.append(row_data)

    print("\n" + "=" * 60)
    print("💾 SAVING RESULTS")

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "ocr_comparison_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"✅ Saved metrics to: {csv_path}")
    print(f"✅ Saved raw text outputs to: {PREDICTIONS_DIR}")
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n📊 FINAL SUMMARY (AVERAGES)")
    models = ["TrOCR", "Tess_Raw", "Tess_Proc", "EasyOCR", "PaddleOCR"]
    
    for model in models:
        if f"{model}_CER" in df.columns:
            avg_cer = df[f"{model}_CER"].mean()
            avg_time = df[f"{model}_Time"].mean()
            avg_ram = df[f"{model}_RAM_MB"].mean()
            avg_cpu = df[f"{model}_CPU_%"].mean()

            print(
                f"🔹 {model.ljust(10)} "
                f"→ CER: {avg_cer:>6.2f}% | "
                f"Time: {avg_time:>5.2f}s | "
                f"RAM: {avg_ram:>6.2f} MB | "
                f"CPU: {avg_cpu:>5.2f}%"
            )

# ==========================================
if __name__ == "__main__":
    main()

    