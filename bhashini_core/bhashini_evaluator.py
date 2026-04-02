import os
import sys
import time
import psutil
import pandas as pd
import jiwer
import re

# ==========================================
# PATH SETUP & IMPORTING THE PLATTER WRAPPER
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLATTER_DIR = os.path.join(BASE_DIR, "PLATTER")
sys.path.append(PLATTER_DIR)

from platter_wrapper import load_platter_model, run_platter

PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "results", "ocr_comparison_results.csv")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "results", "predictions")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "images")
GT_DIR = os.path.join(PROJECT_ROOT, "dataset", "ground_truth")

# ==========================================
# HELPERS
# ==========================================
def normalize_text(text):
    """Unicode-safe normalization (Hindi-friendly)."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_metrics(gt, pred):
    """Calculates CER and WER."""
    if pred.startswith("ERROR") or pred == "EMPTY":
        return 1.0, 1.0

    gt_clean = normalize_text(gt)
    pred_clean = normalize_text(pred)

    if len(gt_clean) == 0:
        return 1.0, 1.0

    try:
        cer = jiwer.cer(gt_clean, pred_clean)
        wer = jiwer.wer(gt_clean, pred_clean)
        return min(cer, 1.0), min(wer, 1.0)
    except:
        return 1.0, 1.0

# ==========================================
# MAIN EVALUATION LOOP
# ==========================================
def run_bhashini_evaluation():
    print("\n🚀 Starting Phase 2: Bhashini (PLATTER) Evaluation...")

    # 1. Load CSV
    if not os.path.exists(CSV_PATH):
        print(f"❌ ERROR: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    # 2. Add columns if missing
    new_columns = [
        'PLATTER_CER',
        'PLATTER_WER',
        'PLATTER_Time (s)',
        'PLATTER_RAM (MB)',
        'PLATTER_CPU (%)',
        'PLATTER_Confidence'
    ]

    for col in new_columns:
        if col not in df.columns:
            df[col] = 0.0

    # 3. Load Model
    platter_ai = load_platter_model("crnn")
    if platter_ai is None:
        print("❌ CRITICAL: Model failed to load")
        return

    platter_metrics = []

    # 4. Loop through dataset
    for index, row in df.iterrows():
        image_name = row['Image_Name']

        if image_name == "AVERAGE":
            continue

        # Caching: Skip already processed
        # Using a safer check in case CER actually equals exactly 0.0
        if row['PLATTER_Time (s)'] > 0.0: 
            print(f"⏭️ Skipping {image_name} (already processed)")
            continue

        print(f"🔄 Processing: {image_name}")

        image_path = os.path.join(DATASET_DIR, image_name)
        gt_path = os.path.join(GT_DIR, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        # File safety check
        if not os.path.exists(image_path) or not os.path.exists(gt_path):
            print(f"⚠️ Missing file for {image_name}, skipping...")
            continue

        # Load Ground Truth
        with open(gt_path, 'r', encoding='utf-8') as f:
            ground_truth = f.read()

        # =============================
        # PERFORMANCE TRACKING
        # =============================
        process = psutil.Process(os.getpid())
        start_ram = process.memory_info().rss / (1024 * 1024)

        # Prime the CPU monitor instantly
        psutil.cpu_percent(interval=None)

        text, time_taken, confidence = run_platter(image_path, platter_ai)
        
        # Get the average CPU usage since we primed it instantly
        cpu_usage = psutil.cpu_percent(interval=None)

        end_ram = process.memory_info().rss / (1024 * 1024)
        ram_used = max(0, end_ram - start_ram)

        # =============================
        # METRICS
        # =============================
        cer, wer = calculate_metrics(ground_truth, text)

        # =============================
        # UPDATE DATAFRAME
        # =============================
        df.at[index, 'PLATTER_CER'] = round(cer, 3)
        df.at[index, 'PLATTER_WER'] = round(wer, 3)
        df.at[index, 'PLATTER_Time (s)'] = round(time_taken, 2)
        df.at[index, 'PLATTER_RAM (MB)'] = round(ram_used, 2)
        df.at[index, 'PLATTER_CPU (%)'] = round(cpu_usage, 2)
        df.at[index, 'PLATTER_Confidence'] = round(confidence, 3)

        # =============================
        # SAVE PREDICTION
        # =============================
        img_folder = os.path.join(PREDICTIONS_DIR, image_name.split('.')[0])
        os.makedirs(img_folder, exist_ok=True)

        with open(os.path.join(img_folder, "platter_pred.txt"), "w", encoding="utf-8") as f:
            f.write(text)

        platter_metrics.append([cer, wer, time_taken, ram_used, cpu_usage])

    # 5. Compute Averages
    if len(platter_metrics) == 0:
        print("❌ No valid results to compute averages.")
        return

    print("\n📊 Calculating Averages...")

    avg_cer = sum(x[0] for x in platter_metrics) / len(platter_metrics)
    avg_wer = sum(x[1] for x in platter_metrics) / len(platter_metrics)
    avg_time = sum(x[2] for x in platter_metrics) / len(platter_metrics)
    avg_ram = sum(x[3] for x in platter_metrics) / len(platter_metrics)
    avg_cpu = sum(x[4] for x in platter_metrics) / len(platter_metrics)

    # Safe AVERAGE row handling
    avg_rows = df.index[df['Image_Name'] == 'AVERAGE'].tolist()

    if avg_rows:
        avg_index = avg_rows[0]
    else:
        avg_index = len(df)
        df.loc[avg_index] = ['AVERAGE'] + [0]*(len(df.columns)-1)

    df.at[avg_index, 'PLATTER_CER'] = round(avg_cer, 3)
    df.at[avg_index, 'PLATTER_WER'] = round(avg_wer, 3)
    df.at[avg_index, 'PLATTER_Time (s)'] = round(avg_time, 2)
    df.at[avg_index, 'PLATTER_RAM (MB)'] = round(avg_ram, 2)
    df.at[avg_index, 'PLATTER_CPU (%)'] = round(avg_cpu, 2)

    # 6. Save CSV
    df.to_csv(CSV_PATH, index=False)

    print(f"\n✅ DONE: PLATTER results appended successfully to CSV")

# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    run_bhashini_evaluation()