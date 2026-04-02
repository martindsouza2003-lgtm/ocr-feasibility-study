import os
import sys
import time
import psutil
import subprocess
import pandas as pd
import jiwer
import re
import shutil

# ==========================================
# PATH SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

CSV_PATH = os.path.join(PROJECT_ROOT, "results", "ocr_comparison_results.csv")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "results", "predictions")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "images")
GT_DIR = os.path.join(PROJECT_ROOT, "dataset", "ground_truth")

BHAASHA_DIR = os.path.join(BASE_DIR, "BhaashaHWOCR")
TEMP_OUT_DIR = os.path.join(BASE_DIR, "bhaasha_temp_out")

# ==========================================
# HELPERS
# ==========================================
def normalize_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_metrics(gt, pred):
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

def print_summary_table(df):
    print("\n📊 FINAL SUMMARY (AVERAGES)")
    
    avg_row = df[df['Image_Name'] == 'AVERAGE'].iloc[0]

    engines = [
        ("TrOCR", "TrOCR"),
        ("Tess_Raw", "Tess_Raw"),
        ("Tess_Proc", "Tess_Proc"),
        ("EasyOCR", "EasyOCR"),
        ("PaddleOCR", "PaddleOCR"),
        ("PLATTER", "PLATTER"),
        ("Bhaasha", "Bhaasha")
    ]

    for display_name, prefix in engines:
        if f"{prefix}_CER" in df.columns:
            cer = avg_row[f"{prefix}_CER"] * 100
            
            time_col = f"{prefix}_Time (s)" if f"{prefix}_Time (s)" in df.columns else f"{prefix}_Time"
            ram_col = f"{prefix}_RAM (MB)" if f"{prefix}_RAM (MB)" in df.columns else f"{prefix}_RAM_MB"
            cpu_col = f"{prefix}_CPU (%)" if f"{prefix}_CPU (%)" in df.columns else f"{prefix}_CPU_%"

            t = avg_row.get(time_col, 0)
            ram = avg_row.get(ram_col, 0)
            cpu = avg_row.get(cpu_col, 0)

            if cer == 0 and t == 0:
                continue

            print(f"🔹 {display_name:<10} → CER: {cer:>6.2f}% | Time: {t:>5.2f}s | RAM: {ram:>6.2f} MB | CPU: {cpu:>5.2f}%")
    print("\n")

# ==========================================
# MAIN EVALUATION
# ==========================================
def run_bhaasha_evaluation():
    print("\n🚀 Starting Bhaasha Evaluation...")

    if not os.path.exists(CSV_PATH):
        print("❌ CSV missing")
        return

    df = pd.read_csv(CSV_PATH)

    new_columns = [
        'Bhaasha_CER', 'Bhaasha_WER',
        'Bhaasha_Time (s)', 'Bhaasha_RAM (MB)',
        'Bhaasha_CPU (%)', 'Bhaasha_Confidence'
    ]

    for col in new_columns:
        if col not in df.columns:
            df[col] = 0.0

    os.makedirs(TEMP_OUT_DIR, exist_ok=True)

    bhaasha_metrics = []
    fail_count = 0

    for index, row in df.iterrows():
        image_name = row['Image_Name']

        if image_name == "AVERAGE":
            continue

        # Smart caching
        if row['Bhaasha_Time (s)'] > 0.0 and row['Bhaasha_CER'] < 1.0:
            print(f"⏭️ Skipping {image_name}")
            bhaasha_metrics.append([
                row['Bhaasha_CER'],
                row['Bhaasha_WER'],
                row['Bhaasha_Time (s)'],
                row['Bhaasha_RAM (MB)'],
                row['Bhaasha_CPU (%)']
            ])
            continue

        print(f"\n🔄 {image_name}")

        image_path = os.path.join(DATASET_DIR, image_name)
        gt_path = os.path.join(GT_DIR, image_name.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt"))

        if not os.path.exists(image_path) or not os.path.exists(gt_path):
            print("⚠️ Missing file")
            continue

        with open(gt_path, 'r', encoding='utf-8') as f:
            gt = f.read()

        cmd = [
            "python", 
            "infer.py",
            "--pretrained", "weights/hindi",
            "--image_path", image_path,
            "--out_dir", TEMP_OUT_DIR
        ]

        start_time = time.time()
        pred_text = ""  # FIXED: Emptied this so it doesn't trigger the ERROR block

        process = subprocess.Popen(
            cmd,
            cwd=BHAASHA_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        max_ram = 0
        cpu_readings = []

        try:
            ps_proc = psutil.Process(process.pid)

            while process.poll() is None:
                if time.time() - start_time > 120:
                    process.kill()
                    pred_text = "ERROR: TIMEOUT"
                    fail_count += 1
                    break

                try:
                    max_ram = max(max_ram, ps_proc.memory_info().rss / (1024 * 1024))
                    cpu_readings.append(ps_proc.cpu_percent(interval=0.1))
                except psutil.NoSuchProcess:
                    break

            _, err = process.communicate()

        except Exception as e:
            process.kill()
            pred_text = f"ERROR: {str(e)}"
            fail_count += 1

        exec_time = time.time() - start_time
        avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0.0

        if process.returncode != 0 and not pred_text.startswith("ERROR"):
            msg = err.decode("utf-8")[:200] if err else "Unknown"
            print(f"❌ Error: {msg}")
            pred_text = "ERROR: SUBPROCESS_FAILED"
            fail_count += 1

        if not pred_text.startswith("ERROR"):
            ocr_file = os.path.join(TEMP_OUT_DIR, "ocr.txt")
            if os.path.exists(ocr_file):
                with open(ocr_file, "r", encoding="utf-8") as f:
                    pred_text = f.read().strip()
            else:
                pred_text = "ERROR: NO_OUTPUT"

        cer, wer = calculate_metrics(gt, pred_text)

        df.at[index, 'Bhaasha_CER'] = round(cer, 3)
        df.at[index, 'Bhaasha_WER'] = round(wer, 3)
        df.at[index, 'Bhaasha_Time (s)'] = round(exec_time, 2)
        df.at[index, 'Bhaasha_RAM (MB)'] = round(max_ram, 2)
        df.at[index, 'Bhaasha_CPU (%)'] = round(avg_cpu, 2)
        df.at[index, 'Bhaasha_Confidence'] = 0.0  # placeholder

        # Save output
        img_folder = os.path.join(PREDICTIONS_DIR, image_name.split('.')[0])
        os.makedirs(img_folder, exist_ok=True)

        with open(os.path.join(img_folder, "bhaashahwocr_pred.txt"), "w", encoding="utf-8") as f:
            f.write(pred_text)

        bhaasha_metrics.append([cer, wer, exec_time, max_ram, avg_cpu])

        # Cleanup
        for f in os.listdir(TEMP_OUT_DIR):
            p = os.path.join(TEMP_OUT_DIR, f)
            try:
                if os.path.isfile(p):
                    os.remove(p)
                else:
                    shutil.rmtree(p)
            except:
                pass

        if fail_count > 5:
            print("❌ Too many failures")
            break

    if len(bhaasha_metrics) == 0:
        print("❌ No results")
        print_summary_table(df)
        return

    avg = list(zip(*bhaasha_metrics))
    avg_vals = [sum(x)/len(x) for x in avg]

    avg_row_idx = df.index[df['Image_Name'] == 'AVERAGE'].tolist()
    if avg_row_idx:
        idx = avg_row_idx[0]
    else:
        idx = len(df)
        df.loc[idx] = ['AVERAGE'] + [0]*(len(df.columns)-1)

    df.at[idx, 'Bhaasha_CER'] = round(avg_vals[0], 3)
    df.at[idx, 'Bhaasha_WER'] = round(avg_vals[1], 3)
    df.at[idx, 'Bhaasha_Time (s)'] = round(avg_vals[2], 2)
    df.at[idx, 'Bhaasha_RAM (MB)'] = round(avg_vals[3], 2)
    df.at[idx, 'Bhaasha_CPU (%)'] = round(avg_vals[4], 2)

    df.to_csv(CSV_PATH, index=False)

    print("\n✅ Bhaasha Evaluation Complete")
    print_summary_table(df)

if __name__ == "__main__":
    run_bhaasha_evaluation()