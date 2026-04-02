import jiwer
import psutil
import os
import threading
import time
import re
import unicodedata

# ==========================================
# PHASE 5: TEXT NORMALIZATION
# ==========================================
def normalize_text(text):
    """
    Cleans OCR output AND Ground Truth for fair metric comparison.
    """
    if not isinstance(text, str) or text == "ERROR":
        return text
        
    text = text.replace('\u200d', '').replace('\u200c', '')
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# ==========================================
# PHASE 6: METRICS (CER & WER)
# ==========================================
def calculate_error_rates(ground_truth, predicted_text):
    """
    Calculates CER & WER (%) after normalization.
    Caps max error at 100.0% for clean business reporting.
    """
    # Normalize FIRST (CRITICAL)
    ground_truth = normalize_text(ground_truth)
    predicted_text = normalize_text(predicted_text)

    # If the ground truth is totally empty, there's nothing to score
    if not ground_truth or not ground_truth.strip():
        return 0.0, 0.0

    # If the model completely failed
    if predicted_text == "ERROR" or not predicted_text.strip():
        return 100.0, 100.0

    try:
        cer = jiwer.cer(ground_truth, predicted_text)
        wer = jiwer.wer(ground_truth, predicted_text)

        # Convert to % and cap at 100 for clean reporting
        cer = min(cer * 100.0, 100.0)
        wer = min(wer * 100.0, 100.0)

        return round(cer, 2), round(wer, 2)

    except Exception as e:
        print(f"⚠️ Metric Error: {e}")
        return 100.0, 100.0

# ==========================================
# PHASE 7: PERFORMANCE MEASUREMENT
# ==========================================
def get_current_ram_mb():
    """Returns the current RAM usage of this Python process in Megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

class ResourceMonitor:
    """
    A daemon background thread class to measure CPU usage DURING model execution.
    """
    def __init__(self):
        self.keep_measuring = True
        self.cpu_measurements = []
        self.thread = None

    def _measure_cpu(self):
        # Using interval=None ensures non-blocking measurement between sleep cycles
        while self.keep_measuring:
            cpu = psutil.cpu_percent(interval=None)
            self.cpu_measurements.append(cpu)
            time.sleep(0.1)

    def start(self):
        self.keep_measuring = True
        self.cpu_measurements = []
        # Prime the psutil cpu_percent tracker
        psutil.cpu_percent(interval=None) 
        
        # ✅ Added user's Daemon optimization
        self.thread = threading.Thread(
            target=self._measure_cpu,
            daemon=True
        )
        self.thread.start()

    def stop(self):
        self.keep_measuring = False
        if self.thread is not None:
            self.thread.join()
        
        # Remove the first measurement (often 0.0 from priming)
        if len(self.cpu_measurements) > 1:
            valid_measurements = self.cpu_measurements[1:]
            return round(sum(valid_measurements) / len(valid_measurements), 2)
        elif self.cpu_measurements:
            return round(self.cpu_measurements[0], 2)
        return 0.0

        # ==========================================
# ISOLATED TEST
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("🧪 TESTING PHASE 6 & 7 INFRASTRUCTURE (UPDATED)")
    print("="*50)

    # 1. Test Metrics
    gt = "मेरा नाम जेमिनी है" 
    pred = "मेरा नाम जेमिनी ही" 
    
    cer, wer = calculate_error_rates(gt, pred)
    print(f"📊 Metric Test -> CER: {cer}% | WER: {wer}%")

    # 2. Test Hardware Monitoring
    print("\n🖥️ Hardware Test -> Simulating heavy workload for 2 seconds...")
    
    ram_before = get_current_ram_mb()
    monitor = ResourceMonitor()
    
    monitor.start()
    
    time.sleep(2) 
    
    avg_cpu = monitor.stop()
    ram_after = get_current_ram_mb()
    ram_delta = max(0, ram_after - ram_before) # Your brilliant fix!

    print(f"💾 RAM Consumed: {ram_delta:.2f} MB")
    print(f"⚙️ Avg CPU Usage: {avg_cpu}%")
    print("="*50)