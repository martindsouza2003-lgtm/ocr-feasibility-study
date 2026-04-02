import cv2
import pytesseract
import time
import os
import easyocr
from paddleocr import PaddleOCR
from preprocessing import preprocess_for_tesseract

# ==========================================
# 🧠 MODEL INITIALIZATION (Load once into RAM)
# ==========================================
print("Loading EasyOCR Model...")
easyocr_reader = easyocr.Reader(['hi', 'en'], gpu=False, verbose=False)

print("Loading PaddleOCR Model...")
# show_log=False stops Paddle from printing 100 lines of debug text
# use_angle_cls=True helps if the handwriting is slightly tilted

paddle_reader = PaddleOCR(use_textline_orientation=True, lang='hi', enable_mkldnn=False)
print("Models Loaded! Starting showdown...\n")

# ==========================================
# 1. TESSERACT FUNCTION (Your Masterpiece)
# ==========================================
def run_tesseract(image_path, apply_preprocessing=False):
    try:
        if not isinstance(image_path, str):
            return "ERROR", 0.0

        if apply_preprocessing:
            img = preprocess_for_tesseract(image_path)
        else:
            img = cv2.imread(image_path)

        if img is None:
            return "ERROR", 0.0

        custom_config = r'--oem 3 --psm 6 --dpi 300'

        start_time = time.time()
        text = pytesseract.image_to_string(img, lang='hin+eng', config=custom_config)
        execution_time = time.time() - start_time

        clean_text = " ".join(text.split())

        if clean_text == "" or clean_text.replace(" ", "") == "":
            return "ERROR", execution_time

        return clean_text, execution_time

    except Exception:
        return "ERROR", 0.0

# ==========================================
# 2. EASYOCR FUNCTION (Your Masterpiece)
# ==========================================
def run_easyocr(image_path):
    try:
        if not isinstance(image_path, str):
            return "ERROR", 0.0

        img = cv2.imread(image_path)
        if img is None:
            return "ERROR", 0.0

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        results = easyocr_reader.readtext(img)
        execution_time = time.time() - start_time

        results = sorted(results, key=lambda x: x[0][0][1])
        filtered = [text for (bbox, text, prob) in results if prob > 0.5]
        
        if not filtered:
            filtered = [text for (bbox, text, prob) in results]

        extracted_text = " ".join(filtered)
        clean_text = " ".join(extracted_text.split())

        if clean_text == "" or clean_text.replace(" ", "") == "":
            return "ERROR", execution_time

        return clean_text, execution_time

    except Exception:
        return "ERROR", 0.0

# ==========================================
# 3. PADDLEOCR FUNCTION (NEW)
# ==========================================
# ==========================================
# 3. PADDLEOCR FUNCTION (CRASH-PROOF V3.0)
# ==========================================
def run_paddleocr(image_path):
    """
    Runs PaddleOCR on RAW image using the .predict() method 
    to bypass Intel MKL-DNN C++ crashes.
    Returns: text (str), execution_time (float)
    """
    try:
        if not isinstance(image_path, str):
            return "ERROR", 0.0

        img = cv2.imread(image_path)
        if img is None:
            return "ERROR", 0.0

        # ⏱ OCR timing
        start_time = time.time()
        
        # 🔧 The Magic Fix: Use .predict() instead of .ocr()
        results = list(paddle_reader.predict(img))
        
        execution_time = time.time() - start_time

        if not results:
            return "ERROR", execution_time

        extracted_text = []

        # 🔧 Parse the new v3.0 Data Structure
        for res_obj in results:
            if hasattr(res_obj, 'res') and isinstance(res_obj.res, dict):
                texts = res_obj.res.get('rec_texts', [])
                scores = res_obj.res.get('rec_scores', [])
                
                for text, score in zip(texts, scores):
                    # We removed the confidence filter so it captures EVERYTHING
                    extracted_text.append(text)
            
            # Fallback for old list structure
            elif isinstance(res_obj, list):
                for line in res_obj:
                    if len(line) >= 2:
                        text = line[1][0]
                        extracted_text.append(text)

        extracted_text = " ".join(extracted_text)
        clean_text = " ".join(extracted_text.split())

        if not clean_text.strip():
            return "ERROR", execution_time

        return clean_text, execution_time

    except Exception as e:
        print(f"❌ PaddleOCR Exception caught: {e}")
        return "ERROR", 0.0
# ==========================================
# --- THE ULTIMATE HEAD-TO-HEAD TEST ---
# ==========================================
if __name__ == "__main__":
    test_image = "dataset/images/hin_sample_01.jpg" 
    
    if os.path.exists(test_image):
        print("="*60)
        print("🥊 THE ULTIMATE SHOWDOWN: Tesseract vs EasyOCR vs PaddleOCR")
        print("="*60 + "\n")
        
        # --- TESSERACT ---
        print("👴 Testing Tesseract (Processed Image)...")
        tes_text, tes_time = run_tesseract(test_image, apply_preprocessing=True) 
        print(f"⏱ Time: {tes_time:.2f} seconds")
        print(f"📝 TEXT: {tes_text}\n")
        print("-" * 40)
        
        # --- EASYOCR ---
        print("🧠 Testing EasyOCR (Raw Image)...")
        easy_text, easy_time = run_easyocr(test_image)
        print(f"⏱ Time: {easy_time:.2f} seconds")
        print(f"📝 TEXT: {easy_text}\n")
        print("-" * 40)

        # --- PADDLEOCR ---
        print("🚀 Testing PaddleOCR (Raw Image)...")
        pad_text, pad_time = run_paddleocr(test_image)
        print(f"⏱ Time: {pad_time:.2f} seconds")
        print(f"📝 TEXT: {pad_text}\n")
        print("="*60)
        
    else:
        print(f"❌ Could not find {test_image}. Are you in the main folder?")