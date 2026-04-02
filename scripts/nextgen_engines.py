import os
import time
import re
import unicodedata
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

# ==========================================
# PHASE 4: TrOCR IMPLEMENTATION (Hindi)
# ==========================================
def load_trocr_model():
    print("⏳ Loading TrOCR Hindi Model...")
    model_id = "sabaridsnfuji/Hindi_Offline_Handwritten_OCR"

    print("   -> Fetching base Image Processor...")
    image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    
    print("   -> Fetching base Hindi Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('surajp/RoBERTa-hindi-guj-san')
    
    print("   -> Assembling custom TrOCR Processor...")
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

    print("   -> Downloading actual Hindi Model Weights...")
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    model.eval()  # ✅ IMPORTANT: Sets model to evaluation mode

    print("✅ TrOCR Model Loaded Successfully!")

    return processor, model

def run_trocr(image_path, processor, model):
    """
    Runs TrOCR on the image using the provided loaded model.
    Returns: text (str), execution_time (float)
    """
    try:
        if not os.path.exists(image_path):
            return "ERROR", 0.0

        image = Image.open(image_path).convert("RGB")

        start_time = time.time()

        pixel_values = processor(image, return_tensors="pt").pixel_values

        # Move to device (optional but good)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pixel_values = pixel_values.to(device)
        model.to(device)

        # Disable gradients (IMPORTANT for CPU RAM and speed)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=256)

        extracted_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        execution_time = time.time() - start_time

        clean_text = extracted_text.strip()

        if not clean_text:
            return "ERROR", execution_time

        return clean_text, execution_time

    except Exception as e:
        print(f"❌ TrOCR Error: {e}")
        return "ERROR", 0.0

# ==========================================
# PHASE 5: TEXT NORMALIZATION
# ==========================================
def normalize_text(text):
    """
    Cleans OCR output AND Ground Truth for fair metric comparison.
    """
    if not isinstance(text, str) or text == "ERROR":
        return text
        
    # 1. Remove Zero-Width Joiners (ZWJ) and Non-Joiners (ZWNJ)
    text = text.replace('\u200d', '').replace('\u200c', '')
    
    # 2. Normalize Unicode (Combines split matras into single characters)
    text = unicodedata.normalize('NFC', text)
    
    # 3. Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # 4. Strip leading/trailing spaces
    return text.strip()

# ==========================================
# TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    test_image = "dataset/images/hin_sample_01.jpg"
    
    if os.path.exists(test_image):
        print("="*50)
        print("🧠 TESTING VISION TRANSFORMER (TrOCR)")
        print("="*50)
        
        proc, mod = load_trocr_model()
        
        print("\n📄 Reading Image...")
        text, exec_time = run_trocr(test_image, proc, mod)
        clean_text = normalize_text(text)
        
        print(f"⏱️ Time Taken: {exec_time:.2f} seconds")
        print(f"📝 Raw Text: {text}")
        print(f"🧹 Normalized: {clean_text}")
        print("="*50)
    else:
        print(f"❌ Test image not found at {test_image}")