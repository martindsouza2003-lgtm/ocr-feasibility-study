import warnings
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
from PIL import Image

warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
MODEL_ID = "sabaridsnfuji/Hindi_Offline_Handwritten_OCR"
IMAGE_PATH = "/home/userbbl25/Downloads/ocr_evolution_project/bhashini_core/doctr_crop_audit/hin_sample_01/B0_L0_W6.png" # Your specific word image
# ==========================================

print(f"\n🚀 Initializing TrOCR for: {MODEL_ID}")

print("📥 Assembling the 'Frankenstein' Processor...")
try:
    # 1. Use Microsoft's official image processor (The "Eyes")
   # 1. Use Microsoft's official image processor, BUT SHRINK TO 224x224!
    image_processor = ViTImageProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten", 
        size={"height": 224, "width": 224}
    )
    
    # 2. Use the Hindi model's specific dictionary (The "Brain")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 3. Combine them to bypass the missing config file!
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
except Exception as e:
    print(f"❌ Failed to build processor: {e}")
    exit()

print("🧠 Loading model weights...")
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

# --- THE MAGIC FIX: PADDING TO SQUARE ---
def pad_to_square(img):
    """Pastes the rectangular word onto the center of a perfect white square"""
    width, height = img.size
    max_dim = max(width, height)
    # Create a new pure white square
    square_canvas = Image.new("RGB", (max_dim, max_dim), "white")
    # Paste the original image in the exact center
    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2
    square_canvas.paste(img, (paste_x, paste_y))
    return square_canvas

print("🖼️ Reading and formatting the image...")
try:
    raw_image = Image.open(IMAGE_PATH).convert("RGB")
    # Stop the AI from stretching the image!
    safe_image = pad_to_square(raw_image)
except FileNotFoundError:
    print(f"\n❌ ERROR: Could not find '{IMAGE_PATH}'.")
    exit()

pixel_values = processor(images=safe_image, return_tensors="pt").pixel_values

print("⏳ Predicting text...")
generated_ids = model.generate(pixel_values, max_length=50)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n" + "="*50)
print(f"🎯 FINAL PREDICTION:\n{generated_text}")
print("="*50 + "\n")