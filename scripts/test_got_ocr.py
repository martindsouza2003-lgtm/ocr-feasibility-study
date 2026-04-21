import warnings
import torch
from transformers import AutoModel, AutoTokenizer

# Hide warnings to keep terminal clean
warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
MODEL_ID = "stepfun-ai/GOT-OCR2_0"
IMAGE_PATH = "/home/userbbl25/Downloads/ocr_evolution_project/dataset/images/hin_sample_01.jpg" # You can put a FULL PAGE image here too!
# ==========================================

print(f"\n🚀 Initializing GOT-OCR 2.0 (CPU Mode)...")

print("📥 Downloading/Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print("🧠 Downloading/Loading the 580M Brain (This will take a few minutes the first time)...")
# We explicitly force it onto the CPU, and use low_cpu_mem_usage to protect your 16GB RAM
model = AutoModel.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True, 
    low_cpu_mem_usage=True, 
    device_map="cpu", 
    use_safetensors=True
)

# Put the model into evaluation mode (inference only, no training)
model = model.eval()

print("🖼️ Processing the image...")
try:
    # GOT-OCR has a built-in function to handle the image natively!
    print("⏳ AI is reading (This may take 15-30 seconds on your i7-6600U)...")
    
    # The 'ocr' type tells the model to just read all the text it sees
    res = model.chat(tokenizer, IMAGE_PATH, ocr_type='ocr')
    
    print("\n" + "="*50)
    print(f"🎯 GOT-OCR OUTPUT:\n{res}")
    print("="*50 + "\n")

except FileNotFoundError:
    print(f"\n❌ ERROR: Could not find '{IMAGE_PATH}'.")
except Exception as e:
    print(f"\n❌ ERROR: {e}")