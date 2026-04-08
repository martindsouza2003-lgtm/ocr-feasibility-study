import os
import time
import base64
import requests
from PIL import Image
import io

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
API_KEY = "sk-9b277064bf5c4259baa2116d4680fc45"  # ⚠️ Put your real key back here!
BASE_URL = "https://llm.buzzybrains.org/api/chat/completions"
MODEL_NAME = "qwen3-vl:8b"

DATASET_FOLDER = "dataset/images"

# --- CHANGED: Use the exact absolute path you requested ---
OUTPUT_FOLDER = "/home/userbbl25/Downloads/ocr_evolution_project/results/predictions"

# Create the main prediction folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# ==========================================

def shrink_and_encode_image(image_path, max_size=1024):
    """Resizes the image to stop Cloudflare 524 Timeouts"""
    print(f"   🖼️ Shrinking {os.path.basename(image_path)}...")
    img = Image.open(image_path).convert("RGB")
    
    # Maintain aspect ratio, but ensure the longest side is 1024px
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Save to a temporary memory buffer, compressing it to 85% quality
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# --- THE PROMPT & INFERENCE TWEAKS ---
system_prompt = (
    "You are an expert paleographer specializing in Devanagari handwriting. "
    "Transcribe the following Hindi text exactly as written. "
    "If a character is ambiguous, use linguistic context to determine the most likely word. "
    "Do not translate. Do not summarize. Just output the raw Devanagari text."
)

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- THE BATCH LOOP ---
# Find all jpg and png files in your dataset folder
image_files = sorted([f for f in os.listdir(DATASET_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"\n🚀 Found {len(image_files)} images. Starting batch processing...\n")

for filename in image_files:
    image_path = os.path.join(DATASET_FOLDER, filename)
    
    # --- FOLDER & FILE NAMING LOGIC ---
    # 1. Get the base name (e.g., "hin_sample_02")
    base_name = os.path.splitext(filename)[0]
    
    # 2. Safely create the folder inside your absolute path
    image_specific_folder = os.path.join(OUTPUT_FOLDER, base_name)
    os.makedirs(image_specific_folder, exist_ok=True) 
    
    # 3. Name the file EXACTLY "qwen.txt"
    txt_filename = "qwen.txt"
    txt_path = os.path.join(image_specific_folder, txt_filename)
    # --------------------------------------
    
    # Skip if we already processed this specific Qwen text file
    if os.path.exists(txt_path):
        print(f"⏩ Skipping {filename} (Qwen text already exists)")
        continue

    print(f"🔄 Processing: {filename}")
    
    try:
        base64_image = shrink_and_encode_image(image_path)
        
        payload = {
            "model": MODEL_NAME,
            "temperature": 0.1, # FORCE AI TO BE STRICT, NOT CREATIVE
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        }
        
        # Send request
        response = requests.post(BASE_URL, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            ai_text = response.json()['choices'][0]['message']['content']
            
            # Save to text file in the specific sub-folder
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(ai_text)
            print(f"   ✅ Saved transcription to {txt_path}")
            
        else:
            print(f"   ❌ SERVER ERROR {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"   ❌ CRASH: {e}")
        
    # Wait 3 seconds between images so we don't get banned by the server!
    time.sleep(3) 

print("\n🎉 BATCH PROCESSING COMPLETE!")