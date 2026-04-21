import os
import time
from google import genai
import PIL.Image
import warnings

# फालतू की वॉर्निंग्स छुपाने के लिए
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
API_KEY = "AIzaSyBBZvNkWSvB42hMVTKjjWWjBEOKPqOYL9M" # यहाँ अपनी API Key डालें
client = genai.Client(api_key=API_KEY)

IMAGE_DIR = "dataset/images"
OUTPUT_DIR = "results/predictions"

print("🚀 25 इमेजेस की बैच प्रोसेसिंग शुरू हो रही है...")

# अगर predictions फोल्डर नहीं है, तो उसे बनाएं
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. LOOP THROUGH IMAGES ---
# फोल्डर की सभी फाइल्स को पढ़ें और लूप चलाएं
for filename in sorted(os.listdir(IMAGE_DIR)):
    # सिर्फ इमेज फाइल्स को प्रोसेस करें
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(IMAGE_DIR, filename)
        
        # एक्सटेंशन हटाकर इमेज का नाम निकालें (e.g., 'hin_sample_01')
        base_name = os.path.splitext(filename)[0]
        
        # इमेज के नाम का नया फोल्डर बनाएं
        img_output_dir = os.path.join(OUTPUT_DIR, base_name)
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)
            
        # टेक्स्ट फाइल का पाथ
        txt_path = os.path.join(img_output_dir, f"{base_name}.txt")
        
        print(f"\n🔄 प्रोसेस हो रहा है: {filename} ...")
        
        try:
            # इमेज लोड करें
            img = PIL.Image.open(img_path)
            prompt = "Please transcribe the Devanagari (Hindi) handwriting in this image. Do not summarize, just output the exact text."
            
            # API कॉल करें
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt, img]
            )
            
            # रिज़ल्ट को .txt फाइल में सेव करें
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(response.text.strip())
                
            print(f"✅ सेव किया गया: {txt_path}")
            
            # CRITICAL: फ्री टियर की लिमिट से बचने के लिए 5 सेकंड रुकें
            print("⏳ रेट लिमिट (Rate Limit) से बचने के लिए 5 सेकंड का ब्रेक...")
            time.sleep(5)
            
        except Exception as e:
            print(f"❌ {filename} पर एरर आ गया: {e}")

print("\n🏁 सारी इमेजेस की प्रोसेसिंग सफलतापूर्वक पूरी हो गई है!")

"sk_dv063h34_YjfiEEQTVny9FPbGXpra1RzK" 