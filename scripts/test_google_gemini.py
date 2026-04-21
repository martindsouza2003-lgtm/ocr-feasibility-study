from google import genai
import PIL.Image
import os
import warnings

# Hide annoying warnings
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
# IMPORTANT: Put your real API key here again!
API_KEY = "AIzaSyBBZvNkWSvB42hMVTKjjWWjBEOKPqOYL9M"
client = genai.Client(api_key=API_KEY)

# Notice the path fix! Since you run the script from the main folder, 
# we don't need the "../" anymore.
IMAGE_PATH = "dataset/images/hin_sample_01.jpg"

# --- 2. LOAD IMAGE ---
print(f"🔄 Looking for image at: {IMAGE_PATH}")
if not os.path.exists(IMAGE_PATH):
    print(f"❌ ERROR: Image not found. Are you sure 'hin_sample_01.jpg' is exactly there?")
    exit()

img = PIL.Image.open(IMAGE_PATH)

# --- 3. TEST GOOGLE GEMINI ---
prompt = "Please transcribe the Devanagari (Hindi) handwriting in this image. Do not summarize, just output the exact text."

print("🚀 Sending image to Google Gemini 1.5 Pro...")
try:
    # This is the new way to call the API!
    response = client.models.generate_content(
       model='gemini-2.5-flash',
        contents=[prompt, img]
    )
    
    print("\n✅ GOOGLE'S RESULT:")
    print("--------------------------------------------------")
    print(response.text.strip())
    print("--------------------------------------------------")
except Exception as e:
    print(f"\n❌ ERROR: {e}")