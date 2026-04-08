import base64
import requests

# --- YOUR VIP BADGE AND ADDRESS ---
# Replace the text inside the quotes with your actual key
API_KEY = "sk-9b277064bf5c4259baa2116d4680fc45"  
BASE_URL = "https://llm.buzzybrains.org/api/chat/completions"

# --- THE ENVELOPE (PAYLOAD) ---
# We will use the deepseek model from your screenshot
MODEL_NAME = "qwen3-vl:8b"
IMAGE_PATH = "dataset/images/hin_sample_07.jpg"

print("🔄 Converting image into web-safe format...")
with open(IMAGE_PATH, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Please transcribe the Devanagari (Hindi) handwriting in this image. Do not summarize, just output the exact text."
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }
    ]
}

print(f"🚀 Sending image to {MODEL_NAME} on the BuzzyBrains server...")
response = requests.post(BASE_URL, headers=headers, json=payload)

# --- THE RESULT ---
if response.status_code == 200:
    print("\n✅ CONNECTION SUCCESSFUL! Here is what the AI read:\n")
    print(response.json()['choices'][0]['message']['content'])
else:
    print(f"\n❌ SERVER ERROR (Code {response.status_code}):")
    print(response.text)