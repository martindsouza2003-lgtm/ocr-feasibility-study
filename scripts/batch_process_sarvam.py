import os
import time
import zipfile
import warnings
from sarvamai import SarvamAI

# Hide annoying warnings
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
API_KEY = "sk_dv063h34_YjfiEEQTVny9FPbGXpra1RzK"   # Paste your key here!
client = SarvamAI(api_subscription_key=API_KEY)

IMAGE_DIR = "dataset/images"
OUTPUT_DIR = "results/predictions"

print("🚀 Starting Sarvam Vision Batch Processing using the official SDK...")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. LOOP THROUGH IMAGES ---
for filename in sorted(os.listdir(IMAGE_DIR)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(IMAGE_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        
        # Create the subfolder for the image
        img_output_dir = os.path.join(OUTPUT_DIR, base_name)
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)
            
        txt_path = os.path.join(img_output_dir, "Sarvam.txt")
        zip_output_path = os.path.join(img_output_dir, "sarvam_temp.zip")
        
        print(f"\n🔄 Processing: {filename} ...")
        
        try:
            # 1. Create a Document Intelligence job for Hindi ("hi-IN")
            job = client.document_intelligence.create_job(
                language="hi-IN",
                output_format="md"
            )
            
            # 2. Upload the handwritten image
            job.upload_file(img_path)
            
            # 3. Start processing
            job.start()
            print("   ⏳ Job started. Waiting for Sarvam's servers to process...")
            
            # 4. Wait for completion
            status = job.wait_until_complete()
            
            if status.job_state.upper() == "COMPLETED":
                # 5. Download the output ZIP file
                job.download_output(zip_output_path)
                
                # 6. Extract the text from the ZIP file silently
                with zipfile.ZipFile(zip_output_path, 'r') as zip_ref:
                    # Sarvam returns a Markdown (.md) file inside the zip
                    md_files = [m for m in zip_ref.namelist() if m.endswith('.md')]
                    if md_files:
                        extracted_content = zip_ref.read(md_files[0]).decode('utf-8')
                        
                        # Save it as our standard Sarvam.txt
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(extracted_content.strip())
                        print(f"✅ Saved successfully: {txt_path}")
                    else:
                        print("❌ No markdown file found in the output zip.")
                        
                # Clean up the temporary zip file
                if os.path.exists(zip_output_path):
                    os.remove(zip_output_path)
            else:
                print(f"❌ Job failed. Final Status: {status.job_state}")
                
            # Rest for 3 seconds to be polite to the API rate limits
            time.sleep(3)
            
        except Exception as e:
            print(f"❌ Script Error on {filename}: {e}")

print("\n🏁 Sarvam Vision processing complete!")