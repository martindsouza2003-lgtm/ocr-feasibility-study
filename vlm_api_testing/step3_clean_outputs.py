import os
import re

# Point this to your actual predictions folder
OUTPUT_FOLDER = "/home/userbbl25/Downloads/ocr_evolution_project/results/predictions"

print("🧹 Starting the AI Disclaimer Cleanup...")

cleaned_count = 0

# Walk through all folders and subfolders in predictions
for root, dirs, files in os.walk(OUTPUT_FOLDER):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            
            # Read the original text
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Look for the exact "(नोट:" string and anything inside those parentheses
            # re.DOTALL allows it to match across multiple lines if the AI pressed Enter
            cleaned_content = re.sub(r'\(नोट:.*?\)', '', content, flags=re.DOTALL)
            
            # Also strip out any stray "नोट:" without parentheses just to be safe
            cleaned_content = re.sub(r'नोट:.*', '', cleaned_content)
            
            # Clean up any extra blank lines left behind
            cleaned_content = "\n".join([line for line in cleaned_content.splitlines() if line.strip() != ""])
            
            # If the text changed, save it back!
            if content != cleaned_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_content)
                print(f"   ✨ Scrubbed disclaimer from: {file}")
                cleaned_count += 1

print(f"\n✅ Cleanup Complete! {cleaned_count} files were fixed.")