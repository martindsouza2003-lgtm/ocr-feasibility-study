import cv2
import os
import sys

# Ensure Python can find our custom scripts
sys.path.append(os.path.dirname(__file__))

from nextgen_engines import load_trocr_model, run_trocr
from metrics_tracker import normalize_text

def main():
    print("="*50)
    print("✂️ INTERACTIVE TrOCR TEST")
    print("="*50)

    image_path = "dataset/images/hin_sample_01.jpg"

    if not os.path.exists(image_path):
        print("❌ Image not found")
        return

    img = cv2.imread(image_path)
    
    # ---------------------------------------------------------
    # 🖱️ THE MAGIC: Let the user draw the box!
    # ---------------------------------------------------------
    print("🖱️ A window is opening! ")
    print("   1. Click and drag to draw a box around EXACTLY ONE line of text.")
    print("   2. Press ENTER or SPACE to confirm the crop.")
    print("   (If you mess up, press 'c' to cancel and redraw)")
    
    # This opens the GUI window
    roi = cv2.selectROI("Draw a box around ONE line, then press ENTER", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi

    # If the user closed the window without drawing
    if w == 0 or h == 0:
        print("❌ Crop cancelled. Exiting.")
        return

    # Crop the image using the exact coordinates you drew
    cropped_img = img[y:y+h, x:x+w]

    debug_path = "dataset/images/debug_crop.jpg"
    cv2.imwrite(debug_path, cropped_img)
    print(f"\n📸 Perfect crop saved to: {debug_path}")

    # ---------------------------------------------------------
    # 🧠 RUN THE AI
    # ---------------------------------------------------------
    proc, mod = load_trocr_model()

    print("🧠 Running TrOCR on your custom cropped line...")
    raw_text, exec_time = run_trocr(debug_path, proc, mod)
    clean_text = normalize_text(raw_text)

    print("-" * 50)
    print(f"⏱ Time: {exec_time:.2f}s")
    print(f"📝 Raw: {raw_text}")
    print(f"🧹 Clean: {clean_text}")
    print("-" * 50)

if __name__ == "__main__":
    main()