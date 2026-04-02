import cv2
import os

def preprocess_for_tesseract(image_path):
    """
    Preprocess image for Tesseract OCR:
    - Dynamic Resize (only if small)
    - Grayscale
    - Denoise (Mild)
    - Adaptive Threshold
    """
    img = cv2.imread(image_path)
    
    if img is None:
        return None

    # 1. DYNAMIC RESIZE: Check the width (index 1 of shape)
    # If the image is less than 1000 pixels wide, scale it up.
    if img.shape[1] < 1000:
        img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Denoise (3x3 protects Hindi matras/bindis)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 4. Adaptive threshold (Static baseline for our experiment)
    processed_img = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # 5. Ensure correct polarity (Dark text on white background)
    if cv2.mean(processed_img)[0] < 127:
        processed_img = cv2.bitwise_not(processed_img)

    return processed_img


# --- Quick Test ---
if __name__ == "__main__":
    test_image = "dataset/images/hin_sample_01.jpg" 
    
    if os.path.exists(test_image):
        print(f"Image found at {test_image}! Applying preprocessing...")
        clean_img = preprocess_for_tesseract(test_image)
        
        if clean_img is not None:
            output_path = "dataset/images/test_processed_output.jpg"
            cv2.imwrite(output_path, clean_img)
            print(f"Done! Check '{output_path}'")
        else:
            print("Processing failed.")
    else:
        print(f"Could not find {test_image}. Are you running this from the main folder?")