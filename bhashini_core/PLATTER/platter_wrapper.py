import os
import time
import torch
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# ==========================================
# PATH SETUP (Robust)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))

WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

CRNN_WEIGHTS = os.path.join(WEIGHTS_DIR, "all_handwritten_crnn_vgg16_bn_hindi.pt")
MASTER_WEIGHTS = os.path.join(WEIGHTS_DIR, "all_handwritten_master_hindi.pt")


# ==========================================
# MODEL LOADER
# ==========================================
def load_platter_model(brain_type="crnn"):
    """Loads PLATTER (DocTR Indic) model with custom Hindi weights."""
    print(f"🧠 Booting PLATTER ({brain_type.upper()} Brain)...")

    # Select architecture
    reco_arch = "crnn_vgg16_bn" if brain_type == "crnn" else "master"
    weights_path = CRNN_WEIGHTS if brain_type == "crnn" else MASTER_WEIGHTS

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Using device: {device}")

    # Load predictor WITH pretrained weights for the Detector
    predictor = ocr_predictor(
        det_arch="db_resnet50",
        reco_arch=reco_arch,
        pretrained=True
    )

    # Load custom Hindi weights for the Recognizer
    if os.path.exists(weights_path):
        try:
            # 1. Load the ENTIRE Hindi model object (The whole car)
            hindi_model = torch.load(weights_path, map_location=device, weights_only=False)
            hindi_model.eval()

            # 2. Swap out the default English brain completely!
            predictor.reco_predictor.model = hindi_model

            # 3. Tell the translator to use the 269 Hindi characters
            if hasattr(hindi_model, 'vocab'):
                predictor.reco_predictor.vocab = hindi_model.vocab
                # Update the DocTR post-processor if it exists
                if hasattr(predictor.reco_predictor, 'task_processor'):
                     predictor.reco_predictor.task_processor.vocab = hindi_model.vocab
            elif hasattr(hindi_model, 'cfg') and 'vocab' in hindi_model.cfg:
                predictor.reco_predictor.vocab = hindi_model.cfg['vocab']

            print("✅ Whole Hindi AI Brain swapped successfully!")

        except Exception as e:
            print(f"❌ Brain Swap failed: {str(e)}")
            return None
    else:
        print(f"❌ ERROR: Weights not found at {weights_path}")
        return None

    # Move model to device
    predictor.to(device)
    return predictor


# ==========================================
# OCR RUNNER
# ==========================================
def run_platter(image_path, predictor):
    """Runs PLATTER OCR on a single image."""
    if predictor is None:
        return "ERROR: MODEL_NOT_LOADED", 0.0, 0.0

    start_time = time.time()

    try:
        # Load image
        doc = DocumentFile.from_images(image_path)

        # Run OCR
        result = predictor(doc)

        extracted_lines = []
        confidences = []

        # Traverse DocTR structure
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = []
                    for word in line.words:
                        words.append(word.value)

                        # Capture confidence (if available)
                        if hasattr(word, "confidence") and word.confidence is not None:
                            confidences.append(word.confidence)

                    line_text = " ".join(words).strip()
                    if line_text:
                        extracted_lines.append(line_text)

        final_text = "\n".join(extracted_lines).strip()

        exec_time = time.time() - start_time

        # Handle EMPTY output
        if not final_text:
            return "EMPTY", exec_time, 0.0

        # Compute average confidence
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return final_text, exec_time, avg_conf

    except Exception as e:
        return f"ERROR: {str(e)}", 0.0, 0.0


# ==========================================
# QUICK TEST (Single Image)
# ==========================================
if __name__ == "__main__":
    test_image = os.path.join(PROJECT_ROOT, "dataset/images/hin_sample_01.jpg")

    if not os.path.exists(test_image):
        print(f"⚠️ Test image not found at: {test_image}")
        exit()

    print("🚀 Running PLATTER test...\n")

    # Load model
    platter_model = load_platter_model("crnn")

    # Run OCR
    text, time_taken, confidence = run_platter(test_image, platter_model)

    print("\n" + "=" * 50)
    print("📝 PLATTER OUTPUT:")
    print("=" * 50)
    print(text)

    print("\n" + "=" * 50)
    print(f"⏱️ Time Taken: {round(time_taken, 2)} sec")
    print(f"📊 Avg Confidence: {round(confidence, 3)}")
    print("=" * 50)