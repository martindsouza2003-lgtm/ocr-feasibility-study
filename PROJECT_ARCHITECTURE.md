# 🚀 Offline Hindi Handwriting OCR: Feasibility & Architecture

## 🎯 Project Goal

To engineer a highly accurate, 100% offline Artificial Intelligence pipeline capable of reading full, messy pages of cursive Hindi (Devanagari) handwriting.

---

## 🏗️ Phase 1: The Global Baseline Audit

We first tested global, out-of-the-box OCR engines to establish a baseline. Because Indic scripts use complex continuous lines (Shirorekha) and modifiers (Matras), generic Western models completely failed.

**Models Tested:**

1. **Tesseract:** (Legacy Google AI) - Fast, but highly inaccurate for cursive.
2. **EasyOCR:** (Deep Learning) - High compute cost, poor Indic context.
3. **PaddleOCR:** (Enterprise AI) - Completely failed (100% CER) on cursive Hindi.
4. **TrOCR:** (Microsoft Transformers) - Failed due to English-biased training vocabulary.

**Phase 1 Output:** We engineered an automated pipeline that tracks RAM, CPU, and Character Error Rate (CER), saving predictions to text files and generating a React-style local HTML dashboard for executive review.

---

## 🇮🇳 Phase 2: Bhashini & PLATTER Integration

To solve the failures of Phase 1, we integrated **Bhashini**, the Indian Government's digital public infrastructure for language AI.

Specifically, we integrated **PLATTER** (Page-Level hAndwriTTen TExt Recognition), a two-stage framework funded by Bhashini and built by IIT Bombay. We utilized their custom Indic-trained DocTR engine to handle page-level cropping and recognition.

**Downloaded Weights (`.pt` files):**
We downloaded specific model weights from the researchers to run the brain locally without the internet:

- `all_handwritten_crnn_vgg16_bn_hindi.pt`: A Convolutional Recurrent Neural Network trained specifically on Hindi handwriting.
- `all_handwritten_master_hindi.pt`: An alternative, highly advanced attention-based architecture (MASTER) trained on Hindi handwriting.

---

## 📂 Master Directory Map

The project is structured using the "Modular Append" strategy. Academic research code is quarantined from our clean production code to prevent dependency conflicts.

```text
📦 ocr_evolution_project/
 ├── 📄 PROJECT_ARCHITECTURE.md     <-- (You are here: The Master Guide)
 │
 ├── 📂 dataset/                    <-- Data Layer
 │    ├── 📂 images/                (Raw handwriting image files)
 │    └── 📂 ground_truth/          (Perfect human transcriptions for math grading)
 │
 ├── 📂 scripts/                    <-- Phase 1: Core Engine Layer
 │    ├── metrics_tracker.py        (Calculates CER, WER, CPU %, and RAM MB)
 │    ├── ocr_engines.py            (Wrappers for Tesseract, EasyOCR, PaddleOCR)
 │    ├── nextgen_engines.py        (Wrapper for Microsoft TrOCR)
 │    └── master_evaluator.py       (The pipeline that runs Phase 1 and outputs the CSV)
 │
 ├── 📂 results/                    <-- Output Layer
 │    ├── ocr_comparison_results.csv(The master spreadsheet of hardware & accuracy math)
 │    ├── final_summary.txt         (Terminal output summary for quick reading)
 │    └── 📂 predictions/           (Contains one folder per image, with .txt AI guesses)
 │
 ├── 📂 audit_reports/              <-- Presentation Layer
 │    ├── index.html                (The live Visual Dashboard for Executives)
 │    └── generate_audit.py         (Script that turns prediction text files into HTML)
 │
 └── 📂 bhashini_core/              <-- Phase 2: Quarantine Zone for Academic AI
      ├── 📂 PLATTER/               (Cloned repository from IIT Bombay)
      │    ├── 📂 weights/          (Holds the .pt Hindi model brains)
      │    └── 📂 platter_env/      (Isolated Python environment for custom DocTR engine)
      └── bhashini_evaluator.py     (Script to run Phase 2 and append to CSV)

## 🛠️ Technology Stack & Dependency Map (The "package.json" equivalent)

To ensure total reproducibility, this project separates dependencies into two isolated Python Virtual Environments.

### 1. Phase 1 Environment (`ocr_env`)
This environment powers the global OCR testing, data calculation, and HTML generation.

* **Core Machine Learning & OCR Engines:**
  * `pytesseract` + OS-level `tesseract-ocr` (Google Legacy OCR)
  * `easyocr` (Deep Learning OCR)
  * `paddlepaddle` & `paddleocr` (Baidu Enterprise OCR)
  * `transformers` & `torch` (Microsoft TrOCR via HuggingFace)
* **Metrics & Hardware Tracking (The "Calculators"):**
  * `jiwer`: Calculates mathematical CER (Character Error Rate) and WER (Word Error Rate).
  * `psutil`: Directly hooks into the OS motherboard to track CPU% and RAM MB usage.
* **Data Processing & Helpers:**
  * `pandas`: Handles the matrix calculations and saves the `ocr_comparison_results.csv`.
  * `opencv-python` (`cv2`): Handles image reading and normalization.

### 2. Phase 2 Quarantine Environment (`platter_env`)
This highly isolated environment is explicitly for running the Bhashini/IIT Bombay academic models. It contains specific version locks and patches required to run legacy academic code on modern hardware.

* **The Core AI Framework:**
  * `torch`, `torchvision`, `torchaudio`: The deep learning brain. *(Note: Code utilizes `weights_only=False` bypass to load full model architectures saved in older PyTorch versions).*
* **The OCR Engine:**
  * `doctr` (Indic Branch): Installed directly via GitHub to access IIT Bombay's unreleased modifications (`git+https://github.com/iitb-research-code/doctr.git@indic`).
* **Academic Patches & Version Locks (CRITICAL):**
  * `Levenshtein`: Manually installed. (Required by Indic DocTR, missing from their setup file).
  * `huggingface_hub==0.22.2`: **Strictly Locked.** Must be downgraded to this exact version to prevent `ImportError: cannot import name 'HfFolder'`, as modern HuggingFace deprecated functions used by the researchers.
* **Data Evaluation Pipeline:**
  * `pandas`, `psutil`, `jiwer`: Re-installed in this environment to allow `bhashini_evaluator.py` to calculate system metrics and CER independently of Phase 1.

 ### 3. Phase 3 Quarantine Environment (`bhaasha_env`)
This isolated environment runs the heavy IIIT Hyderabad two-stage pipeline. We completely discarded their bloated `requirements.txt` (which contained 10GB of conflicting NVIDIA GPU drivers) to build a stable, CPU-safe stack.
* **Vision & Cropping (The Cutter):**
  * `ultralytics==8.3.22`: The YOLOv8 engine responsible for detecting and cropping words/lines.
  * `opencv-python`, `scikit-image`, `matplotlib`: Bounding box geometry and image slicing.
* **The OCR Brain (The Reader):**
  * `torch`, `torchvision`: *(Note: Their `infer.py` script was heavily rewritten by our team to strip out 7 hardcoded `.cuda()` hardware checks and safely extract Multi-GPU `module.` weight prefixes onto a standard CPU).*
* **Academic Dependencies:**
  * `lmdb`: Lightning Memory-Mapped Database, used for rapid image parsing.
  * `tqdm`, `pyyaml`: Configuration and progress parsing.
* **Data Evaluation (Subprocess Math):**
  * `pandas`, `psutil`, `jiwer`: Installed to run our `bhaasha_evaluator.py` robot script, preventing massive memory leaks during inference.

### 3. Presentation Layer (UI/Frontend)
* **Tailwind CSS (via CDN):** Used inside `generate_audit.py` to automatically style the output HTML dashboard dynamically without needing a local CSS compilation step or a Node.js server.
```
