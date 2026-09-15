# CodeOrbit Tech AI Internship - Task 3: Image Classification Using a Pretrained Model

**Organization:** CodeOrbit Tech  
**Domain:** Artificial Intelligence 
**Task:** Task 3 - Image Classification Using a Pretrained Model  

---

## 1. Project Title
**Image Classification Using Pretrained MobileNetV2 with Tkinter Desktop GUI and Python CLI**

---

## 2. Objective
The objective of this project is to implement a lightweight, reliable image classification application using a pretrained deep learning model. The application classifies given input images, displays the predicted category label with confidence percentages, and demonstrates how transfer learning and pretrained vision models operate without training a neural network from scratch.

---

## 3. Task Description
- Load an industry-standard, lightweight pretrained computer vision model.
- Provide a set of real sample images (`sample_images/`) representing diverse real-world objects.
- Preprocess input images to meet the exact input tensor shape, color channel distribution, and normalization requirements.
- Run forward inference on the images using the model.
- Extract and display the top predicted class labels alongside confidence scores.
- Offer both a **Tkinter Desktop GUI** for interactive file uploads and an automated **Command Line Interface (CLI)** for quick terminal execution.
- Maintain simple, beginner-friendly Python code suitable for academic viva and technical interviews.

---

## 4. Technologies Used
- **Programming Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch & Torchvision
- **Image Processing Library:** Pillow (PIL)
- **GUI Toolkit:** Python Tkinter (built-in)

---

## 5. Python Version
- **Recommended:** Python 3.9, 3.10, 3.11, or 3.12 (64-bit)

---

## 6. Libraries Used & Justification

| Library | Purpose & Justification |
| :--- | :--- |
| `torch` | Core PyTorch deep learning framework. Provides fast tensor operations and handles model execution (`torch.no_grad()`, softmax). |
| `torchvision` | Provides the official pretrained `MobileNetV2` model weights and standard image transformation pipelines (`transforms`). |
| `pillow` | Python Imaging Library used to safely open, decode, verify, resize, and convert various image formats (`.jpg`, `.png`, `.webp`, etc.) to RGB. |
| `tkinter` | Standard Python GUI library used to build the desktop interface without requiring web servers or complex browser runtimes. |

---

## 7. Model Used: MobileNetV2
- **Model Name:** MobileNetV2 (`mobilenet_v2`)
- **Trained On:** ImageNet-1K dataset (1.4+ million images across 1,000 object categories)
- **Model Size:** ~14 MB (very fast to download and run on student laptops)
- **Input Resolution:** 224 × 224 pixels (3-channel RGB)
- **Parameters:** ~3.5 Million parameters (optimized for edge and mobile inference)

### Why MobileNetV2 over heavier models (like VGG16 or ResNet-152)?
1. **Lightweight & Fast:** VGG16 is over 500 MB; MobileNetV2 is only ~14 MB.
2. **CPU Friendly:** Executes in milliseconds without needing an expensive dedicated GPU.
3. **High Accuracy:** Employs **inverted residual blocks** and **depthwise separable convolutions** to achieve near ResNet accuracy with a fraction of the computational footprint.

---

## 8. Features
- **Dual Interface:**
  - **Desktop GUI:** Interactive Tkinter window with file browser, image preview thumbnail, primary predicted label card, and top-3 confidence breakdown bars.
  - **CLI Mode:** Automated batch classification that reads from `sample_images/` or a specified image path and prints structured ASCII confidence bars.
- **Image Preprocessing Pipeline:** Automated RGB conversion, 224×224 center-crop resize, Tensor conversion, and ImageNet mean/std normalization.
- **Top-3 Predictions:** Displays both the #1 most probable class and the top-3 alternatives with percentage probabilities.
- **Robust Error Handling:** Catches corrupt or invalid image files cleanly and provides helpful user messages.
- **Zero Cloud / Web Server Dependencies:** Runs 100% locally on the student machine.

---

## 9. How the Pretrained Model Works (Simple Explanation)

1. **Pretraining on ImageNet:**
   MobileNetV2 has already been trained on over 1.4 million labeled images spanning 1,000 everyday categories (animals, vehicles, tools, food, etc.). During training, it learned to recognize visual patterns from simple edges up to complex object parts.

2. **Image Preprocessing:**
   The input image is converted to 3 RGB channels, resized so its shortest side is 256 pixels, center-cropped to exactly 224×224 pixels, scaled to `[0.0, 1.0]`, and normalized using ImageNet standard constants:
   $$\text{mean} = [0.485, 0.456, 0.406], \quad \text{std} = [0.229, 0.224, 0.225]$$

3. **Feature Extraction:**
   The image tensor passes through depthwise separable convolutional layers. Early layers detect lines and color contrasts; middle layers detect textures and shapes; deep layers detect high-level semantic features (such as floppy ears, wheels, or mug handles).

4. **Classification Head:**
   The pooled feature vector is passed to a final linear (dense) layer with 1,000 output nodes (logits), one for each ImageNet category.

5. **Softmax Function:**
   The raw output logits are converted into probabilities summing up to 100%:
   $$P(\text{class}_i) = \frac{e^{z_i}}{\sum_{j=1}^{1000} e^{z_j}}$$
   The category with the highest probability is our predicted label.

---

## 10. Project Structure

```text
codeorbit-task3-image-classification/
│
├── main.py              # Main Python application (Tkinter GUI + CLI classifier)
├── requirements.txt     # Essential dependencies (torch, torchvision, pillow)
├── README.md            # Comprehensive project documentation
└── sample_images/       # Sample images for testing
    ├── dog.jpg          # Golden retriever sample
    ├── cat.jpg          # Tabby cat sample
    ├── sports_car.jpg   # Sports car sample
    ├── coffee_mug.jpg   # Coffee mug sample
    └── banana.jpg       # Banana fruit sample
```

---

## 11. Installation Instructions

### Step 1: Clone or Download the Project
Extract the project folder onto your computer.

### Step 2: Open Terminal / Command Prompt
Navigate into the project directory:
```bash
cd codeorbit-task3-image-classification
```

### Step 3: (Recommended) Create a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```
*(On first run, PyTorch will automatically download the ~14MB MobileNetV2 weights into your local cache).*

---

## 12. How to Run the Project

### Option A: Launch the Desktop GUI Application
Simply run `main.py` without arguments:
```bash
python main.py
```
- A graphical window will open.
- Click **"📁 Choose Image from PC..."** to select any `.jpg` or `.png` from your computer.
- Or click one of the quick-test sample buttons (**Dog**, **Cat**, **Sports Car**, **Coffee Mug**) to classify instantly!

### Option B: Run in Console / CLI Mode
To run automated batch classification of all sample images in your terminal:
```bash
python main.py --cli
```

### Option C: Classify a Specific Image from Terminal
```bash
python main.py sample_images/sports_car.jpg
```

---

## 13. How to Add and Test Your Own Images
1. Save any image (JPEG or PNG format) into the `sample_images/` directory.
2. In the GUI, click **"Choose Image from PC..."** and select your newly added file.
3. In the CLI, run:
   ```bash
   python main.py sample_images/your_new_image.jpg
   ```

---

## 14. Example Output

### Console (CLI) Output:
```text
=================================================================
 CODEORBIT TECH AI INTERNSHIP - TASK 3
 Image Classification Using Pretrained MobileNetV2
=================================================================
[1/3] Loading pretrained MobileNetV2 model weights...
[2/3] Pretrained model loaded successfully!

[3/3] Classifying 5 image(s)...

-----------------------------------------------------------------
Image 1/5: dog.jpg
-> Predicted Label:   Golden Retriever
-> Confidence Score:  94.12%

   Top-3 Predictions:
   1. Golden Retriever          94.12%  ██████████████████
   2. Labrador Retriever         3.85%  
   3. Cocker Spaniel             0.62%  
-----------------------------------------------------------------
Image 2/5: sports_car.jpg
-> Predicted Label:   Sports Car
-> Confidence Score:  89.47%

   Top-3 Predictions:
   1. Sports Car                89.47%  ██████████████
   2. Convertible                5.12%  
   3. Grille                     1.30%  
-----------------------------------------------------------------
```

---

## 15. Limitations
1. **Limited to 1,000 ImageNet Classes:** The model can only recognize categories included in the ImageNet dataset. Highly specialized domains (such as specific medical scans, rare plant diseases, or regional Indian dishes) require fine-tuning or transfer learning.
2. **Single Primary Object Focus:** MobileNetV2 is an *image classification* model, not an *object detection* model. If an image contains both a cat, a dog, and a bicycle, it outputs the single dominant category rather than drawing bounding boxes.
3. **Input Resolution Constraints:** High-resolution photos are scaled down to 224×224 pixels, which may discard tiny, fine-grained details.

---

## 16. Conclusion
This project successfully fulfills **Task 3 of the CodeOrbit Tech AI Internship**. By leveraging PyTorch's pretrained **MobileNetV2**, we achieved rapid, accurate object classification on a standard laptop with zero cloud dependencies. Both an accessible Tkinter Desktop interface and an efficient CLI were developed, illustrating end-to-end practical application of deep learning in computer vision.
