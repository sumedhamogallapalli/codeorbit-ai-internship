"""
CodeOrbit Tech Artificial Intelligence Internship
Task 3: Image Classification Using a Pretrained Model

Student: Final Year B.Tech (Computer Science & Engineering)
Model: MobileNetV2 (Pretrained on ImageNet-1K)
Framework: PyTorch & Torchvision
UI: Tkinter Desktop GUI + Command Line Interface (CLI)
"""

import os
import sys
from pathlib import Path

# Check and import required libraries with helpful error messages for students
try:
    from PIL import Image
except ImportError:
    print("[Error] Pillow is not installed. Please install it using: pip install pillow")
    sys.exit(1)

try:
    import torch
    import torch.nn.functional as F
    from torchvision import models
except ImportError:
    print("[Error] PyTorch or Torchvision is not installed.")
    print("Please install requirements using: pip install -r requirements.txt")
    print("Or run: pip install torch torchvision pillow")
    sys.exit(1)


# ==============================================================================
# 1. MODEL LOADING & PREPROCESSING
# ==============================================================================

def load_model():
    """
    Loads the pretrained MobileNetV2 model and its associated preprocessing transforms.
    
    Why MobileNetV2?
    - Lightweight: ~14MB download, fast inference even on standard student laptops.
    - Accurate: Trained on the ImageNet dataset across 1,000 diverse real-world classes.
    - Pretrained: No model training required; weights are downloaded automatically.
    """
    print("[1/3] Loading pretrained MobileNetV2 model weights...")
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.eval()  # Set model to evaluation (inference) mode
    
    # Standard preprocessing pipeline for MobileNetV2
    transforms_fn = weights.transforms()
    
    # 1000 human-readable class names from ImageNet
    categories = weights.meta["categories"]
    
    print("[2/3] Pretrained model loaded successfully!")
    return model, transforms_fn, categories


def preprocess_image(image_path, transforms_fn):
    """
    Preprocesses an input image according to the model's required format:
    1. Opens the image using PIL and converts to 3-channel RGB.
    2. Resizes and center-crops the image to 224x224 pixels.
    3. Converts pixel values [0, 255] into a float Tensor [0.0, 1.0].
    4. Normalizes with ImageNet standard mean and standard deviation.
    5. Adds a batch dimension: shape becomes [1, 3, 224, 224].
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image '{image_path}'. Reason: {e}")
    
    # Apply standard torchvision transformation pipeline
    input_tensor = transforms_fn(image)
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    batch_tensor = input_tensor.unsqueeze(0)
    return image, batch_tensor


def classify_image(model, batch_tensor, categories, top_k=3):
    """
    Passes the preprocessed image tensor through MobileNetV2:
    1. Computes raw unnormalized scores (logits) across 1000 ImageNet classes.
    2. Applies Softmax to convert raw logits into probability percentages (0% - 100%).
    3. Retrieves top_k most probable predictions.
    """
    with torch.no_grad():  # Disable gradient calculation for fast inference
        outputs = model(batch_tensor)
        probabilities = F.softmax(outputs[0], dim=0)
    
    # Get top_k probabilities and their corresponding class indices
    top_prob, top_indices = torch.topk(probabilities, top_k)
    
    predictions = []
    for prob, idx in zip(top_prob, top_indices):
        class_name = categories[idx.item()]
        # Format class name (e.g. 'golden_retriever' -> 'Golden Retriever')
        formatted_name = class_name.replace("_", " ").title()
        confidence = prob.item() * 100
        predictions.append({
            "class_name": formatted_name,
            "raw_label": class_name,
            "confidence": confidence,
            "index": idx.item()
        })
    
    return predictions


# ==============================================================================
# 2. CONSOLE / CLI MODE
# ==============================================================================

def run_cli(image_path=None):
    """
    Runs image classification from the command line.
    If no image_path is specified, tests all images in the 'sample_images' folder.
    """
    print("\n" + "=" * 65)
    print(" CODEORBIT TECH AI INTERNSHIP - TASK 3")
    print(" Image Classification Using Pretrained MobileNetV2")
    print("=" * 65)

    model, transforms_fn, categories = load_model()

    # Determine files to process
    sample_dir = Path(__file__).parent / "sample_images"
    test_files = []

    if image_path:
        p = Path(image_path)
        if not p.exists():
            print(f"[Error] File not found: {image_path}")
            return
        test_files.append(p)
    elif sample_dir.exists():
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        test_files = [f for f in sorted(sample_dir.iterdir()) if f.suffix.lower() in supported_exts]
        if not test_files:
            print(f"[Notice] No images found in '{sample_dir}'. Place some .jpg or .png images there.")
            return
    else:
        print(f"[Notice] Directory '{sample_dir}' not found. Please provide an image path.")
        return

    print(f"\n[3/3] Classifying {len(test_files)} image(s)...\n")

    for idx, img_file in enumerate(test_files, start=1):
        print("-" * 65)
        print(f"Image {idx}/{len(test_files)}: {img_file.name}")
        try:
            _, batch_tensor = preprocess_image(img_file, transforms_fn)
            predictions = classify_image(model, batch_tensor, categories, top_k=3)

            top = predictions[0]
            print(f"-> Predicted Label:   {top['class_name']}")
            print(f"-> Confidence Score:  {top['confidence']:.2f}%\n")
            print("   Top-3 Predictions:")
            for rank, pred in enumerate(predictions, start=1):
                bar = "█" * int(pred["confidence"] / 5)
                print(f"   {rank}. {pred['class_name']:<25} {pred['confidence']:>6.2f}%  {bar}")
        except Exception as e:
            print(f"   [Failed to process]: {e}")

    print("-" * 65)
    print("\nHow Pretrained MobileNetV2 Works:")
    print("1. Pretraining: The model was previously trained on 1.4+ million images (ImageNet).")
    print("2. Convolutional Layers: Extract hierarchical features (edges -> textures -> shapes -> object parts).")
    print("3. Inverted Residual Blocks: Use depthwise separable convolutions for high efficiency.")
    print("4. Classification Head: Fully connected layer outputs logits for 1000 object categories.")
    print("5. Softmax Activation: Converts raw logits into calibrated confidence percentages.")
    print("=" * 65 + "\n")


# ==============================================================================
# 3. TKINTER DESKTOP APPLICATION (GUI)
# ==============================================================================

def launch_gui():
    """
    Launches the desktop GUI built using standard Python Tkinter.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk

    # Initialize Tk root
    root = tk.Tk()
    root.title("CodeOrbit Tech AI Internship - Image Classification")
    root.geometry("680x780")
    root.minsize(580, 680)
    root.configure(bg="#F4F6F9")

    # Load Model (show loading screen first)
    status_var = tk.StringVar(value="Loading pretrained MobileNetV2 model...")

    # Header Frame
    header_frame = tk.Frame(root, bg="#1E293B", padx=20, pady=16)
    header_frame.pack(fill=tk.X)

    title_lbl = tk.Label(
        header_frame,
        text="Task 3: Pretrained Image Classifier",
        font=("Helvetica", 16, "bold"),
        fg="#FFFFFF",
        bg="#1E293B"
    )
    title_lbl.pack(anchor="w")

    subtitle_lbl = tk.Label(
        header_frame,
        text="CodeOrbit Tech AI Internship | Model: MobileNetV2 (ImageNet-1K)",
        font=("Helvetica", 10),
        fg="#94A3B8",
        bg="#1E293B"
    )
    subtitle_lbl.pack(anchor="w", pady=(2, 0))

    # Controls Frame (Buttons)
    btn_frame = tk.Frame(root, bg="#F4F6F9", pady=12, padx=20)
    btn_frame.pack(fill=tk.X)

    # Main Content Area
    content_frame = tk.Frame(root, bg="#FFFFFF", padx=20, pady=16, relief=tk.GROOVE, bd=1)
    content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

    # Image Preview Canvas
    img_lbl = tk.Label(
        content_frame,
        text="No image selected\n\nClick 'Choose Image from PC' or pick a sample below",
        font=("Helvetica", 11),
        fg="#64748B",
        bg="#F8FAFC",
        width=40,
        height=14,
        relief=tk.RIDGE
    )
    img_lbl.pack(pady=10)

    # Prediction Result Card
    result_card = tk.Frame(content_frame, bg="#EFF6FF", padx=16, pady=12, relief=tk.RIDGE, bd=1)
    result_card.pack(fill=tk.X, pady=8)

    pred_title = tk.Label(
        result_card,
        text="Predicted Classification:",
        font=("Helvetica", 10, "bold"),
        fg="#1E40AF",
        bg="#EFF6FF"
    )
    pred_title.pack(anchor="w")

    pred_label_var = tk.StringVar(value="Waiting for image...")
    pred_lbl = tk.Label(
        result_card,
        textvariable=pred_label_var,
        font=("Helvetica", 16, "bold"),
        fg="#0F172A",
        bg="#EFF6FF"
    )
    pred_lbl.pack(anchor="w", pady=(2, 2))

    pred_conf_var = tk.StringVar(value="")
    pred_conf_lbl = tk.Label(
        result_card,
        textvariable=pred_conf_var,
        font=("Helvetica", 11),
        fg="#2563EB",
        bg="#EFF6FF"
    )
    pred_conf_lbl.pack(anchor="w")

    # Top-3 Alternative Predictions Frame
    top3_frame = tk.Frame(content_frame, bg="#FFFFFF")
    top3_frame.pack(fill=tk.X, pady=(6, 0))

    top3_title = tk.Label(
        top3_frame,
        text="Top Predictions & Confidence Breakdown:",
        font=("Helvetica", 9, "bold"),
        fg="#475569",
        bg="#FFFFFF"
    )
    top3_title.pack(anchor="w", pady=(0, 4))

    top3_vars = [tk.StringVar(value="—") for _ in range(3)]
    for var in top3_vars:
        lbl = tk.Label(
            top3_frame,
            textvariable=var,
            font=("Consolas", 10),
            fg="#334155",
            bg="#F1F5F9",
            anchor="w",
            padx=8,
            pady=4
        )
        lbl.pack(fill=tk.X, pady=2)

    # Explanation Box
    expl_frame = tk.Frame(root, bg="#F1F5F9", padx=20, pady=8)
    expl_frame.pack(fill=tk.X, side=tk.BOTTOM)

    expl_text = (
        "How it works: MobileNetV2 was trained on ImageNet (1000 categories). "
        "It resizes images to 224x224, extracts feature patterns through depthwise "
        "convolutions, and outputs probabilities via Softmax."
    )
    expl_lbl = tk.Label(
        expl_frame,
        text=expl_text,
        font=("Helvetica", 8),
        fg="#64748B",
        bg="#F1F5F9",
        wraplength=620,
        justify="left"
    )
    expl_lbl.pack(anchor="w")

    # App State
    loaded_data = {"model": None, "transforms": None, "categories": None, "photo": None}

    def process_and_display(file_path):
        if not file_path:
            return
        try:
            # 1. Preprocess & Classify
            orig_img, batch_tensor = preprocess_image(file_path, loaded_data["transforms"])
            preds = classify_image(loaded_data["model"], batch_tensor, loaded_data["categories"], top_k=3)

            # 2. Update Image Display (fit neatly within 260x220)
            display_img = orig_img.copy()
            display_img.thumbnail((260, 220), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(display_img)
            loaded_data["photo"] = photo  # Keep reference to prevent garbage collection
            img_lbl.config(image=photo, text="", width=260, height=220)

            # 3. Update Results
            top = preds[0]
            pred_label_var.set(top["class_name"])
            pred_conf_var.set(f"Confidence: {top['confidence']:.2f}%")

            for i, p in enumerate(preds):
                bar = "█" * int(p["confidence"] / 6)
                top3_vars[i].set(f"{i+1}. {p['class_name']:<24} {p['confidence']:>6.2f}%  {bar}")

        except Exception as err:
            messagebox.showerror("Error Processing Image", f"Failed to process image:\n{err}")

    def select_file():
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
        chosen = filedialog.askopenfilename(
            title="Select an Image to Classify",
            filetypes=filetypes
        )
        if chosen:
            process_and_display(chosen)

    def test_sample(sample_name):
        sample_path = Path(__file__).parent / "sample_images" / sample_name
        if not sample_path.exists():
            messagebox.showwarning("Sample Not Found", f"Could not find '{sample_path}'.")
            return
        process_and_display(str(sample_path))

    # Add Action Buttons
    upload_btn = tk.Button(
        btn_frame,
        text="📁 Choose Image from PC...",
        font=("Helvetica", 10, "bold"),
        bg="#2563EB",
        fg="#FFFFFF",
        activebackground="#1D4ED8",
        activeforeground="#FFFFFF",
        padx=14,
        pady=6,
        relief=tk.FLAT,
        cursor="hand2",
        command=select_file
    )
    upload_btn.pack(side=tk.LEFT)

    # Sample Quick Buttons
    sample_dir = Path(__file__).parent / "sample_images"
    if sample_dir.exists():
        sample_files = [f.name for f in sorted(sample_dir.iterdir()) if f.suffix.lower() in {".jpg", ".png"}]
        if sample_files:
            tk.Label(btn_frame, text="Or quick-test sample:", font=("Helvetica", 9), bg="#F4F6F9", fg="#64748B").pack(side=tk.LEFT, padx=(16, 6))
            for s_name in sample_files[:4]:  # Show up to 4 quick test buttons
                label_clean = s_name.replace(".jpg", "").replace("_", " ").title()
                btn = tk.Button(
                    btn_frame,
                    text=label_clean,
                    font=("Helvetica", 9),
                    bg="#E2E8F0",
                    fg="#1E293B",
                    padx=8,
                    pady=4,
                    relief=tk.FLAT,
                    cursor="hand2",
                    command=lambda name=s_name: test_sample(name)
                )
                btn.pack(side=tk.LEFT, padx=3)

    # Load Model in Background / On Start
    try:
        m, t, c = load_model()
        loaded_data["model"] = m
        loaded_data["transforms"] = t
        loaded_data["categories"] = c
        status_var.set("Model Ready")

        # Auto-load the first sample image if available
        first_sample = Path(__file__).parent / "sample_images" / "dog.jpg"
        if first_sample.exists():
            process_and_display(str(first_sample))
    except Exception as e:
        messagebox.showerror("Model Load Error", f"Could not load MobileNetV2 model:\n{e}")

    root.mainloop()


# ==============================================================================
# 4. MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Check for CLI flags or headless container
    has_cli_arg = len(sys.argv) > 1 and (sys.argv[1] in ["--cli", "-c", "--help", "-h"] or os.path.isfile(sys.argv[1]))
    
    # Check if a graphical display environment exists
    can_use_gui = False
    if not has_cli_arg:
        try:
            import tkinter
            # Check DISPLAY variable on Linux/Unix
            if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
                can_use_gui = False
            else:
                can_use_gui = True
        except ImportError:
            can_use_gui = False

    if can_use_gui and not has_cli_arg:
        print("[Starting] Launching Tkinter Desktop Application...")
        launch_gui()
    else:
        # CLI fallback / Command line mode
        image_arg = None
        if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
            image_arg = sys.argv[1]
        elif len(sys.argv) > 1 and sys.argv[1] not in ["--cli", "-c"]:
            print(f"Usage: python main.py [optional_path_to_image] [--cli]")
        
        run_cli(image_arg)
