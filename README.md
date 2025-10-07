# Face Image Enhancement Pipeline

This project provides a simple pipeline to restore low-quality face images and unify their backgrounds.  
It combines **GFPGAN** for face restoration and **Rembg** for background removal to produce clean, high-resolution facial images with consistent backgrounds.

---

## 🧠 Overview

Low-quality or blurry facial images often suffer from poor details and inconsistent backgrounds.  
This project solves these problems in two stages:

1. **Face Restoration (GFPGAN)**  
   - Uses [GFPGAN](https://github.com/TencentARC/GFPGAN) to enhance and restore facial details.  
   - Improves image resolution and visual clarity.  
   - Works well on old, compressed, or low-resolution photos.

2. **Background Unification (Rembg)**  
   - Uses [Rembg](https://github.com/danielgatis/rembg) to remove or replace backgrounds.  
   - Produces consistent, clean outputs with transparent or solid-color backgrounds.  

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/face-enhancement-pipeline.git
cd face-enhancement-pipeline
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:
```
gfpgan
rembg
opencv-python
numpy
```

*(Optional: you may need `torch` and `torchvision` if not installed automatically by GFPGAN)*

---

## 🚀 Usage

### 1. Restore Faces with GFPGAN
```bash
python restore_faces.py --input input_images/ --output restored_faces/
```

### 2. Remove/Unify Backgrounds with Rembg
```bash
python unify_background.py --input restored_faces/ --output final_images/
```

---

## 🧩 Example Workflow

1. **Input**: Low-quality face image  
2. **GFPGAN Output**: High-resolution, restored face  
3. **Rembg Output**: Background removed or unified (transparent or white)  

| Step | Example |
|------|----------|
| Input | ![input](examples/input.jpg) |
| GFPGAN Output | ![restored](examples/restored.jpg) |
| Final (Rembg) | ![final](examples/final.png) |

---

## 🪄 Customization

- You can modify the background after removal using tools like **Pillow** or **OpenCV**.
- If you want a solid background instead of transparency:
  ```bash
  rembg i --background white input.png output.png
  ```

---

## 📁 Project Structure

```
face-enhancement-pipeline/
│
├── input_images/          # Original low-quality face images
├── restored_faces/        # GFPGAN outputs
├── final_images/          # Rembg outputs
│
├── restore_faces.py       # Script for GFPGAN restoration
├── unify_background.py    # Script for Rembg background processing
│
├── requirements.txt
└── README.md
```

---

## 💡 Notes

- GFPGAN can be GPU-accelerated if CUDA is available.
- For batch processing, both scripts can be easily adapted to handle multiple images.
- You can integrate the pipeline into web apps or data preprocessing workflows.

---

## 🧾 License

This project follows the open-source licenses of **GFPGAN** and **Rembg**.  
Please refer to their repositories for license details.
