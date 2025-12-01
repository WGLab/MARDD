# GestaltM³D-VL: Multimodal Vision–Language Diagnosis of Mendelian Rare Diseases

> **GestaltM³D-VL** (Gestalt Multimodal Model for Mendelian Disease Diagnosis with Vision–Language Integration)  
> A multimodal model leveraging **facial images** and **HPO-encoded clinical text** for **Mendelian rare disease diagnosis**, with **StyleGAN3-based image synthesis and privacy evaluation**.

---

## Teaser & Key Figures

> 💡 本 README 假设你会在 `docs/figures/` 下放置对应图片文件。  
> 下面是推荐的图文件名与 README 中的引用方式，你可以用 PPT 导出或重新画图后，保存为相同路径即可，GitHub 会自动渲染。

### Overall pipeline

<p align="center">
  <img src="docs/figures/overview_pipeline.png" alt="Overall pipeline of GestaltM3D-VL" width="800">
</p>

### Example faces & HPO text

<p align="center">
  <img src="docs/figures/example_cases.png" alt="Example GMDB cases with facial images and HPO descriptions" width="800">
</p>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset: GMDB Ten-Disease Subset](#2-dataset-gmdb-ten-disease-subset)
3. [Data Preprocessing Pipeline](#3-data-preprocessing-pipeline)
4. [Part I – StyleGAN3 Synthesis & Privacy Evaluation](#4-part-i--stylegan3-synthesis--privacy-evaluation)
5. [Part II – GestaltM³D-VL for Multimodal Diagnosis](#5-part-ii--gestaltm³d-vl-for-multimodal-diagnosis)
6. [Repository Structure](#6-repository-structure)
7. [Installation](#7-installation)
8. [Data Preparation](#8-data-preparation)
9. [Training & Evaluation](#9-training--evaluation)
10. [Figures to Include](#10-figures-to-include)
11. [Citation](#11-citation)
12. [License & Disclaimer](#12-license--disclaimer)

---

## 1. Project Overview

This repository implements the pipeline for **GMDB ten-disease** and **CHOP craniofacial/genetics** cohorts:

1. **Part I – StyleGAN3-based Image Synthesis & Privacy Evaluation**
   - Train **class-conditional StyleGAN3** on:
     - GMDB ten-disease cohort
     - Large unaffected/healthy cohort
   - Generate realistic facial images for **rare Mendelian syndromes**.
   - Perform **match attacks** using ArcFace embeddings to evaluate **identity leakage** when synthetic faces are used.

2. **Part II – GestaltM³D-VL (Vision–Language Multimodal Diagnosis)**
   - Adapt **Qwen-2-VL / Qwen-2.5-VL / Qwen-3-VL** into a **sequence classification** model.
   - Inputs:
     - **Facial image** (preprocessed 224×224 front-view face).
     - **Clinical text**: HPO-encoded phenotypes (+/- demographics).
   - Output:
     - **Disease label** in a long-tailed label space dominated by rare Mendelian syndromes.
   - Loss:
     - **Class-Balanced Focal Loss** to focus on underrepresented diseases.

<p align="center">
  <img src="docs/figures/method_block_diagram.png" alt="High-level block diagram of the full method" width="800">
</p>

---

## 2. Dataset: GMDB Ten-Disease Subset

We use the **GestaltMatcher Database (GMDB)**, which pairs:

- **Facial images**
- **HPO-encoded clinical text**
- **Demographic information** (ethnicity, age, sex)

Label space: **528 syndromes/disorders** (244 frequent, 284 rare).  
In this repository we focus on the **ten diseases with the most distinctive facial phenotypes**, curated by clinical geneticists.

Key stats for the **ten-disease subset**:

- **1,847 cases** with patient-level splits.
- **Test set**: only **HPO-annotated cases** (≈ 818 cases).
- Non-HPO cases are used **only as data augmentation** during training.

> ⚠️ **Dataset access**  
> GMDB and CHOP cohorts are **not included** in this repository.  
> You must obtain data access following your **institutional, IRB, and data-use agreements**, then place the files under `data/raw/`.

<p align="center">
  <img src="docs/figures/dataset_stats.png" alt="Label distribution and cohort statistics for the GMDB ten-disease subset" width="800">
</p>

---

## 3. Data Preprocessing Pipeline

Original facial images in GMDB/CHOP often suffer from:

- Low resolution, blur
- Grayscale or washed-out colors
- Cluttered backgrounds

To address this, we build a **three-stage preprocessing pipeline**:

1. **Color Restoration – DDColor**
   - Colorize grayscale / washed images.

2. **Face Restoration & Super-Resolution – GFPGAN**
   - Enhance facial details and upsample low-resolution faces.

3. **Face Detection & Normalization – MediaPipe Face Detection**
   - Detect faces and perform **tight cropping**.
   - Remove background and resize to **224×224** RGB.
   - Normalize the input space and control token budget for the multimodal LLM.

The script `src/data/preprocess_faces.py` orchestrates DDColor, GFPGAN, and MediaPipe and writes processed images into `data/processed/images`.

<p align="center">
  <img src="docs/figures/preprocessing_pipeline.png" alt="DDColor + GFPGAN + MediaPipe facial preprocessing pipeline" width="800">
</p>

---

## 4. Part I – StyleGAN3 Synthesis & Privacy Evaluation

### 4.1 StyleGAN3 Overview

We use **StyleGAN3** as a **class-conditional generator**:

- Alias-free generation with improved geometric consistency.
- Strong support for face synthesis with fewer texture-artifacts than StyleGAN2.
- Separate **mapping network** and **synthesis network**, with class conditioning.

Training data:

- **Ten-disease cohort**: 1,847 patients.
- **Unaffected / healthy cohort**: O(10k) individuals  
  → mitigates distribution shift and age bias.

<p align="center">
  <img src="docs/figures/stylegan3_overview.png" alt="StyleGAN3 architecture and training overview" width="800">
</p>

### 4.2 Kept vs Held-Out Split (In-/Out-of-Distribution)

We define:

- **In-Distribution (ID)**: patients **kept** in StyleGAN3 training.
- **Out-of-Distribution (OOD)**: patients **held-out** and **removed** from StyleGAN3 training.

For each disease:

1. Split into **Kept (train)** and **Held-out (removed)** according to predefined ratios.
2. For each configuration, **retrain StyleGAN3 from scratch** using only Kept samples.
3. Sample a **large synthetic probe set** per class.

<p align="center">
  <img src="docs/figures/kept_vs_heldout.png" alt="Kept vs held-out partition for privacy experiments" width="800">
</p>

### 4.3 Training & Sampling

Configuration sketch (see `configs/stylegan3/ten_disease_default.yaml`):

- Training iterations: e.g., **25,000 kimg**
- Conditional labels: { ten-disease classes + unaffected class }
- Augmentations: StyleGAN3 default configuration

Run:

```bash
bash scripts/train_stylegan3.sh
bash scripts/sample_stylegan3.sh
```

### 4.4 Match Attack & Privacy Evaluation

We collaborate with a security/ML group to perform **match attacks**:

1. Gather **real** CdLS and other ten-disease-class images (ID & OOD).
2. Generate **synthetic** faces via StyleGAN3.
3. Extract **ArcFace embeddings** for all real & synthetic images.
4. Compute similarity-based metrics to quantify **identity leakage**.

Code entry point:

- `src/models/stylegan3/privacy_match_attack.py`
- `scripts/run_match_attack.sh`

<p align="center">
  <img src="docs/figures/stylegan3_privacy.png" alt="Match attack framework on StyleGAN3 synthetic faces" width="800">
</p>

---

## 5. Part II – GestaltM³D-VL for Multimodal Diagnosis

### 5.1 Motivation

We target **Mendelian rare disease diagnosis** from:

- **Facial gestalt** (preprocessed front-face images).
- **HPO-encoded clinical phenotypes** (+/- demographics).

Challenges:

- GMDB is **long-tailed** (hundreds of rare diseases with few samples).
- Data is **noisy** (missing text, variable image quality).
- Naive cross-entropy tends to overfit **head classes**.

### 5.2 Backbone: Qwen-VL Family

We build on **Qwen-VL** models:

- **Qwen-2-VL-7B-Instruct**
- **Qwen-2.5-VL-7B-Instruct** (current default backbone)
- Planned **Qwen-3-VL-8B-Instruct** support

Advantages:

- Strong **vision–language alignment**.
- Good **parameter–compute trade-off** (7–8B).
- Handles **long clinical prompts** and **noisy facial inputs**.

### 5.3 Model Architecture (Sequence Classification)

GestaltM³D-VL adapts Qwen-VL into **multimodal sequence classification**:

1. Inputs:
   - **Image**: 224×224 face.
   - **Text**:  
     - HPO-encoded phenotypes  
     - Optionally demographics (we often **drop** demographics for privacy and robustness).
2. Qwen-VL encoder processes the joint image-text prompt.
3. We extract the hidden state at a special **`[CLS]` token**.
4. Feed into a **Linear(D → #classes)** layer.
5. Train with **Class-Balanced Focal Loss**.

Training strategy:

- Freeze **text encoder** (and optionally some vision blocks).
- Train:
  - Multimodal projector
  - Classifier head
  - Optionally top layers via **LoRA**.

<p align="center">
  <img src="docs/figures/mm_llm_architecture.png" alt="GestaltM3D-VL multimodal sequence classification architecture" width="800">
</p>

### 5.4 Loss: Class-Balanced Focal Loss

To cope with long-tailed labels:

- Let \( n_y \) be the number of samples of class \( y \).
- We compute **effective class weights** \( \alpha_y \) based on inverse frequency.
- Use **focal loss** with focusing parameter \( \gamma \) to emphasize hard examples.

Implementation:

- `src/models/mm_llm/losses.py`

### 5.5 Observations (from internal experiments)

- **Multimodal > Image-only**:  
  GestaltM³D-VL with **image+text** consistently outperforms image-only baselines.
- **Dropping demographics**:
  - Comparable or better performance.
  - Avoids explicit use of race/ethnicity, improving **fairness & privacy**.
- **Model upgrade**:
  - Qwen-2.5-VL backbone performs better than Qwen-2-VL.

---

## 6. Repository Structure

A recommended repository layout:

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── configs/
│   ├── stylegan3/
│   │   └── ten_disease_default.yaml
│   └── mm_llm/
│       └── qwen2_5_vl_ten_disease.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│       └── ten_disease/
│           ├── train.csv
│           ├── val.csv
│           └── test.csv
├── docs/
│   └── figures/
│       ├── overview_pipeline.png
│       ├── example_cases.png
│       ├── dataset_stats.png
│       ├── preprocessing_pipeline.png
│       ├── stylegan3_overview.png
│       ├── kept_vs_heldout.png
│       ├── stylegan3_privacy.png
│       ├── mm_llm_architecture.png
│       └── method_block_diagram.png
├── src/
│   ├── data/
│   │   ├── gmdb_dataset.py
│   │   ├── preprocess_faces.py
│   │   └── build_splits.py
│   ├── models/
│   │   ├── stylegan3/
│   │   │   ├── train_stylegan3.py
│   │   │   ├── sample_stylegan3.py
│   │   │   └── privacy_match_attack.py
│   │   └── mm_llm/
│   │       ├── mm_classifier.py
│   │       ├── losses.py
│   │       ├── train_mm_llm.py
│   │       └── evaluate_mm_llm.py
│   └── utils/
│       ├── config.py
│       ├── logging_utils.py
│       ├── metrics.py
│       └── seed.py
└── scripts/
    ├── prepare_gmdb.sh
    ├── train_stylegan3.sh
    ├── sample_stylegan3.sh
    ├── run_match_attack.sh
    ├── train_mm_llm.sh
    └── eval_mm_llm.sh
```

---

## 7. Installation

### 7.1 Conda environment

```bash
conda env create -f environment.yml
conda activate gestaltm3d-vl
```

or using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` (sketch):

- `torch`, `torchvision`, `torchaudio`
- `transformers`, `accelerate`, `bitsandbytes` (optional)
- `timm`
- `opencv-python`, `mediapipe`
- `gfpgan`, `ddcolor` (or equivalent packages)
- `scikit-learn`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `rich`, `pyyaml`

---

## 8. Data Preparation

### 8.1 Place raw data

1. Place GMDB / CHOP metadata and images under:

```text
data/raw/
  gmdb_metadata.csv
  images/
    patient_*.jpg
```

> The exact file naming depends on your internal data export.  
> Adjust paths in `src/data/build_splits.py` and `src/data/preprocess_faces.py` if needed.

### 8.2 Build patient-level splits

```bash
python -m src.data.build_splits \
  --input_meta data/raw/gmdb_metadata.csv \
  --output_dir data/splits/ten_disease
```

This will generate:

- `train.csv`
- `val.csv`
- `test.csv`

with columns such as:

- `image_path`
- `disease_label` (int)
- `hpo_text`
- `demographics` (optional)

### 8.3 Run facial preprocessing

```bash
python -m src.data.preprocess_faces \
  --input_dir data/raw/images \
  --meta_csv data/splits/ten_disease/train.csv \
  --output_dir data/processed/images
```

Repeat for val/test or integrate full-split logic in the script.

---

## 9. Training & Evaluation

### 9.1 StyleGAN3

```bash
# Train StyleGAN3 on GMDB ten-disease + unaffected cohort
bash scripts/train_stylegan3.sh

# Sample synthetic faces
bash scripts/sample_stylegan3.sh

# Run match attack & privacy evaluation
bash scripts/run_match_attack.sh
```

Make sure to configure:

- Dataset paths
- Class labels
- Kept vs held-out partitions

in `configs/stylegan3/ten_disease_default.yaml`.

### 9.2 GestaltM³D-VL (Multimodal Diagnosis)

```bash
# Train GestaltM3D-VL
bash scripts/train_mm_llm.sh

# Evaluate on held-out test set
bash scripts/eval_mm_llm.sh
```

`configs/mm_llm/qwen2_5_vl_ten_disease.yaml` controls:

- Backbone choice (Qwen-2-VL vs Qwen-2.5-VL vs Qwen-3-VL)
- Learning rate & batch size
- Which modules to freeze / finetune
- Whether demographics are used in the text prompt

---

## 10. Figures to Include

To make the README visually complete on GitHub, we recommend preparing the following images under `docs/figures/`:

1. `overview_pipeline.png`  
   - End-to-end diagram:  
     data → preprocessing → StyleGAN3 → privacy → GestaltM³D-VL training/eval.

2. `example_cases.png`  
   - A grid with example patient faces & short phenotype descriptions (de-identified / synthetic if needed).

3. `dataset_stats.png`  
   - Bar plot / histogram of label counts, plus cohort statistics.

4. `preprocessing_pipeline.png`  
   - Before/after examples for DDColor + GFPGAN + MediaPipe cropping.

5. `stylegan3_overview.png`  
   - StyleGAN3 architecture sketch and training scheme.

6. `kept_vs_heldout.png`  
   - Schematic of Kept vs Held-out partition for privacy experiments.

7. `stylegan3_privacy.png`  
   - Match attack / ArcFace embedding-based privacy evaluation workflow.

8. `mm_llm_architecture.png`  
   - GestaltM³D-VL vision–language classification architecture with `[CLS]` head.

9. `method_block_diagram.png`  
   - High-level method block diagram (two towers: generator & classifier).

You can export these directly from your PPT or recreate them in any vector-graphics tool.

---

## 11. Citation

If you use this codebase or ideas in your research, please cite the relevant GMDB / GestaltMatcher / Qwen-VL works (placeholder format):

```bibtex
@article{your_gestaltm3dvl_paper,
  title   = {GestaltM3D-VL: Multimodal Vision--Language Diagnosis of Mendelian Rare Diseases},
  author  = {Your Name and Collaborators},
  journal = {To appear},
  year    = {2025}
}
```

And also consider citing:

- GestaltMatcher / GestaltMML papers
- Qwen2-VL / Qwen2.5-VL / Qwen3-VL technical reports
- StyleGAN3, DDColor, GFPGAN, MediaPipe

---

## 12. License & Disclaimer

- **License**: To be decided (e.g., MIT / Apache-2.0 / custom research license).  
- **Data**: This repository does **not** contain patient-identifiable data.  
- **Ethics**:
  - Any use of real patient data must follow appropriate **IRB**, **data-use agreements**, and **local regulations**.
  - Synthetic images generated by StyleGAN3 can still leak information; always perform privacy analysis before public release.

> This repository is intended for **research** and **method development** only and should **not** be used as a stand-alone medical device or for clinical decision-making without proper validation and regulatory approval.
