# LJC_Stripformer (Stripformer-based Image Deblurring)

This repository contains **research & demo code** for image deblurring based on **Stripformer**.  
Some modules and overall structure are adapted from the **official Stripformer implementation**.

> If you use this code for research, please also cite and credit the original Stripformer work.

---

## 🔍 Overview
- **Task**: Image Deblurring (GoPro, HIDE)
- **Backbone**: Stripformer / Stripformer variants (including cross-attention experiments)
- **Framework**: Python + PyTorch
- **Repository**: https://github.com/JcLee96/LJC_Stripformer

---

## 📁 Repository Structure
```text
.
├─ train_Stripformer_cross_att_t2.py        # training entry
├─ test_Stripformer_gopro.py                # test entry
├─ predict_GoPro_test_results.py            # prediction / result export
├─ extract_result.py                        # result extraction utilities
├─ metric_counter.py                        # metrics counter
├─ dataset.py                               # dataset loader
├─ aug.py                                   # augmentation
├─ config/
│   ├─ config_Stripformer_gopro.yaml
│   ├─ config_Stripformer_gopro2.yaml
│   ├─ config_Stripformer_gopro3.yaml
│   ├─ config_Stripformer_gopro4.yaml
│   └─ config_Stripformer_pretrained.yaml
├─ models/
│   ├─ Stripformer.py
│   ├─ Stripformer_cross_att.py
│   ├─ networks.py / blocks.py / losses.py ...
├─ util/
│   ├─ metrics.py / util.py / visualizer.py ...
└─ Pre_trained_model/
    └─ Stripformer_gopro.pth

## 📁 Training/Test Process

################################################################################
# LJC_Stripformer : One-shot setup / download / train / test
################################################################################

# 1. Clone repository
git clone https://github.com/JcLee96/LJC_Stripformer.git
cd LJC_Stripformer

# 2. Create conda environment
conda create -n ljc_stripformer python=3.10 -y
conda activate ljc_stripformer

# 3. Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install python dependencies
pip install -r requirements.txt

################################################################################
# 5. Official repositories & dataset links (manual download)
################################################################################

# Stripformer (official)
# https://github.com/pp00704831/Stripformer

# Stripformer pretrained weights (GoPro)
# https://drive.google.com/drive/folders/1-4v8R8iYyqP4n8l7G3YhH0p2t3c2vX6P
# → Download: Stripformer_gopro.pth
# → Place to: Pre_trained_model/Stripformer_gopro.pth

# GoPro dataset (official)
# https://seungjunnah.github.io/Datasets/gopro
# https://drive.google.com/drive/folders/1HczByhAj9h6A3X1K_xlZt4lGvZp1pFzZ

# HIDE dataset (official)
# https://github.com/joanshen0508/HIDE
# https://drive.google.com/drive/folders/1rLZs5E_JoBFeoJEB6Digw1k3DybZ0_Sp

################################################################################
# 6. Train
################################################################################

python train_Stripformer_cross_att_t2.py \
  --config config/config_Stripformer_gopro.yaml

################################################################################
# 7. Test (GoPro)
################################################################################

python test_Stripformer_gopro.py \
  --config config/config_Stripformer_gopro.yaml \
  --weights Pre_trained_model/Stripformer_gopro.pth

################################################################################
# 8. Predict / Export results
################################################################################

python predict_GoPro_test_results.py \
  --config config/config_Stripformer_gopro.yaml \
  --weights Pre_trained_model/Stripformer_gopro.pth

python extract_result.py


