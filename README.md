# LJC_Stripformer (Stripformer-based Image Deblurring)

This repository contains **research & demo code** for image deblurring based on **Stripformer**,  
and includes **partial implementations and experimental code** from the paper:

**“DELEca: Deblurring for Long and Short Exposure Images with a Dual-Branch Multimodal Cross-Attention Mechanism.”**

Some modules and the overall structure are adapted from the **official Stripformer implementation**  
and extended for **dual-branch and cross-attention based deblurring experiments**.

> If you use this code for research, please also cite and credit both  
> the original **Stripformer** work and the **DELEca** paper.

---

## 🔍 Overview
- **Task**: Image Deblurring (GoPro, HIDE)
- **Backbone**: Stripformer / Stripformer variants (including cross-attention experiments)
- **Framework**: Python + PyTorch
- **Repository**: https://github.com/JcLee96/LJC_Stripformer

---

## 📁 Repository Structure & Training / Test Process
```bash
################################################################################
# Repository Structure
################################################################################
# .
# ├─ train_Stripformer_cross_att_t2.py
# ├─ test_Stripformer_gopro.py
# ├─ predict_GoPro_test_results.py
# ├─ extract_result.py
# ├─ metric_counter.py
# ├─ dataset.py
# ├─ aug.py
# ├─ config/
# │   ├─ config_Stripformer_gopro.yaml
# │   ├─ config_Stripformer_gopro2.yaml
# │   ├─ config_Stripformer_gopro3.yaml
# │   ├─ config_Stripformer_gopro4.yaml
# │   └─ config_Stripformer_pretrained.yaml
# ├─ models/
# │   ├─ Stripformer.py
# │   ├─ Stripformer_cross_att.py
# │   ├─ networks.py / blocks.py / losses.py
# ├─ util/
# │   ├─ metrics.py / util.py / visualizer.py
# └─ Pre_trained_model/
#     └─ Stripformer_gopro.pth

################################################################################
# One-shot setup / download / train / test
################################################################################

# Clone repository
git clone https://github.com/JcLee96/LJC_Stripformer.git
cd LJC_Stripformer

# Create conda environment
conda create -n ljc_stripformer python=3.10 -y
conda activate ljc_stripformer

# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

################################################################################
# Official resources
################################################################################

# Stripformer (official)
# https://github.com/pp00704831/Stripformer

# Pretrained weights (GoPro)
# https://drive.google.com/drive/folders/1-4v8R8iYyqP4n8l7G3YhH0p2t3c2vX6P

# GoPro dataset
# https://seungjunnah.github.io/Datasets/gopro

# HIDE dataset
# https://github.com/joanshen0508/HIDE

################################################################################
# Train
################################################################################
python train_Stripformer_cross_att_t2.py \
  --config config/config_Stripformer_gopro.yaml

################################################################################
# Test
################################################################################
python test_Stripformer_gopro.py \
  --config config/config_Stripformer_gopro.yaml \
  --weights Pre_trained_model/Stripformer_gopro.pth

################################################################################
# Predict / Export results
################################################################################
python predict_GoPro_test_results.py \
  --config config/config_Stripformer_gopro.yaml \
  --weights Pre_trained_model/Stripformer_gopro.pth

python extract_result.py
