# U-Net3+ with Deep Supervision for Semantic Segmentation of Aerial Images (Potsdam)


## Project Overview

This project implements semantic segmentation using the **U-Net3+ architecture** with optional **deep supervision** on the **ISPRS Potsdam** dataset. The goal is to classify each pixel in high-resolution aerial RGB images into one of several land cover classes.

![Potsdam Overview](https://github.com/Taha-bouhafa1/Aerial-Image-Segmentation/blob/main/assets/potsdam.png)

---

## Dataset

- **Source**: [2D Semantic Labeling Contest – Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
- The Potsdam dataset contains ultra high-resolution aerial RGB images (6000×6000) and their pixel-wise semantic segmentation masks.
- Images were tiled into **256x256** or **512x512** patches depending on the training configuration.
- The label masks contain the following classes:

| Class               | RGB Value         |
|---------------------|-------------------|
| Impervious surfaces | (255, 255, 255)   |
| Building            | (0, 0, 255)       |
| Low vegetation      | (0, 255, 255)     |
| Tree                | (0, 255, 0)       |
| Car                 | (255, 255, 0)     |
| Clutter/background  | (255, 0, 0)       |

### Example Samples

Each sample below shows the original image (left) and its corresponding ground truth mask (right):

![Image1](https://github.com/Taha-bouhafa1/Aerial-Image-Segmentation/blob/main/assets/image1.jpg)  
![Image2](https://github.com/Taha-bouhafa1/Aerial-Image-Segmentation/blob/main/assets/image2.jpg)  
![Image3](https://github.com/Taha-bouhafa1/Aerial-Image-Segmentation/blob/main/assets/image3.jpg)  

---

## U-Net3+ Architecture

This project uses **U-Net3+**, a powerful segmentation model that enhances the classic U-Net by introducing full-scale skip connections across different levels of the encoder and decoder. This helps the network better preserve fine-grained spatial details.

![U-Net3+ Architecture](https://github.com/Taha-bouhafa1/Aerial-Image-Segmentation/blob/main/assets/unet3%2B.png)

---

###  Deep Supervision 

We also implement **deep supervision**, where auxiliary outputs are generated from intermediate decoder stages and supervised during training. This leads to better convergence and regularization.

![Deep Supervision](https://github.com/Taha-bouhafa1/Aerial-Image-Segmentation/blob/main/assets/Deep%20Supervision.png)

You can enable/disable deep supervision easily in the code.

---

## 📁 Project Structure

```bash
Aerial-Image-Segmentation/
│
├── README.md
├── assets/ # Images used in README (architecture, examples)
├── data/
│ ├── raw/ # Original Potsdam images and masks
│ └── processed/
│ ├── images/ # Tiled image patches (512x512)
│ ├── masks/ # Tiled RGB masks
│ └── masks_converted/ # Class-indexed masks
│
├── outputs/ # Logs, predictions, saved checkpoints
├── training.ipynb # Jupyter notebook (optional exploration)
│
└── src/
├── data/
│ ├── potsdam_dataset.py # Custom PyTorch Dataset class
│ ├── tile_images.py # Script to tile large 6000x6000 images
│ └── label_converter.py # Script to convert RGB masks to class indices
│
├── model/
│ └── unet3plus.py # U-Net3+ model with deep supervision support
│
├── train.py # Main training script
└── utils.py # Utility functions (metrics, visualization, etc.)

```


---

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Prepare the dataset:

   Place raw images and masks in data/raw/images and data/raw/masks

   Run tiling and mask conversion:
   
```bash
python src/data/tile_images.py
python src/data/label_converter.py
```
---

##  How to Train

You can train the model by running the following command:

```bash
!python src/train.py
```
---

##  Reference

Refer to the official paper for more technical details:  
[**UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation**](https://arxiv.org/pdf/2004.08790)  

---
## Citation
```bash
@misc{bouhafa2025unet3plus,
  author       = {Taha Bouhafa},
  title        = {U-Net3+ with Deep Supervision for Semantic Segmentation of Aerial Images (Potsdam)},
  year         = {2025},
  howpublished = {\url{https://github.com/Taha-bouhafa1/Aerial-Image-Segmentation}},
  note         = {GitHub repository}
}
```
