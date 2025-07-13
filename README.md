# Semantic Segmentation with U-Net3+ on Potsdam Dataset

## Project Overview

This project implements semantic segmentation using the U-Net3+ architecture with optional deep supervision on the Potsdam dataset. The goal is to classify each pixel in high-resolution aerial RGB images into one of several land cover classes.

![Potsdam Overview](https://github.com/Taha-bouhafa1/Aerial-Image-Segmentation/blob/main/assets/potsdam.png)

---

## Dataset

- **Source**: [2D Semantic Labeling Contest – Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
- The Potsdam dataset consists of very high-resolution aerial RGB images (6000×6000) and corresponding labeled masks.
- The dataset includes the following classes:

| Class               | RGB Value         |
|---------------------|-------------------|
| Impervious surfaces | (255, 255, 255)   |
| Building            | (0, 0, 255)       |
| Low vegetation      | (0, 255, 255)     |
| Tree                | (0, 255, 0)       |
| Car                 | (255, 255, 0)     |
| Clutter/background  | (255, 0, 0)       |

- Images and masks were tiled into patches of size **512x512** (or 256x256 based on config) for training.

### Example Samples

Each sample below includes the original image on the left and its corresponding ground truth mask on the right:
 
 ![Image1](https://github.com/Taha-bouhafa1/Aerial-Image-Segmentation/blob/main/assets/image1.jpg)  
 
 ![Image2](https://github.com/Taha-bouhafa1/Aerial-Image-Segmentation/blob/main/assets/image2.jpg) 
 
![Image3](https://github.com/Taha-bouhafa1/Aerial-Image-Segmentation/blob/main/assets/image3.jpg) 

---

## Repository Structure

- `data/raw/` – Original dataset images and masks  
- `data/processed/images/` – Processed and tiled image patches  
- `data/processed/masks_converted/` – Corresponding class index masks  
- `src/data/potsdam_dataset.py` – Custom PyTorch `Dataset` class and image transformations  
- `src/models/unet3plus.py` – U-Net3+ model implementation with deep supervision  
- `src/train.py` – Training loop  
- `scripts/tile_images.py` – Script to tile large images  
- `scripts/label_converter.py` – Script to convert RGB masks to class indices  

---

## Setup

1. Install required packages:
   ```bash
   pip install -r requirements.txt
