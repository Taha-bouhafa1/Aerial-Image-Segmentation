# Semantic Segmentation with U-Net3+ on Potsdam Dataset

## Project Overview

This project implements semantic segmentation using the U-Net3+ architecture on the Potsdam dataset. The goal is to classify each pixel in high-resolution aerial images into one of several land cover classes.

## Dataset

- The Potsdam dataset contains high-resolution aerial RGB images and corresponding labeled masks.
- Images and masks are preprocessed into 512x512 pixel tiles for training.

## Repository Structure

- `data/raw/` - Original dataset images and masks.
- `data/processed/images/` - Processed and tiled image patches.
- `data/processed/masks_converted/` - Corresponding class index masks.
- `src/data/potsdam_dataset.py` - Custom PyTorch Dataset class and image transforms.
- `src/models/unet3plus.py` - U-Net3+ model implementation.
- `src/train.py` - Training script.
- `scripts/tile_images.py` - Script to tile large images into patches.
- `scripts/label_converter.py` - Converts color-coded masks into class index masks.

## Setup

1. Install required packages:
    ```
    pip install -r requirements.txt
    ```
2. Prepare the dataset:
    - Place raw images and masks in `data/raw/`
    - Run `tile_images.py` to create tiles
    - Run `label_converter.py` to convert masks to class indices

## Training

- Modify hyperparameters in `train.py` as needed.
- Run training:
    ```
    python src/train.py
    ```

## Evaluation

- The model is evaluated using metrics such as IoU and Dice coefficient.

## License

This project is licensed under the MIT License.
