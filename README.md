# Semantic Segmentation with U-Net3+ on Potsdam Dataset

## Project Overview

This project implements semantic segmentation using the U-Net3+ architecture with optional deep supervision on the Potsdam dataset. The goal is to classify each pixel in high-resolution aerial RGB images into one of several land cover classes.

## Dataset

- The Potsdam dataset consists of high-resolution aerial RGB images and corresponding labeled masks.
- Images and masks are preprocessed into tiles of size 512x512 pixels for training.

## Repository Structure

- `data/raw/` - Original dataset images and masks.
- `data/processed/images/` - Processed and tiled image patches.
- `data/processed/masks_converted/` - Corresponding class index masks.
- `src/data/potsdam_dataset.py` - Custom PyTorch Dataset class and image transformations.
- `src/models/unet3plus.py` - U-Net3+ model implementation supporting deep supervision.
- `src/train.py` - Training script.
- `scripts/tile_images.py` - Script to split large images into smaller tiles.
- `scripts/label_converter.py` - Script to convert color-coded masks into class index masks.

## Setup

1. Install required packages:
    ```
    pip install -r requirements.txt
    ```
2. Prepare the dataset:
    - Place raw images and masks into the `data/raw/` directory.
    - Run `tile_images.py` to generate image and mask tiles.
    - Run `label_converter.py` to convert color-coded masks into class index masks.

## Training

- Modify hyperparameters (such as batch size, learning rate, epochs) in `src/train.py` as needed.
- Enable or disable deep supervision via the model initialization.
- To start training, run:
    ```
    python src/train.py
    ```

## Evaluation

- The model is evaluated using metrics such as Intersection over Union (IoU) and Dice coefficient.

## Checkpoints

- Model checkpoints are saved periodically during training to enable resuming.

## License

This project is licensed under the MIT License.

