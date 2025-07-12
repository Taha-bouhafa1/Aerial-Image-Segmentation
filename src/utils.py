# src/utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_prediction(image, mask, pred):
    """
    Plot input image, ground truth, and prediction.
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image.permute(1, 2, 0).cpu())
    axs[0].set_title("Input Image")
    axs[1].imshow(mask.cpu(), cmap="tab10")
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred.cpu(), cmap="tab10")
    axs[2].set_title("Prediction")
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    plt.show()