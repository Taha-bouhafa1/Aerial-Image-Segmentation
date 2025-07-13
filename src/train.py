import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import trange
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.segmentation import DiceScore
from src.data.potsdam_dataset import PotsdamDataset, get_transforms
from src.models.unet3plus import UNet3Plus

BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
NUM_CLASSES = 6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = get_transforms()

full_dataset = PotsdamDataset(
    images_dir='data/processed/images',
    masks_dir='data/processed/masks_converted',
    transform=transform
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = UNet3Plus(n_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses, train_ious, train_dices = [], [], []
val_ious, val_dices = [], []

train_iou_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES).to(DEVICE)
train_dice_metric = DiceScore(num_classes=NUM_CLASSES).to(DEVICE)
val_iou_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES).to(DEVICE)
val_dice_metric = DiceScore(num_classes=NUM_CLASSES).to(DEVICE)

def training(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    train_iou_metric.reset()
    train_dice_metric.reset()

    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(images)
        loss = 0
        for out in outputs:
            loss += criterion(out, masks)
        loss = loss / len(outputs)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = torch.argmax(outputs[0], dim=1)
        train_iou_metric.update(preds, masks)
        train_dice_metric.update(preds, masks)

    return total_loss / len(loader), train_iou_metric.compute().item(), train_dice_metric.compute().item()


def evaluate(model, loader):
    model.eval()
    val_iou_metric.reset()
    val_dice_metric.reset()

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs[0], dim=1)  # only use y1
            val_iou_metric.update(preds, masks)
            val_dice_metric.update(preds, masks)

    return val_iou_metric.compute().item(), val_dice_metric.compute().item()

for epoch in trange(EPOCHS, desc="Epochs"):
    train_loss, train_iou, train_dice = training(model, train_loader, criterion, optimizer)
    val_iou, val_dice = evaluate(model, val_loader)

    train_losses.append(train_loss)
    train_ious.append(train_iou)
    train_dices.append(train_dice)
    val_ious.append(val_iou)
    val_dices.append(val_dice)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Train Dice: {train_dice:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}")

torch.save(model.state_dict(), 'model_unet3plus.pth')

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(train_ious, label='Train IoU')
plt.plot(val_ious, label='Val IoU', linestyle='--')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(train_dices, label='Train Dice')
plt.plot(val_dices, label='Val Dice', linestyle='--')
plt.grid(True)

plt.tight_layout()
plt.legend()
plt.savefig("training_curves.png")
plt.show()
