import torch
from torch.utils.data import DataLoader
from dataset import RoadsDataset

def get_loaders(
    train_dir, train_maskdir,
    val_dir, val_maskdir,
    test_dir, test_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    tile_size=512,
    train_stride=512,
    eval_stride=256,
):
    train_ds = RoadsDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        tile_size=tile_size,
        stride=train_stride,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_ds = RoadsDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
        tile_size=tile_size,
        stride=eval_stride,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_ds = RoadsDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=val_transform,
        tile_size=tile_size,
        stride=eval_stride,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

def update_iou_stats(logits, masks, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = probs > threshold

    masks = masks.bool()

    intersection = (preds & masks).sum()
    union = (preds | masks).sum()

    return intersection, union
