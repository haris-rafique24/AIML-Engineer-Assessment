import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from data_loader import BCCDDataset
from model_factory import get_model
import config

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(), # Converts PIL Image to Tensor
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Pass the transform here to fix the AttributeError
    full_dataset = BCCDDataset(root=config.DATA_ROOT, transforms=get_transform())
    
    # --- Proper Train/Validation Split (80-20)
    dataset_size = len(full_dataset)
    indices = torch.randperm(dataset_size).tolist()
    train_split = int(0.8 * dataset_size)
    
    train_indices = indices[:train_split]
    val_indices = indices[train_split:]

    train_loader = DataLoader(
        Subset(full_dataset, train_indices), 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    model = get_model(num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=config.LR, 
        momentum=config.MOMENTUM, 
        weight_decay=config.WEIGHT_DECAY
    )

    print(f"Starting training on {config.DEVICE}...")
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        epoch_loss = 0
        for images, targets in train_loader:
            # Now images are Tensors and .to() will work
            images = list(image.to(config.DEVICE) for image in images)
            targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Loss: {epoch_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()