import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from data_loader import BCCDDataset
from model_factory import get_model
import config

# Define the transform required for Faster R-CNN
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def visualize_errors(num_samples=3):
    # Pass transforms here to ensure img is a Tensor, not a PIL Image
    dataset = BCCDDataset(root=config.DATA_ROOT, transforms=get_transform())
    
    model = get_model(config.NUM_CLASSES)
    # Load weights into the architecture
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    for i in range(num_samples):
        img, target = dataset[i]
        
        # Move image to the correct device
        img_input = img.to(config.DEVICE)
        
        with torch.no_grad():
            # Model expects a list of tensors
            prediction = model([img_input])

        # Move back to CPU and change dimensions for plotting (C, H, W) -> (H, W, C)
        plot_img = img.permute(1, 2, 0).cpu().numpy()
        
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(plot_img)

        # Draw Ground Truth (Green)
        for box in target['boxes']:
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                   linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        # Draw Predictions (Red)
        res = prediction[0]
        for box, score in zip(res['boxes'], res['scores']):
            if score > 0.5: # Use confidence threshold
                box = box.cpu().numpy()
                rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                       linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        
        plt.title(f"Sample {i}: Green=Ground Truth, Red=Prediction")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    visualize_errors()