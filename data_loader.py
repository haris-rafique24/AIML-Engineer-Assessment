import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset

class BCCDDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # Load all image files, sorting them to ensure alignment with annotations
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "Annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        ann_path = os.path.join(self.root, "Annotations", self.annotations[idx])
        img = Image.open(img_path).convert("RGB")
        
        # Parse XML (Pascal VOC format)
        tree = ET.parse(ann_path)
        root = tree.getroot()
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # --- Validation Check ---
            # PyTorch models require xmax > xmin and ymax > ymin to calculate loss.
            if xmax > xmin and ymax > ymin:
                label = obj.find('name').text
                # Map labels to integers (Example: RBC=1, WBC=2, Platelets=3)
                label_map = {"RBC": 1, "WBC": 2, "Platelets": 3}
                labels.append(label_map[label])
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                # Log the issue so you know which file has a "dot" annotation
                print(f"Warning: Skipping invalid box {xmin, ymin, xmax, ymax} in {self.annotations[idx]}")

        # Safety net: If an image has NO valid boxes, move to the next image
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.imgs))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)