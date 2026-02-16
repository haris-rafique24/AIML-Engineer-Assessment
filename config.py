import torch

# Dataset Paths
DATA_ROOT = "BCCD_Dataset-master/BCCD_Dataset-master/BCCD"
MODEL_SAVE_PATH = "model_weights.pth"

# Hyperparameters
NUM_CLASSES = 4  # 3 classes (RBC, WBC, Platelets) + 1 Background
BATCH_SIZE = 4
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 10

# Environment
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')