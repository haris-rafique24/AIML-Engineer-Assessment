import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    # Load a stronger model pre-trained on COCO (i.e R-CNN ResNet50 FPN)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    #baseline-model: Faster R-CNN model with a MobileNetV3-Large FPN backbone(uncomment this to run with the baseline model)
    #model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    
    # Replace the classifier head for our 3 blood cell classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model