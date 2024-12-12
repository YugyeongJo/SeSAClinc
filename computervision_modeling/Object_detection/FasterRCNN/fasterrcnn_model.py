# fasterrcnn_model.py
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class FRCNNObjectDetector(FasterRCNN):
    def __init__(self, num_classes=3, **kwargs):
        # ResNet50 Backbone + FPN 사용
        backbone = resnet_fpn_backbone('resnet50', pretrained=False)
        # Faster R-CNN 모델을 초기화
        super(FRCNNObjectDetector, self).__init__(backbone, num_classes, **kwargs)