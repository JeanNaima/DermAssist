import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

class ViTSkinLesionClassifier(nn.Module):
    def __init__(self, num_classes=8, model_name="google/vit-base-patch16-224"):
        super().__init__()
        
        # Load pre-trained ViT model
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
    def forward(self, x):
        return self.vit(x).logits

def create_model(num_classes=8):
    return ViTSkinLesionClassifier(num_classes=num_classes)