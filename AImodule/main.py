import torch
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

# Load pretrained ViT
model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
model.eval()

# Random image (for test)
x = torch.randn(1, 3, 224, 224)
y = model(x)

print("Output shape:", y.shape)