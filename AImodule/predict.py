import torch
from PIL import Image
import torch.nn.functional as F
from models.vit_model import create_model
from config import Config
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SkinLesionPredictor:
    def __init__(self, model_path):
        self.config = Config()
        self.device = self.config.device
        
        # Load model
        self.model = create_model(self.config.num_classes)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = A.Compose([
            A.Resize(self.config.image_size, self.config.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                      std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Reverse class mapping
        self.idx_to_class = {v: k for k, v in self.config.class_names.items()}
    
    def predict(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = self.transform(image=image)['image']
        image = image.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        predicted_class = self.idx_to_class[predicted.item()]
        confidence = confidence.item()
        
        return predicted_class, confidence

# Example usage
if __name__ == "__main__":
    predictor = SkinLesionPredictor("models/saved_models/best_model.pth")
    
    # Test prediction
    image_path = "path_to_your_test_image.jpg"
    class_name, confidence = predictor.predict(image_path)
    print(f"Predicted: {class_name} with confidence: {confidence:.4f}")