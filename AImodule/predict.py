import torch
from PIL import Image
import torch.nn.functional as F
from models.vit_model import create_model
from config import config
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


class SkinLesionPredictor:
    def __init__(self, model_path):
        self.config = config
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

        print("Model loaded successfully!")

    def predict(self, image_path):
        if not os.path.exists(image_path):
            return f"Error: Image not found at {image_path}", 0.0

        try:
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

            # Get all probabilities
            all_probs = probabilities.cpu().numpy()[0]
            class_probs = {self.idx_to_class[i]: float(all_probs[i])
                           for i in range(len(all_probs))}

            return predicted_class, confidence, class_probs

        except Exception as e:
            return f"Error during prediction: {str(e)}", 0.0, {}


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = SkinLesionPredictor("saved_models/best_model.pth")

    # Test prediction
    test_image = "data/images/ISIC_0034524.jpg"  # Replace with actual image
    if os.path.exists(test_image):
        class_name, confidence, all_probs = predictor.predict(test_image)
        print(f"Predicted: {class_name}")
        print(f"Confidence: {confidence:.4f}")
        print("All probabilities:")
        for cls, prob in all_probs.items():
            print(f"  {cls}: {prob:.4f}")
    else:
        print(f"Test image not found: {test_image}")