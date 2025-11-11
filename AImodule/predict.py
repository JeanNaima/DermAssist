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
import csv

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

    # This will return the diagnosis from the CSV data given an image name
    # change the return varable to the appropriate column index as needed
    def csv_diagnosis(self, csv_data, image_name):
        for row in csv_data:
            if row[0] == image_name:
                return row[9]
        return "Unknown"

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = SkinLesionPredictor("saved_models/best_model2.0.pth")

    # Test prediction
    test_image = "data/images/ISIC_0034524.jpg"  # Replace with actual image
    test_dir = "AImodule/data/images/Challenge2018/"  # Directory with test images

    csv_file = "AImodule/data/challenge-2018.csv" #This is the csv file path that has all the diagnosis info
    csv_data = []

    with open(csv_file, newline='') as datasetfile:
        reader = csv.reader(datasetfile)
        for row in reader:
            csv_data.append(row)
    csv_data = csv_data[1:]  # Skip header that is just text

    if os.path.exists(test_dir):
        for img_name in os.listdir(test_dir):
            img_path = os.path.join(test_dir, img_name)
            class_name, confidence, all_probs = predictor.predict(img_path)
            print(f"Image: {img_name}")
            print(f"Predicted: {class_name}")
            print(f"Actual diagnosis: {predictor.csv_diagnosis(csv_data, img_name)}") 
            print(f"Confidence: {confidence:.4f}")
            print("All probabilities:")
            for cls, prob in all_probs.items():
                print(f"  {cls}: {prob:.4f}")
            print("-" * 30)
    else:
        print(f"Test directory not found: {test_dir}")