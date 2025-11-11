# test_model.py
import torch
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
from PIL import Image
import torch.nn.functional as F
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import csv

# Import your model
from models.vit_model import create_model
from config import config

class SimpleTester:
    def __init__(self, model_path="saved_models/best_model3.0.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model
        self.model = create_model(config.num_classes)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        else:
            print(f"Model file not found: {model_path}")
            return

        # Transform (same as training)
        self.transform = A.Compose([
            A.Resize(config.image_size, config.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # Class names
        self.class_names = {
            0: 'nv (Melanocytic nevi)',
            1: 'mel (Melanoma)',
            2: 'bkl (Benign keratosis)',
            3: 'bcc (Basal cell carcinoma)',
            4: 'akiec (Actinic keratoses)',
            5: 'vasc (Vascular lesions)',
            6: 'df (Dermatofibroma)',
            7:   'scr (Scar)'  # Optional - only in some datasets
        }

    def predict(self, image_path):
        if not os.path.exists(image_path):
            return f"Image not found: {image_path}", 0.0, {}

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image = self.transform(image=image)['image']
            image = image.unsqueeze(0).to(self.device)  # Add batch dimension

            # Predict
            with torch.no_grad():
                outputs = self.model(image)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            predicted_class = self.class_names[predicted.item()]
            confidence = confidence.item()

            # Get all probabilities
            all_probs = probabilities.cpu().numpy()[0]
            class_probs = {self.class_names[i]: float(all_probs[i])
                           for i in range(len(all_probs))}

            return predicted_class, confidence, class_probs

        except Exception as e:
            return f"Error: {str(e)}", 0.0, {}

    # This will return the diagnosis from the CSV data given an image name
    # change the return varable to the appropriate column index as needed
    def csv_diagnosis(self, csv_data, image_name):
        for row in csv_data:
            if row[0] in image_name:
                diagnosis = row[9].lower()

                if 'melanocytic nevi' in diagnosis or 'nevus' in diagnosis:
                    return 'nv (melanocytic nevi, nevus)'
                elif 'melanoma' in diagnosis:
                    return 'mel (melanoma)'
                elif 'benign keratosis' in diagnosis or 'solar lentigo' in diagnosis or 'seborrheic keratosis' in diagnosis or 'lichen planus-like keratosis' in diagnosis:
                    return 'bkl (benign keratosis, solar lentigo, seborrheic keratosis, lichen planus-like keratosis)'
                elif 'basal cell carcinoma' in diagnosis:
                    return 'bcc (basal cell carcinoma)'
                elif 'actinic keratoses' in diagnosis:
                    return 'akiec (actinic keratoses)'
                elif 'vasc' in diagnosis:
                    return 'vasc (vascular lesions)'
                elif 'dermatofibroma' in diagnosis:
                    return 'df (dermatofibroma)'
                elif 'scar' in diagnosis:
                    return 'scr (scar)'
                else:
                    return "Not Classified"
        return "Not Classified"
    
    def append_to_txt(self, text):
        with open("test_output.txt", "a") as f:
            f.write(text + "\n")

def main():
    print("Testing Skin Lesion Model")
    print("=" * 40)

    # Initialize tester
    tester = SimpleTester()

    # Test on images from your dataset
    test_images = [
        "data/image/ISIC_0053453.jpg",  # Replace with actual images you have
        "data/image/ISIC_0034525.jpg",
        "data/image/ISIC_0034526.jpg"
    ]

    test_dir = "data/images/Challenge2018/"  # Directory with test images

    csv_file = "data/challenge-2018.csv" #This is the csv file path that has all the diagnosis info
    csv_data = []

    with open(csv_file, newline='') as datasetfile:
        reader = csv.reader(datasetfile)
        for row in reader:
            csv_data.append(row)
    csv_data = csv_data[1:]  # Skip header that is just text

    total = 0
    correct = 0

    issues_with = ""

    if os.path.exists(test_dir):
        for image_name in os.listdir(test_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(test_dir, image_name)
                print(f"\nTesting: {os.path.basename(image_path)}")
                total += 1

                class_name, confidence, all_probs = tester.predict(image_path)
                actual_diagnosis = tester.csv_diagnosis(csv_data, image_name)

                print(f"Prediction: {class_name}")
                print(f"****Actual diagnosis: {actual_diagnosis}****")

                if actual_diagnosis.split()[0] in class_name.split(): 
                    correct += 1
                else:
                    issues_with += f"Predicted {class_name}, Actual {actual_diagnosis}\n"
                    
                print(f"Confidence: {confidence:.4f}")
                print("All probabilities:")
                for cls, prob in all_probs.items():
                    print(f"  {cls}: {prob:.4f}")
                print("-" * 30)
                tester.append_to_txt(f"Image: {image_name}, Predicted: {class_name}, *Actual: {actual_diagnosis}*, Confidence: {confidence:.4f}")

    else:
        print(f"⚠Test directory not found: {test_dir}")

    print(f"\nModel testing complete!")
    print(f"\nTotal images tested: {total} - Correct predictions: {correct}")
    print(f"\nAccuracy: {(correct/total*100):.2f}%")

    tester.append_to_txt(f"\nTotal images tested: {total} - Correct predictions: {correct} - Accuracy: {(correct/total*100):.2f}%")
    tester.append_to_txt("=" * 40)
    tester.append_to_txt(f"ISSUES WITH: \n{issues_with}")
    tester.append_to_txt("=" * 40)

if __name__ == "__main__":
    main()