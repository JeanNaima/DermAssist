import os

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from pathlib import Path

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
from PIL import Image

from config import config

# Import your model
from models.vit_model import create_model

MODEL_PATH = Path(__file__).resolve().parent / "saved_models" / "best_model4.5.pth"
print("Model path:", MODEL_PATH)
print("Exists:", MODEL_PATH.exists())


class SimpleTester:
    def __init__(self, model_path: Path | str = MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model
        self.model = create_model(config.num_classes)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        else:
            # print(f"Model file not found: {model_path}")
            return

        # Transform (same as training)
        self.transform = A.Compose(
            [
                A.Resize(config.image_size, config.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        # Class names
        self.class_names = {
            0: "nv (Melanocytic nevi)",
            1: "mel (Melanoma)",
            2: "bkl (Benign keratosis)",
            3: "bcc (Basal cell carcinoma)",
            4: "akiec (Actinic keratoses)",
            5: "vasc (Vascular lesions)",
            6: "df (Dermatofibroma)",
            7: "scr (Scar)",  # Optional - only in some datasets
        }

    def predict(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = self.transform(image=image)["image"]
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension

        # Predict
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = self.class_names[predicted.item()]
        confidence = confidence.item()
        predicted_index = int(predicted.item())

        # Get all probabilities
        all_probs = probabilities.cpu().numpy()[0]
        class_probs = {
            self.class_names[i]: float(all_probs[i]) for i in range(len(all_probs))
        }

        return predicted_class, confidence, class_probs, predicted_index

    # This will return the diagnosis from the CSV data given an image name
    # change the return varable to the appropriate column index as needed
    def csv_diagnosis(self, csv_data, image_name):
        for row in csv_data:
            if row[0] in image_name:
                diagnosis = row[9].lower()

                if "melanocytic nevi" in diagnosis or "nevus" in diagnosis:
                    return "nv (melanocytic nevi, nevus)"
                elif "melanoma" in diagnosis:
                    return "mel (melanoma)"
                elif (
                    "benign keratosis" in diagnosis
                    or "solar lentigo" in diagnosis
                    or "seborrheic keratosis" in diagnosis
                    or "lichen planus-like keratosis" in diagnosis
                ):
                    return "bkl (benign keratosis, solar lentigo, seborrheic keratosis, lichen planus-like keratosis)"
                elif "basal cell carcinoma" in diagnosis:
                    return "bcc (basal cell carcinoma)"
                elif "actinic keratoses" in diagnosis:
                    return "akiec (actinic keratoses)"
                elif "vasc" in diagnosis:
                    return "vasc (vascular lesions)"
                elif "dermatofibroma" in diagnosis:
                    return "df (dermatofibroma)"
                elif "scar" in diagnosis:
                    return "scr (scar)"
                else:
                    return "Not Classified"
        return "Not Classified"

    def diagnosis_to_index(self, diagnosis_str):
        """Map a diagnosis string (from csv_diagnosis) to a class index.
        Returns None if it cannot be mapped.
        """
        if diagnosis_str is None:
            return None
        diagnosis = diagnosis_str.lower()
        # Check common keywords
        if (
            "melanocytic" in diagnosis
            or "nevus" in diagnosis
            or diagnosis.startswith("nv")
        ):
            return 0
        if "melanoma" in diagnosis or diagnosis.startswith("mel"):
            return 1
        if (
            "benign keratosis" in diagnosis
            or "keratosis" in diagnosis
            or diagnosis.startswith("bkl")
        ):
            return 2
        if "basal cell" in diagnosis or diagnosis.startswith("bcc"):
            return 3
        if "actinic" in diagnosis or diagnosis.startswith("akiec"):
            return 4
        if (
            "vasc" in diagnosis
            or "vascular" in diagnosis
            or diagnosis.startswith("vasc")
        ):
            return 5
        if "dermatofibroma" in diagnosis or diagnosis.startswith("df"):
            return 6
        if "scar" in diagnosis or diagnosis.startswith("scr"):
            return 7
        return None
