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
            predicted_index = int(predicted.item())

            # Get all probabilities
            all_probs = probabilities.cpu().numpy()[0]
            class_probs = {self.class_names[i]: float(all_probs[i])
                           for i in range(len(all_probs))}

            return predicted_class, confidence, class_probs, predicted_index

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
    
    def diagnosis_to_index(self, diagnosis_str):
        """Map a diagnosis string (from csv_diagnosis) to a class index.
        Returns None if it cannot be mapped.
        """
        if diagnosis_str is None:
            return None
        diagnosis = diagnosis_str.lower()
        # Check common keywords
        if 'melanocytic' in diagnosis or 'nevus' in diagnosis or diagnosis.startswith('nv'):
            return 0
        if 'melanoma' in diagnosis or diagnosis.startswith('mel'):
            return 1
        if 'benign keratosis' in diagnosis or 'keratosis' in diagnosis or diagnosis.startswith('bkl'):
            return 2
        if 'basal cell' in diagnosis or diagnosis.startswith('bcc'):
            return 3
        if 'actinic' in diagnosis or diagnosis.startswith('akiec'):
            return 4
        if 'vasc' in diagnosis or 'vascular' in diagnosis or diagnosis.startswith('vasc'):
            return 5
        if 'dermatofibroma' in diagnosis or diagnosis.startswith('df'):
            return 6
        if 'scar' in diagnosis or diagnosis.startswith('scr'):
            return 7
        return None

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

    num_classes = config.num_classes
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int) # Confusion matrix: rows = actual, cols = predicted
    evaluated = 0 

    if os.path.exists(test_dir):
        for image_name in os.listdir(test_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(test_dir, image_name)
                print(f"\nTesting: {os.path.basename(image_path)}")
                total += 1
                
                class_name, confidence, all_probs, predicted_index = tester.predict(image_path)
                actual_diagnosis = tester.csv_diagnosis(csv_data, image_name)
                actual_index = tester.diagnosis_to_index(actual_diagnosis)

                print(f"Prediction: {class_name}")
                print(f"****Actual diagnosis: {actual_diagnosis}****")

                if actual_index is None:
                    issues_with += f"UNMAPPED Actual {actual_diagnosis} for image {image_name} - Predicted {class_name}\n"
                else:
                    evaluated += 1
                    confusion_matrix[actual_index, predicted_index] += 1
                    if actual_index == predicted_index:
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
    print(f"\nTotal images scanned: {total} - Evaluated (with labels): {evaluated} - Correct predictions: {correct}")
    
    accuracy = (correct / evaluated * 100) if evaluated > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.2f}%")

    true_pos = np.diag(confusion_matrix).astype(float)
    false_pos = confusion_matrix.sum(axis=0).astype(float) - true_pos
    false_negative = confusion_matrix.sum(axis=1).astype(float) - true_pos


    with np.errstate(divide='ignore', invalid='ignore'):
        precision_per_class = np.divide(true_pos, true_pos + false_pos, out=np.zeros_like(true_pos), where=(true_pos + false_pos) != 0)
        recall_per_class = np.divide(true_pos, true_pos + false_negative, out=np.zeros_like(true_pos), where=(true_pos + false_negative) != 0)

    macro_precision = float(np.nanmean(precision_per_class)) if precision_per_class.size > 0 else 0.0
    macro_recall = float(np.nanmean(recall_per_class)) if recall_per_class.size > 0 else 0.0

    total_truepos = true_pos.sum()
    total_falsepos = false_pos.sum()
    total_falseneg = false_negative.sum()
    micro_precision = float(total_truepos / (total_truepos + total_falsepos)) if (total_truepos + total_falsepos) > 0 else 0.0
    micro_recall = float(total_truepos / (total_truepos + total_falseneg)) if (total_truepos + total_falseneg) > 0 else 0.0

    print('\nPer-class precision and recall:')
    for i in range(num_classes):
        cls_name = tester.class_names.get(i, f'class_{i}')
        print(f"  {i} - {cls_name}: Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}")

    print(f"\nMacro (Average) Precision: {macro_precision:.4f}")
    print(f"  On average across all classes, about {macro_precision*100:.2f}% of the samples the model predicted as belonging to a specific class were actually correct")
    print(f"\nMacro (Average) Recall: {macro_recall:.4f}")
    print(f"  On average across all classes, the model correctly identified about {macro_recall*100:.2f}% of the true samples belonging to each class")
    print(f"\nMicro Precision: {micro_precision:.4f}")
    print(f"  {micro_precision*100:.2f}% of all predicted positives (across all classes) were correct")
    print(f"\nMicro Recall: {micro_recall:.4f}")
    print(f"  The model correctly identified {micro_recall*100:.2f}% of all actual positive cases")

    print(f"\nPrecision = Of all the samples the model predicted as this class, how many were correct?")
    print(f"Recall = Of all the samples that actually belong to this class, how many did the model get correct")
    print(f"Micro Precision and Recall consider overall totals across all classes. Lower number means poor performance with rare classes")

    tester.append_to_txt(f"\nTotal images scanned: {total} - Evaluated (with mapped labels): {evaluated} - Correct predictions: {correct} - Accuracy: {accuracy:.2f}%")
    tester.append_to_txt("=" * 40)
    tester.append_to_txt("Per-class precision and recall:")
    for i in range(num_classes):
        cls_name = tester.class_names.get(i, f'class_{i}')
        tester.append_to_txt(f"{i} - {cls_name}: Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}")
    tester.append_to_txt(f"Macro Precision: {macro_precision:.4f}")
    tester.append_to_txt(f"Macro Recall: {macro_recall:.4f}")
    tester.append_to_txt(f"Micro Precision: {micro_precision:.4f}")
    tester.append_to_txt(f"Micro Recall: {micro_recall:.4f}")
    tester.append_to_txt("=" * 40)
    tester.append_to_txt(f"ISSUES WITH: \n{issues_with}")
    tester.append_to_txt("=" * 40)

if __name__ == "__main__":
    main()