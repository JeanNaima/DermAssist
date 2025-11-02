import torch


class Config:
    def __init__(self):
        # Data
        self.csv_path = "data/ISIC2018_Task3_Test_GroundTruth.csv"
        self.image_dir = "data/images/"
        self.image_size = 224
        self.batch_size = 16

        # Model
        self.model_name = "google/vit-base-patch16-224"
        self.num_classes = 7

        # Training
        self.epochs = 10
        self.learning_rate = 2e-5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Classes mapping
        self.class_names = {
            'nv': 0,  # Melanocytic nevi
            'mel': 1,  # Melanoma
            'bkl': 2,  # Benign keratosis-like lesions
            'bcc': 3,  # Basal cell carcinoma
            'akiec': 4,  # Actinic keratoses
            'vasc': 5,  # Vascular lesions
            'df': 6  # Dermatofibroma
        }


# Create a global config instance
config = Config()