import torch

class Config:
    # Data
    csv_path = "data/ISIC2018_Task3_Test_GroundTruth.csv"
    image_dir = "data/images/"
    
    image_size = 224
    batch_size = 32
    
    # Model
    model_name = "google/vit-base-patch16-224"
    num_classes = 7  # Based on your CSV: nv, mel, bkl, bcc, akiec, vasc, df
    
    # Training
    epochs = 20
    learning_rate = 2e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Classes mapping (from your CSV analysis)
    class_names = {
        'nv': 0,      # Melanocytic nevi
        'mel': 1,     # Melanoma
        'bkl': 2,     # Benign keratosis-like lesions
        'bcc': 3,     # Basal cell carcinoma
        'akiec': 4,   # Actinic keratoses
        'vasc': 5,    # Vascular lesions
        'df': 6       # Dermatofibroma
    }